# import general packages
import os
from pathlib import Path
import torch
import yaml
import time
import json
import numpy as np
import argparse

# import specific packages
import librosa
import soundfile as sf
import phonemizer
from nltk.tokenize import word_tokenize

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

# import local packages
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
import utils
import models
from text_utils import TextCleaner
textclenaer = TextCleaner()
from infer_utils import length_to_mask
from infer_utils import compute_style_from_path as compute_style

def inference(text, ref_s, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 

        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device), 
                                          embedding=bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)


        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)


        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, 
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))
    
        
    return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later 

def run_infer(text, ref_s, diffusion_steps, embedding_scale, alpha=0.3, beta=0.7):

    start = time.time()
    wav = inference(text, ref_s, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
    end = time.time()
    duration_proc = end - start
    duration_out = len(wav) / 24000
    rtf = duration_proc / duration_out

    return wav, rtf, (duration_proc, duration_out)

def parse_args():
    usage = 'usage: measure RTF through multiple runs of the regular inference'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--model-path', type=str, help='model path')
    parser.add_argument('--model-name', type=str, help='model name')
    parser.add_argument('--output-path', type=str, help='output path')
    parser.add_argument('--run-id', type=str, help='prefix to distinguish different runs')
    parser.add_argument('--device', type=str, default='cpu', help='gpu/cup device')
    parser.add_argument('--manifest-file', type=str, help='manifest file with ref audio wav file and texts')
    parser.add_argument('--diffusion-steps', type=int, default=5, help='diffusion steps')
    parser.add_argument('--embedding-scale', type=float, default=1, help='embedding scale')
    parser.add_argument('--alpha', type=float, default=0.3, help='timbre parameter')
    parser.add_argument('--beta', type=float, default=0.7, help='prosody parameter')
    parser.add_argument('--num-reps', type=int, default=10, help='number of rep runs')
    parser.add_argument('--num-warmup', type=int, default=3, help='number of warmup runs')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()
    # work_path = os.getcwd() # e.g., '/home/users/zge/code/repo/style-tts2'
    # args.model_path = os.path.join(work_path, 'Models', 'LibriTTS')
    # args.model_name = 'epochs_2nd_00020.pth'
    # args.output_path = os.path.join(work_path, 'Outputs', 'RTF')
    # args.run_id = 'exp1'
    # args.manifest_file = os.path.join(args.output_path, 'manifest.txt')
    # args.device = 'cuda:1'
    # args.diffusion_steps = 5
    # args.embedding_scale = 1
    # args.alpha = 0.3
    # args.beta = 0.7
    # args.num_reps = 10
    # args.num_warmup = 3

    # set and create output dir (if needed)
    utils.set_path(args.output_path)

    # set GPU/CPU device
    if 'cuda' in args.device:
        if not torch.cuda.is_available():
            print ('device set to {}, but cuda is not available, so swithed to cpu'.format(args.device))
            device = 'cpu'
        else:
            device = args.device
    else:
        device = 'cpu'
    print('device: {}'.format(device))

    # get hostname
    hostname = utils.get_hostname()
    print('computer: {}'.format(hostname))

    # get gpu info
    if 'cuda' in device:
        parts = device.split(':')
        if len(parts) == 1:
            device_id = 0
        else:
            device_id = int(parts[1])    
        gpu_info = utils.get_gpu_info(device_id)
    else:
        gpu_info = ''
    print(gpu_info)

    # load phonemizer
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us', preserve_punctuation=True,  with_stress=True)

    # load config
    config_path = os.path.join(args.model_path, 'config.yml')
    config = yaml.safe_load(open(config_path))

    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = models.load_ASR_models(ASR_path, ASR_config)

    # load BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = models.load_F0_models(F0_path)

    model_params = utils.recursive_munch(config['model_params'])
    model = models.build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    model_file = os.path.join(args.model_path, args.model_name)
    params_whole = torch.load(model_file, map_location='cpu')
    params = params_whole['net']

    for key in model:
        if key in params:
            print('%s loaded' % key)
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                # load params
                model[key].load_state_dict(new_state_dict, strict=False)
    #             except:
    #                 _load(params[key], model[key])
    _ = [model[key].eval() for key in model]

    sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
    )

    # read in the reference audios and texts
    reference_dicts = utils.read_manifest(args.manifest_file)
    num_samples = len(reference_dicts)
    print('# of samples: {}'.format(num_samples))

    # run inference
    rtfs_avg = [0 for _ in range(args.num_reps)]
    for n in range(args.num_reps):
        rtfs = [0 for _ in range(num_samples)]
        for i, v in enumerate(reference_dicts.values()):

            ref_path, text = v
            ref_s = compute_style(model, ref_path, device=device)

            ref_wav, _ = librosa.load(ref_path, sr=24000)
            duration_ref = len(ref_wav) / 24000

            wav, rtf, (duration_proc, duration_out) = run_infer(text, ref_s,
                args.diffusion_steps, args.embedding_scale, args.alpha, args.beta)
            print("Run {}/{}, Sample {}/{}, RTF: {:.4f} ({:.4f} / {:.4f})".format(
                n+1, args.num_reps, i+1, num_samples, rtf, duration_proc, duration_out))

            output_name = '{}-{}-{}-{}-{}-{}-{}.wav'.format(args.run_id, n, i, 
                args.diffusion_steps, args.embedding_scale, args.alpha, args.beta)
            output_file = os.path.join(args.output_path, output_name)
            sf.write(output_file, wav, 24000)
            print('wrote output file: {}'.format(output_file))

            meta = {'syn-wav': output_file, 'text': text, 'ref-wav': ref_path, 'dur-ref': duration_ref,
                'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': args.diffusion_steps,
                'embedding-scale': args.embedding_scale, 'alpha': args.alpha, 'beta': args.beta,
                'hostname': hostname, 'gpu': gpu_info, 'run-id': args.run_id}
            output_jsonfile = os.path.join(args.output_path, output_name.replace('.wav', '.json'))
            with open(output_jsonfile, 'w') as fp:
                json.dump(meta, fp, indent=2)

            rtfs[i] = rtf

        rtfs_avg[n] = np.mean(rtfs)

    rtf_avg = np.mean(rtfs_avg[args.num_warmup:])
    print('average RTF: {:.3f}'.format(rtf_avg))

    # write the log file
    logfile = os.path.join(args.output_path, '{}.log'.format(args.run_id))
    with open(logfile, 'w') as f:
        f.write('model path: {}\n'.format(args.model_path))
        f.write('manifest file: {}\n'.format(args.manifest_file))
        f.write('number of runs: {}\n'.format(args.num_reps))
        f.write('number of warmup: {}\n'.format(args.num_warmup))
        f.write('diffusion steps: {}\n'.format(args.diffusion_steps))
        f.write('embedding_scale: {}\n'.format(args.embedding_scale))
        f.write('alpha: {}\n'.format(args.alpha))
        f.write('beta: {}\n'.format(args.beta))
        f.write('hostname: {}\n'.format(hostname))
        f.write('{}\n'.format(gpu_info))
        f.write('\n')
        for i in range(args.num_reps):
            f.write('  run {}/{}: RTF {:.3f}\n'.format(i+1, args.num_reps, rtfs_avg[i]))
        f.write('\n')
        f.write('Overall RTF: {:.3f}\n'.format(rtf_avg))
    print('wrote log to: {}'.format(logfile))    

