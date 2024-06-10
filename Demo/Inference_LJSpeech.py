import os
from pathlib import Path
import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import argparse
from nltk.tokenize import word_tokenize
import soundfile as sf
from shutil import copyfile

from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()

from infer_utils import length_to_mask
from infer_utils import compute_style_from_path as compute_style

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

def inference(text, noise, ref_s=None, diffusion_steps=5, embedding_scale=1, alpha=-1, beta=-1):
    text = text.strip()
    text = text.replace('"', '')
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 

        if ref_s == None:
            s_pred = sampler(noise, 
                embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
                embedding_scale=embedding_scale).squeeze(0)
        else:
            s_pred = sampler(noise, 
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s, # reference from the same speaker as the embedding
                num_steps=diffusion_steps).squeeze(1)      

        ref = s_pred[:, :128]
        if alpha >= 0 and alpha <= 1:
            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]

        s = s_pred[:, 128:]
        if beta >=0 and beta <= 1:
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_dur[-1] += 5

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        out2 = out.squeeze().cpu().numpy()
        # out2 = out.squeeze().cpu().detach().numpy()
        
    return out2     

def run_infer(text, noise, ref_s=None, diffusion_steps=5, embedding_scale=1, alpha=-1, beta=-1):

    start = time.time()
    wav = inference(text, noise, ref_s=ref_s, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale, alpha=alpha, beta=beta)
    end = time.time()
    duration_proc = end - start
    duration_out = len(wav) / 24000
    rtf = duration_proc / duration_out

    return wav, rtf, (duration_proc, duration_out)

def parse_args():
    usage = 'usage: inference demo for LJSpeech'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--config-path', type=str, help='config path')
    parser.add_argument('--model-path', type=str, help='model path')
    parser.add_argument('--output-path', type=str, help='output path')
    parser.add_argument('--device', type=str, default='cpu', help='gpu/cpu device')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # work_path = os.getcwd() # e.g., '/home/users/zge/code/repo/style-tts2'
    # args.config_path = os.path.join(work_path, 'Models', 'LJSpeech', 'Original', 'config.yml')
    # args.model_path = os.path.join(work_path, 'Models', 'LJSpeech', 'Original', 'epochs_2nd_00100.pth')
    # args.output_path = os.path.join(work_path, 'Outputs', 'Demo', 'LJSpeech')
    # args.device = 'cuda:0' # 'cuda', 'cuda:x', or 'cpu'

    # set and create output dir (if needed)
    set_path(args.output_path)

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
    hostname = get_hostname()
    print('computer: {}'.format(hostname))

    # get gpu info
    if 'cuda' in device:
        parts = device.split(':')
        if len(parts) == 1:
            device_id = 0
        else:
            device_id = int(parts[1])    
        gpu_info = get_gpu_info(device_id)
    else:
        gpu_info = ''
    print('GPU info: {} @ {}'.format(gpu_info, hostname))

    ## Load models

    import phonemizer
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us', preserve_punctuation=True,  with_stress=True)

    # load config
    config = yaml.safe_load(open(args.config_path))

    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    # load BERT model
    from Utils.PLBERT.util import load_plbert
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    model_params = recursive_munch(config['model_params'])
    model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    params_whole = torch.load(args.model_path, map_location='cpu')
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

    from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )

    ## Synthesize speech

    # synthesize a text
    text = ''' StyleTTS 2 is a text-to-speech model that leverages style diffusion and adversarial training with large speech language models to achieve human-level text-to-speech synthesis. '''

    ## Basic synthesis (5 diffusion steps)

    exp_id = 0
    output_path = os.path.join(args.output_path, 'exp-{:02d}'.format(exp_id))
    os.makedirs(output_path, exist_ok=True)
    print('output path for exp {}: {}'.format(exp_id, output_path))

    for diffusion_steps in [5, 10]:
        for embedding_scale in  [1, 2]:
            noise = torch.randn(1,1,256).to(device)
            # noise = torch.randn((1, 256)).unsqueeze(1).to(device)
            wav, rtf, (duration_proc, duration_out) = run_infer(text, noise, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
            print('basic, diffusion steps: {}, embedding scale: {}'.format(diffusion_steps, embedding_scale))
            print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

            # import IPython.display as ipd
            # display(ipd.Audio(wav, rate=24000))
            output_filename = 'basic-{}-{}.wav'.format(diffusion_steps, embedding_scale)    
            output_filepath = os.path.join(output_path, output_filename)
            sf.write(output_filepath, wav, 24000)
            print('wrote output file: {}'.format(output_filepath))

    ## Basic synthesis (testing voice cloning)

    exp_id = 1
    output_path = os.path.join(args.output_path, 'exp-{:02d}'.format(exp_id))
    os.makedirs(output_path, exist_ok=True)
    print('output path for exp {}: {}'.format(exp_id, output_path))

    reference_dicts = {}
    reference_dicts['696_92939'] = "Demo/reference_audio/696_92939_000016_000006.wav"
    reference_dicts['1789_142896'] = "Demo/reference_audio/1789_142896_000022_000005.wav"
    nsamples = len(reference_dicts)

    for i, ref_path in enumerate(reference_dicts.values()):
        ref_id = os.path.splitext(os.path.basename(ref_path))[0]
        ref_id = '_'.join(ref_id.split('_')[:-2])
        reference_filename = 'reference-{}-{}.wav'.format(i, ref_id)
        reference_filepath = os.path.join(output_path, reference_filename)
        copyfile(ref_path, reference_filepath)
        print('copied reference file: {} -> {}'.format(ref_path, reference_filepath))

    rtfs = [0 for _ in range(nsamples)]
    diffusion_steps = 10
    embedding_scale = 2
    alpha = 0.0
    beta = 0.1
    noise = torch.randn(1,1,256).to(device)
    for i, ref_path in enumerate(reference_dicts.values()):

        print('({}/{}) reference: {}'.format(i+1, nsamples, ref_path))
        ref_path = os.path.join(os.getcwd(), ref_path)
        ref_s = compute_style(model, ref_path)

        # get ref wav duration
        ref_wav, _ = librosa.load(ref_path, sr=24000)
        duration_ref = len(ref_wav) / 24000

        # get syn wav and rtf
        wav, rtf, (duration_proc, duration_out) = run_infer(text, noise, ref_s, diffusion_steps, embedding_scale, alpha=alpha, beta=beta)
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

        # write output syn wav file
        ref_id = os.path.splitext(os.path.basename(ref_path))[0]
        ref_id = '_'.join(ref_id.split('_')[:-2])
        output_filename = 'vc-{}-{}-{}-{}-{}-{}.wav'.format(i, ref_id, diffusion_steps, embedding_scale, alpha, beta)
        output_filepath = os.path.join(output_path, output_filename)
        sf.write(output_filepath, wav, 24000)

    ## Speech expressiveness

    texts = {}
    texts['Happy'] = "We are happy to invite you to join us on a journey to the past, where we will visit the most amazing monuments ever built by human hands."
    texts['Sad'] = "I am sorry to say that we have suffered a severe setback in our efforts to restore prosperity and confidence."
    texts['Angry'] = "The field of astronomy is a joke! Its theories are based on flawed observations and biased interpretations!"
    texts['Surprised'] = "I can't believe it! You mean to tell me that you have discovered a new species of bacteria in this pond?"
    nsamples = len(texts)

    ### With embedding scale 1

    rtfs = [0 for _ in range(nsamples)]
    diffusion_steps = 10
    embedding_scale = 1
    for i, (emo, v) in enumerate(texts.items()):
        noise = torch.randn(1,1,256).to(device)
        wav, rtf, (duration_proc, duration_out) = run_infer(v, noise, diffusion_steps, embedding_scale)
        print('expressive, diffusion steps: {}, embedding scale: {}'.format(diffusion_steps, embedding_scale))
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))
        # display(ipd.Audio(wav, rate=24000, normalize=False))

        # write wav to output file 
        output_filename = '{}-{}-{}.wav'.format(emo.lower(), diffusion_steps, embedding_scale)
        output_filepath = os.path.join(args.output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        rtfs[i] = rtf

    rtf_avg = np.mean(rtfs)
    print('average RTF (expressive, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))    

    # Happy, RTF: 0.0223 (0.2044 / 9.1750)
    # Sad, RTF: 0.0245 (0.1690 / 6.9000)
    # Angry, RTF: 0.0232 (0.1718 / 7.4000)
    # Surprised, RTF: 0.0236 (0.1691 / 7.1750)

    ### With embedding scale 2

    rtfs = [0 for _ in range(nsamples)]
    diffusion_steps = 10
    embedding_scale = 2
    for i, (emo, v) in enumerate(texts.items()):
        noise = torch.randn(1,1,256).to(device)
        wav, rtf, (duration_proc, duration_out) = run_infer(v, noise, diffusion_steps, embedding_scale)
        print('expressive, diffusion steps: {}, embedding scale: {}'.format(diffusion_steps, embedding_scale))
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))
        # display(ipd.Audio(wav, rate=24000, normalize=False))

        # write wav to output file 
        output_filename = '{}-{}-{}.wav'.format(emo.lower(), diffusion_steps, embedding_scale)
        output_filepath = os.path.join(args.output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        rtfs[i] = rtf

    rtf_avg = np.mean(rtfs)
    print('average RTF (expressive, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))     
