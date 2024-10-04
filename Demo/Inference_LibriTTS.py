import os
from pathlib import Path
import torch
import glob

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
import json
import pyloudnorm as pyln
from numpy.linalg import norm
from librosa.util import normalize

# import importlib
# importlib.reload(tester)

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()

from infer_utils import length_to_mask
from infer_utils import compute_style_from_path as compute_style

from pyannote.audio import Inference
spkr_embedding = Inference("pyannote/embedding", window="whole")

def get_audio(speaker_path, meter, sample_rate=16000, mono_channel=True):

    audio, sr = sf.read(speaker_path) # load audio (shape: samples, channels)
    # assert sr == sample_rate, 'sampling rate is {} (should be {})'.format(sr, sample_rate)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    loudness = meter.integrated_loudness(audio) # measure loudness
    audio = pyln.normalize.loudness(audio, loudness, -20.0)

    if mono_channel and len(audio.shape) > 1 and audio.shape[1] > 1:
        # select the 1st channel (channel 0)
        audio = audio[:,0]

    return audio

def extract_spkr_embedding(wav, sample_rate):

     # normalize wav
    wav = normalize(wav) * 0.95
    # convert wav (np.ndarray:(nsamples,) -> torch.Tensor: (1,nsamples))
    wav = torch.FloatTensor(wav).unsqueeze(0)
    # get speaker embedding (np.ndarry: (spkr_emb_dim:512,))
    speaker_embedding = spkr_embedding({'waveform': wav, 'sample_rate': sample_rate})

    return speaker_embedding

def cos_sim(A,B):
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    return cosine

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

        s_pred_1st = s_pred[:, :128] # acustics from text
        ref_s_1st = ref_s[:, :128] # acoustic from audio
        s_pred_2nd = s_pred[:, 128:] # prosody from text
        ref_s_2nd = ref_s[:, 128:] # prosody from audio

        ref = alpha * s_pred_1st + (1 - alpha)  * ref_s_1st # acoustics
        s = beta * s_pred_2nd + (1 - beta) * ref_s_2nd # prosody

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

        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        out2 = out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later
        # out2 = out.squeeze().cpu().detach().numpy()[..., :-50] 

    return out2     

def LFinference(text, s_prev, ref_s, alpha = 0.3, beta = 0.7, t = 0.7, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    ps = ps.replace('``', '"')
    ps = ps.replace("''", '"')

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
        
        if s_prev is not None:
            # convex combination of previous and current style
            s_pred = t * s_prev + (1 - t) * s_pred
        
        s = s_pred[:, 128:]
        ref = s_pred[:, :128]
        
        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        s_pred = torch.cat([ref, s], dim=-1)

        d = model.predictor.text_encoder(d_en, 
                                         s, input_lengths, text_mask)

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

        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        out2 = out.squeeze().cpu().numpy()[..., :-100] # weird pulse at the end of the model, need to be fixed later
        
    return out2, s_pred

def STinference(text, ref_s, ref_text, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
    ref_text = ref_text.strip()
    ps = global_phonemizer.phonemize([ref_text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    ref_tokens = textclenaer(ps)
    ref_tokens.insert(0, 0)
    ref_tokens = torch.LongTensor(ref_tokens).to(device).unsqueeze(0)
    
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
        
        ref_input_lengths = torch.LongTensor([ref_tokens.shape[-1]]).to(device)
        ref_text_mask = length_to_mask(ref_input_lengths).to(device)
        ref_bert_dur = model.bert(ref_tokens, attention_mask=(~ref_text_mask).int())

        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device), 
                                          embedding=ref_bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)

        s_pred_1st = s_pred[:, :128] # acustics from text
        ref_s_1st = ref_s[:, :128] # acoustic from audio
        s_pred_2nd = s_pred[:, 128:] # prosody from text
        ref_s_2nd = ref_s[:, 128:] # prosody from audio

        ref = alpha * s_pred_1st + (1 - alpha)  * ref_s_1st # acoustics
        s = beta * s_pred_2nd + (1 - beta) * ref_s_2nd # prosody

        d = model.predictor.text_encoder(d_en, 
                                         s, input_lengths, text_mask)

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

        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        out2 = out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later
        # out2 = out.squeeze().cpu().detach().numpy()[..., :-50] 
        
    return out2

def run_infer(text, ref_s, diffusion_steps, embedding_scale, alpha=0.3, beta=0.7):

    start = time.time()
    wav = inference(text, ref_s, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
    end = time.time()
    duration_proc = end - start
    duration_out = len(wav) / 24000
    rtf = duration_proc / duration_out

    return wav, rtf, (duration_proc, duration_out)

def run_LFinfer(passage, ref_s, diffusion_steps=5, embedding_scale=1, alpha=0.3, beta=0.7, t=0.7):

    sentences = passage.split('.') # simple split by comma
    nsentences = len(sentences)

    wavs = []
    durations_proc = []
    durations_out = []
    s_prev = None
    for i, text in enumerate(sentences):
        if text.strip() == "": continue
        text += '.' # add it back
        
        start = time.time()
        wav, s_prev = LFinference(text, s_prev, ref_s, alpha=alpha, beta=beta, t=t, 
                                    diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
        end = time.time()
        dur_proc = end - start
        dur_wav = len(wav) / 24000
        durations_proc.append(dur_proc)
        durations_out.append(dur_wav)
        wavs.append(wav)

    wav = np.concatenate(wavs)
    duration_proc = sum(durations_proc)
    duration_out = sum(durations_out)
    rtf = duration_proc / duration_out

    return wav, rtf, (duration_proc, duration_out)

def run_STinfer(text, ref_s, ref_text, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1):

    start = time.time()
    wav = STinference(text, ref_s, ref_text, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)    
    end = time.time()
    duration_proc = end - start
    duration_out = len(wav) / 24000
    rtf = duration_proc / duration_out

    return wav, rtf, (duration_proc, duration_out)

def filter_path(paths, keywords):
    for kw in keywords:
        paths = [f for f in paths if kw not in f]
    return paths

def parse_args():
    usage = 'usage: inference demo for LibriTTS'
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
    # args.config_path = os.path.join(work_path, 'Models', 'LibriTTS', 'config.yml')
    # args.model_path = os.path.join(work_path, 'Models', 'LibriTTS', 'epochs_2nd_00020.pth')
    # args.output_path = os.path.join(work_path, 'Outputs', 'Demo', 'LibriTTS')
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
    meter = pyln.Meter(24000)
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
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
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
 
    #%% Synthesize speech

    exp_id = 0
    output_path = os.path.join(args.output_path, 'exp-{:02d}'.format(exp_id))
    os.makedirs(output_path, exist_ok=True)
    print('output path for exp {}: {}'.format(exp_id, output_path))
    
    text = ''' StyleTTS 2 is a text to speech model that leverages style diffusion and adversarial training with large speech language models to achieve human level text to speech synthesis. '''

    ## Basic synthesis (5 diffusion steps, seen speakers)

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
    diffusion_steps = 5
    embedding_scale = 1
    alpha=0.3
    beta=0.7
    for i, ref_path in enumerate(reference_dicts.values()):

        print('({}/{}) reference: {}'.format(i+1, nsamples, ref_path))

        # compute reference style from ref wav file
        ref_path = os.path.join(os.getcwd(), ref_path)
        ref_s = compute_style(model, ref_path, device=device)

        # get ref wav duration
        ref_wav, _ = librosa.load(ref_path, sr=24000)
        duration_ref = len(ref_wav) / 24000

        # get syn wav and rtf
        wav, rtf, (duration_proc, duration_out) = run_infer(text, ref_s, diffusion_steps, embedding_scale, alpha=alpha, beta=beta)
        print('basic, seen speakers, diffusion steps: {}, embedding scale: {}'.format(diffusion_steps, embedding_scale))
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

        # write output syn wav file
        ref_id = os.path.splitext(os.path.basename(ref_path))[0]
        ref_id = '_'.join(ref_id.split('_')[:-2])
        output_filename = 'basic-{}-{}-{}-{}-{}-{}.wav'.format(i, ref_id, diffusion_steps, embedding_scale, alpha, beta)    
        output_filepath = os.path.join(output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        # compute SSS (method 2, read from audio files)
        ref_wav2 = get_audio(ref_path, meter, sample_rate=24000)
        wav2 = get_audio(output_filepath, meter, sample_rate=24000)
        ref_spkr_embedding = extract_spkr_embedding(ref_wav2, 24000)
        syn_spkr_embedding = extract_spkr_embedding(wav2, 24000)
        sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)
        print('sss: {:.3f}'.format(sss))

        meta = {'syn-wav': output_filepath, 'text': text, 'ref-wav': ref_path, 'dur-ref': duration_ref,
                'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': diffusion_steps,
                'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'seen-speaker': True,
                'sss': float(sss), 'hostname': hostname, 'gpu': gpu_info, 'exp-id': exp_id, 'topic': 'basic'}
        output_jsonfile = os.path.join(output_path, output_filename.replace('.wav', '.json'))
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2)

        rtfs[i] = rtf

    rtf_avg = np.mean(rtfs)
    print('average RTF (basic, seen speakers, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))

    #%% With higher diffusion steps (more diverse)
    # Since the sampler is ancestral, the higher the stpes, the more diverse the samples are, with the cost of slower synthesis speed.

    rtfs = [0 for _ in range(nsamples)]
    diffusion_steps = 10
    embedding_scale = 1
    alpha = 0.3
    beta = 0.7
    for i, ref_path in enumerate(reference_dicts.values()):

        print('({}/{}) reference: {}'.format(i+1, nsamples, ref_path))

        ref_path = os.path.join(os.getcwd(), ref_path)
        ref_s = compute_style(model, ref_path, device=device)

        ref_wav, _ = librosa.load(ref_path, sr=24000)
        duration_ref = len(ref_wav) / 24000

        wav, rtf, (duration_proc, duration_out) = run_infer(text, ref_s, diffusion_steps, embedding_scale, alpha=alpha, beta=beta)
        print('basic, seen speakers, diffusion steps: {}, embedding scale: {}'.format(diffusion_steps, embedding_scale))
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

        ref_id = os.path.splitext(os.path.basename(ref_path))[0]
        ref_id = '_'.join(ref_id.split('_')[:-2])
        output_filename = 'basic-{}-{}-{}-{}-{}-{}.wav'.format(i, ref_id, diffusion_steps, embedding_scale, alpha, beta)    
        output_filepath = os.path.join(output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        # compute SSS (method 2, read from audio files)
        ref_wav2 = get_audio(ref_path, meter)
        wav2 = get_audio(output_filepath, meter)
        ref_spkr_embedding = extract_spkr_embedding(ref_wav2, 24000)
        syn_spkr_embedding = extract_spkr_embedding(wav2, 24000)
        sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)
        print('sss: {:.3f}'.format(sss))

        meta = {'syn-wav': output_filepath, 'text': text, 'ref-wav': ref_path, 'dur-ref': duration_ref,
                'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': diffusion_steps,
                'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'seen-speaker': True,
                'sss': float(sss), 'hostname': hostname, 'gpu': gpu_info, 'exp-id': exp_id, 'topic': 'more diverse'}        
        output_jsonfile = os.path.join(output_path, output_filename.replace('.wav', '.json'))
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2)

        rtfs[i] = rtf

    rtf_avg = np.mean(rtfs)
    print('average RTF (basic, seen speakers, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))

    #%% Basic synthesis (5 diffusion steps, unseen speakers)

    exp_id = 1
    output_path = os.path.join(args.output_path, 'exp-{:02d}'.format(exp_id))
    os.makedirs(output_path, exist_ok=True)
    print('output path for exp {}: {}'.format(exp_id, output_path))

    reference_dicts = {}
    # format: (path, text)
    reference_dicts['1221-135767'] = ("Demo/reference_audio/1221-135767-0014.wav", "Yea, his honourable worship is within, but he hath a godly minister or two with him, and likewise a leech.")
    reference_dicts['5639-40744'] = ("Demo/reference_audio/5639-40744-0020.wav", "Thus did this humane and right minded father comfort his unhappy daughter, and her mother embracing her again, did all she could to soothe her feelings.")
    reference_dicts['908-157963'] = ("Demo/reference_audio/908-157963-0027.wav", "And lay me down in my cold bed and leave my shining lot.")
    reference_dicts['4077-13754'] = ("Demo/reference_audio/4077-13754-0000.wav", "The army found the people in poverty and left them in comparative wealth.")
    nsamples = len(reference_dicts)

    for i, v in enumerate(reference_dicts.values()):
        ref_path, text = v
        ref_id = os.path.splitext(os.path.basename(ref_path))[0]
        ref_id = '_'.join(ref_id.split('-')[:-1])
        reference_filename = 'reference-{}-{}.wav'.format(i, ref_id)
        reference_filepath = os.path.join(output_path, reference_filename)
        copyfile(ref_path, reference_filepath)
        print('copied reference file: {} -> {}'.format(ref_path, reference_filepath))

    rtfs = [0 for _ in range(nsamples)]
    diffusion_steps = 5
    embedding_scale = 1
    alpha = 0.3
    beta = 0.7
    for i, v in enumerate(reference_dicts.values()):
        ref_path, text = v
        print('({}/{}) reference: {}, text: {}'.format(i+1, nsamples, ref_path, text))

        ref_path = os.path.join(os.getcwd(), ref_path)
        ref_s = compute_style(model, ref_path, device=device)

        ref_wav, _ = librosa.load(ref_path, sr=24000)
        duration_ref = len(ref_wav) / 24000

        wav, rtf, (duration_proc, duration_out) = run_infer(text, ref_s, diffusion_steps, embedding_scale, alpha=alpha, beta=beta)
        print('basic, unseen speakers, diffusion steps: {}, embedding scale: {}'.format(diffusion_steps, embedding_scale))
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

        ref_id = os.path.splitext(os.path.basename(ref_path))[0]
        ref_id = '_'.join(ref_id.split('-')[:-1])
        output_filename = 'basic-{}-{}-{}-{}-{}-{}.wav'.format(i, ref_id, diffusion_steps, embedding_scale, alpha, beta)    
        output_filepath = os.path.join(output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        ref_wav2 = get_audio(ref_path, meter)
        wav2 = get_audio(output_filepath, meter)
        ref_spkr_embedding = extract_spkr_embedding(ref_wav2, 24000)
        syn_spkr_embedding = extract_spkr_embedding(wav2, 24000)
        sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)
        print('sss: {:.3f}'.format(sss))

        meta = {'syn-wav': output_filepath, 'text': text, 'ref-wav': ref_path, 'dur-ref': duration_ref,
                'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': diffusion_steps,
                'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'seen-speaker': False,
                'sss': float(sss), 'hostname': hostname, 'gpu': gpu_info, 'exp-id': exp_id, 'topic': 'unseen speakers'}  
        output_jsonfile = os.path.join(output_path, output_filename.replace('.wav', '.json'))
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2)

        rtfs[i] = rtf

    rtf_avg = np.mean(rtfs)
    print('average RTF (basic, unseen speakers, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))

    #%% Speech expressiveness

    exp_id = 2
    output_path = os.path.join(args.output_path, 'exp-{:02d}'.format(exp_id))
    os.makedirs(output_path, exist_ok=True)
    print('output path for exp {}: {}'.format(exp_id, output_path))

    ## With embedding scale 1

    ref_path = "Demo/reference_audio/1221-135767-0014.wav"
    ref_path = os.path.join(os.getcwd(), ref_path)
    ref_s = compute_style(model, ref_path, device=device)

    ref_wav, _ = librosa.load(ref_path, sr=24000)
    duration_ref = len(ref_wav) / 24000
    
    texts = {}
    texts['Happy'] = "We are happy to invite you to join us on a journey to the past, where we will visit the most amazing monuments ever built by human hands."
    texts['Sad'] = "I am sorry to say that we have suffered a severe setback in our efforts to restore prosperity and confidence."
    texts['Angry'] = "The field of astronomy is a joke! Its theories are based on flawed observations and biased interpretations!"
    texts['Surprised'] = "I can't believe it! You mean to tell me that you have discovered a new species of bacteria in this pond?"
    nsamples = len(texts)

    ref_id = os.path.splitext(os.path.basename(ref_path))[0]
    ref_id = '_'.join(ref_id.split('-')[:-1])

    rtfs = [0 for _ in range(nsamples)]
    diffusion_steps = 10
    embedding_scale = 1
    alpha = 0.3
    beta = 0.7
    for i, (emo, text) in enumerate(texts.items()):

        wav, rtf, (duration_proc, duration_out) = run_infer(text, ref_s, diffusion_steps, embedding_scale, alpha=alpha, beta=beta)
        print('expressive, unseen speakers, diffusion steps: {}, embedding scale: {}'.format(diffusion_steps, embedding_scale))
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

        output_filename = '{}-{}-{}-{}-{}-{}.wav'.format(emo.lower(), ref_id, diffusion_steps, embedding_scale, alpha, beta)    
        output_filepath = os.path.join(output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        ref_wav2 = get_audio(ref_path, meter)
        wav2 = get_audio(output_filepath, meter)
        ref_spkr_embedding = extract_spkr_embedding(ref_wav2, 24000)
        syn_spkr_embedding = extract_spkr_embedding(wav2, 24000)
        sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)
        print('sss: {:.3f}'.format(sss))

        meta = {'syn-wav': output_filepath, 'text': text, 'ref-wav': ref_path, 'dur-ref': duration_ref,
                'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': diffusion_steps,
                'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'seen-speaker': False,
                'emo': emo, 'sss': float(sss), 'hostname': hostname, 'gpu': gpu_info, 'exp-id': exp_id, 'topic': 'expressiveness'}
        output_jsonfile = os.path.join(output_path, output_filename.replace('.wav', '.json'))
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2)

        rtfs[i] = rtf

    rtf_avg = np.mean(rtfs)
    print('average RTF (expressive, unseen speakers, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))

    #%% With embedding scale 2

    rtfs = [0 for _ in range(nsamples)]
    diffusion_steps = 10
    embedding_scale = 2
    alpha = 0.3
    beta = 0.7
    for i, (emo, text) in enumerate(texts.items()):

        wav, rtf, (duration_proc, duration_out) = run_infer(text, ref_s, diffusion_steps, embedding_scale, alpha=alpha, beta=beta)
        print('basic, unseen speakers, diffusion steps: {}, embedding scale: {}'.format(diffusion_steps, embedding_scale))
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

        output_filename = '{}-{}-{}-{}-{}-{}.wav'.format(emo.lower(), ref_id, diffusion_steps, embedding_scale, alpha, beta)    
        output_filepath = os.path.join(output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        ref_wav2 = get_audio(ref_path, meter)
        wav2 = get_audio(output_filepath, meter)
        ref_spkr_embedding = extract_spkr_embedding(ref_wav2, 24000)
        syn_spkr_embedding = extract_spkr_embedding(wav2, 24000)
        sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)
        print('sss: {:.3f}'.format(sss))

        meta = {'syn-wav': output_filepath, 'text': text, 'ref-wav': ref_path, 'dur-ref': duration_ref,
                'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': diffusion_steps,
                'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'seen-speaker': False,
                'emo': emo, 'sss': float(sss), 'hostname': hostname, 'gpu': gpu_info, 'exp-id': exp_id, 'topic': 'expressiveness'}
        output_jsonfile = os.path.join(output_path, output_filename.replace('.wav', '.json'))
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2)

        rtfs[i] = rtf

    rtf_avg = np.mean(rtfs)
    print('average RTF (expressive, unseen speakers, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))

    #%% With embedding scale 2, alpha 0.5 and beta 0.9
    # alpha and beta is the factor to determine how much we use the style sampled based on the text instead of the reference.
    # The higher the value of alpha and beta, the more suitable the style it is to the text, but less similar to the reference.
    # Using higher beta makes the synthesized speech more emotional, at the cost of lower similarity to the reference.
    # alpha determines the timbre of the speaker while beta determines the prosody.

    rtfs = [0 for _ in range(nsamples)]
    diffusion_steps = 10
    embedding_scale = 2
    alpha = 0.5
    beta = 0.9
    for i, (emo, text) in enumerate(texts.items()):

        wav, rtf, (duration_proc, duration_out) = run_infer(text, ref_s, diffusion_steps, embedding_scale, alpha=alpha, beta=beta)
        print('basic, unseen speakers, diffusion steps: {}, embedding scale: {}'.format(diffusion_steps, embedding_scale))
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

        output_filename = '{}-{}-{}-{}-{}-{}.wav'.format(emo.lower(), ref_id, diffusion_steps, embedding_scale, alpha, beta)    
        output_filepath = os.path.join(output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        ref_wav2 = get_audio(ref_path, meter)
        wav2 = get_audio(output_filepath, meter)
        ref_spkr_embedding = extract_spkr_embedding(ref_wav2, 24000)
        syn_spkr_embedding = extract_spkr_embedding(wav2, 24000)
        sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)
        print('sss: {:.3f}'.format(sss))

        meta = {'syn-wav': output_filepath, 'text': text, 'ref-wav': ref_path, 'dur-ref': duration_ref,
                'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': diffusion_steps,
                'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'seen-speaker': False,
                'emo': emo, 'sss': float(sss), 'hostname': hostname, 'gpu': gpu_info, 'exp-id': exp_id, 'topic': 'expressiveness'}
        output_jsonfile = os.path.join(output_path, output_filename.replace('.wav', '.json'))
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2)

        rtfs[i] = rtf

    rtf_avg = np.mean(rtfs)
    print('average RTF (expressive, unseen speakers, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))

    #%% Zero-shot speaker adapatation 1 (Acoustic Environment Maintenance)

    # maintain the acoustic environment in the speaker (timbre) -> alpha = 0 (speaker as close to the refrence as possible)
    # change the prosody (beta) acording to the text

    exp_id = 3
    output_path = os.path.join(args.output_path, 'exp-{:02d}'.format(exp_id))
    os.makedirs(output_path, exist_ok=True)
    print('output path for exp {}: {}'.format(exp_id, output_path))

    reference_dicts = {}
    # format: (path, text)
    reference_dicts['3'] = ("Demo/reference_audio/3.wav", "As friends thing I definitely I've got more male friends.")
    reference_dicts['4'] = ("Demo/reference_audio/4.wav", "Everything is run by computer but you got to know how to think before you can do a computer.")
    reference_dicts['5'] = ("Demo/reference_audio/5.wav", "Then out in LA you guys got a whole another ball game within California to worry about.")
    nsamples = len(reference_dicts)

    for i, v in enumerate(reference_dicts.values()):
        ref_path, text = v
        ref_id = os.path.splitext(os.path.basename(ref_path))[0]
        reference_filename = 'reference-{}-{}.wav'.format(i, ref_id)
        reference_filepath = os.path.join(output_path, reference_filename)
        copyfile(ref_path, reference_filepath)
        print('copied reference file: {} -> {}'.format(ref_path, reference_filepath))    

    rtfs = [0 for _ in range(nsamples)]
    diffusion_steps = 5
    embedding_scale = 1
    alpha = 0.0
    beta = 1.0
    for i, v in enumerate(reference_dicts.values()):
        ref_path, text = v

        ref_path = os.path.join(os.getcwd(), ref_path)
        ref_s = compute_style(model, ref_path, device=device)

        ref_wav, _ = librosa.load(ref_path, sr=24000)
        duration_ref = len(ref_wav) / 24000

        wav, rtf, (duration_proc, duration_out) = run_infer(text, ref_s, diffusion_steps, embedding_scale, alpha=alpha, beta=beta)
        print('acoustic environment maintenance, unseen speakers, diffusion steps: {}, embedding scale: {}, alpha: {}, beta: {}'.format(
            diffusion_steps, embedding_scale, alpha, beta))
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

        ref_id = os.path.splitext(os.path.basename(ref_path))[0]
        output_filename = 'aem-{}-{}-{}-{}-{}-{}.wav'.format(i, ref_id, diffusion_steps, embedding_scale, alpha, beta)    
        output_filepath = os.path.join(output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        ref_wav2 = get_audio(ref_path, meter)
        wav2 = get_audio(output_filepath, meter)
        ref_spkr_embedding = extract_spkr_embedding(ref_wav2, 24000)
        syn_spkr_embedding = extract_spkr_embedding(wav2, 24000)
        sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)
        print('sss: {:.3f}'.format(sss))

        meta = {'syn-wav': output_filepath, 'text': text, 'ref-wav': ref_path, 'dur-ref': duration_ref,
                'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': diffusion_steps,
                'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'seen-speaker': False,
                'sss': float(sss), 'hostname': hostname, 'gpu': gpu_info, 'exp-id': exp_id, 'topic': 'acoustic environment maintenance'}
        output_jsonfile = os.path.join(output_path, output_filename.replace('.wav', '.json'))
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2)

        rtfs[i] = rtf

    rtf_avg = np.mean(rtfs)
    print('average RTF (adaption, unseen speakers, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))

    #%% Zero-shot speaker adapatation 2 (Speaker's Emotion Maintenance)

    # maintain speaker's emotion (prosody) -> beta = 0.1 to make the speaker as closer to the reference as possible
    # while having some diversity through the slight timbre change (small alpha, e.g. alpha=0.3)

    exp_id = 4
    output_path = os.path.join(args.output_path, 'exp-{:02d}'.format(exp_id))
    os.makedirs(output_path, exist_ok=True)
    print('output path for exp {}: {}'.format(exp_id, output_path))

    reference_dicts = {}
    # format: (path, text)
    reference_dicts['Anger'] = ("Demo/reference_audio/anger.wav", "We have to reduce the number of plastic bags.")
    reference_dicts['Sleepy'] = ("Demo/reference_audio/sleepy.wav", "We have to reduce the number of plastic bags.")
    reference_dicts['Amused'] = ("Demo/reference_audio/amused.wav", "We have to reduce the number of plastic bags.")
    reference_dicts['Disgusted'] = ("Demo/reference_audio/disgusted.wav", "We have to reduce the number of plastic bags.")
    nsamples = len(reference_dicts)

    for i, v in enumerate(reference_dicts.values()):
        ref_path, text = v
        ref_id = os.path.splitext(os.path.basename(ref_path))[0]
        reference_filename = 'reference-{}-{}.wav'.format(i, ref_id)
        reference_filepath = os.path.join(output_path, reference_filename)
        copyfile(ref_path, reference_filepath)
        print('copied reference file: {} -> {}'.format(ref_path, reference_filepath))

    rtfs = [0 for _ in range(nsamples)]
    diffusion_steps = 10
    embedding_scale = 1
    alpha = 0.0
    beta = 1.0
    for i, (emo, v) in enumerate(reference_dicts.items()):
        ref_path, text = v

        ref_path = os.path.join(os.getcwd(), ref_path)
        ref_s = compute_style(model, ref_path, device=device)

        ref_wav, _ = librosa.load(ref_path, sr=24000)
        duration_ref = len(ref_wav) / 24000

        wav, rtf, (duration_proc, duration_out) = run_infer(text, ref_s, diffusion_steps, embedding_scale, alpha=alpha, beta=beta)
        print('speaker emotion maintenance, unseen speakers, diffusion steps: {}, embedding scale: {}, alpha: {}, beta: {}'.format(
            diffusion_steps, embedding_scale, alpha, beta))
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

        ref_id = os.path.splitext(os.path.basename(ref_path))[0]
        output_filename = '{}-{}-{}-{}-{}-{}-{}.wav'.format(emo.lower(), i, ref_id, diffusion_steps, embedding_scale, alpha, beta)    
        output_filepath = os.path.join(output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        ref_wav2 = get_audio(ref_path, meter)
        wav2 = get_audio(output_filepath, meter)
        ref_spkr_embedding = extract_spkr_embedding(ref_wav2, 24000)
        syn_spkr_embedding = extract_spkr_embedding(wav2, 24000)
        sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)
        print('sss: {:.3f}'.format(sss))

        meta = {'syn-wav': output_filepath, 'text': text, 'ref-wav': ref_path, 'dur-ref': duration_ref,
                'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': diffusion_steps,
                'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'seen-speaker': False,
                'emo': emo, 'sss': float(sss), 'hostname': hostname, 'gpu': gpu_info, 'exp-id': exp_id, 'topic': 'speaker emotion maintenance'}
        output_jsonfile = os.path.join(output_path, output_filename.replace('.wav', '.json'))
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2)

        rtfs[i] = rtf

    rtf_avg = np.mean(rtfs)
    print('average RTF (adaptation, unseen speakers, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))

    #%% Longform Narration

    exp_id = 5
    output_path = os.path.join(args.output_path, 'exp-{:02d}'.format(exp_id))
    os.makedirs(output_path, exist_ok=True)
    print('output path for exp {}: {}'.format(exp_id, output_path))

    passage = '''If the supply of fruit is greater than the family needs, it may be made a source of income by sending the fresh fruit to the market if there is one near enough, or by preserving, canning, and making jelly for sale. To make such an enterprise a success the fruit and work must be first class. There is magic in the word "Homemade," when the product appeals to the eye and the palate; but many careless and incompetent people have found to their sorrow that this word has not magic enough to float inferior goods on the market. As a rule large canning and preserving establishments are clean and have the best appliances, and they employ chemists and skilled labor. The home product must be very good to compete with the attractive goods that are sent out from such establishments. Yet for first class home made products there is a market in all large cities. All first-class grocers have customers who purchase such goods.'''

    # unseen speaker
    ref_path = "Demo/reference_audio/1221-135767-0014.wav"

    ref_id = os.path.splitext(os.path.basename(ref_path))[0]
    ref_id = '_'.join(ref_id.split('-')[:-1])
    reference_filename = 'reference-{}.wav'.format(ref_id)
    reference_filepath = os.path.join(output_path, reference_filename)
    copyfile(ref_path, reference_filepath)
    print('copied reference file: {} -> {}'.format(ref_path, reference_filepath))

    ref_path = os.path.join(os.getcwd(), ref_path)
    ref_s = compute_style(model, ref_path, device=device)

    ref_wav, _ = librosa.load(ref_path, sr=24000)
    duration_ref = len(ref_wav) / 24000

    diffusion_steps = 10
    embedding_scale = 1.5
    alpha = 0.3
    beta = 0.9  # more suitable for the text
    t = 0.7
    wav, rtf, (duration_proc, duration_out) = run_LFinfer(passage, ref_s, diffusion_steps=diffusion_steps,
        embedding_scale=embedding_scale, alpha=0.3, beta=0.9, t=0.7)

    output_filename = 'LF-{}-{}-{}-{}-{}.wav'.format(ref_id, diffusion_steps, embedding_scale, alpha, beta)    
    output_filepath = os.path.join(output_path, output_filename)
    sf.write(output_filepath, wav, 24000)
    print('wrote output file: {}'.format(output_filepath))

    ref_wav2 = get_audio(ref_path, meter)
    wav2 = get_audio(output_filepath, meter)
    ref_spkr_embedding = extract_spkr_embedding(ref_wav2, 24000)
    syn_spkr_embedding = extract_spkr_embedding(wav2, 24000)
    sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)
    print('sss: {:.3f}'.format(sss))

    meta = {'syn-wav': output_filepath, 'text': text, 'ref-wav': ref_path, 'dur-ref': duration_ref,
        'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': diffusion_steps,
        'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'seen-speaker': False,
        'sss': float(sss), 'hostname': hostname, 'gpu': gpu_info, 'exp-id': exp_id, 'topic': 'longform narration'}
    output_jsonfile = os.path.join(output_path, output_filename.replace('.wav', '.json'))
    with open(output_jsonfile, 'w') as fp:
        json.dump(meta, fp, indent=2)
   
    #%% Style Transfer

    exp_id = 6
    output_path = os.path.join(args.output_path, 'exp-{:02d}'.format(exp_id))
    os.makedirs(output_path, exist_ok=True)
    print('output path for exp {}: {}'.format(exp_id, output_path))

    ref_texts = {}
    ref_texts['Happy'] = "We are happy to invite you to join us on a journey to the past, where we will visit the most amazing monuments ever built by human hands."
    ref_texts['Sad'] = "I am sorry to say that we have suffered a severe setback in our efforts to restore prosperity and confidence."
    ref_texts['Angry'] = "The field of astronomy is a joke! Its theories are based on flawed observations and biased interpretations!"
    ref_texts['Surprised'] = "I can't believe it! You mean to tell me that you have discovered a new species of bacteria in this pond?"

    ref_path = "Demo/reference_audio/1221-135767-0014.wav"
    ref_id = os.path.splitext(os.path.basename(ref_path))[0]
    ref_id = '_'.join(ref_id.split('-')[:-1])
    reference_filename = 'reference-{}.wav'.format(ref_id)
    reference_filepath = os.path.join(output_path, reference_filename)
    copyfile(ref_path, reference_filepath)
    print('copied reference file: {} -> {}'.format(ref_path, reference_filepath))

    ref_path = os.path.join(os.getcwd(), ref_path)
    ref_s = compute_style(model, ref_path, device=device)

    ref_wav, _ = librosa.load(ref_path, sr=24000)
    duration_ref = len(ref_wav) / 24000

    text = "Yea, his honourable worship is within, but he hath a godly minister or two with him, and likewise a leech."
    
    diffusion_steps=10
    embedding_scale=1.5
    alpha = 0.5
    beta = 0.9
    for i, (emo, ref_text) in enumerate(ref_texts.items()):

        wav, rtf, (duration_proc, duration_out) = run_STinfer(text, ref_s, ref_text,
            alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

        output_filename = '{}-{}-{}-{}-{}-{}.wav'.format(emo.lower(), ref_id, diffusion_steps, embedding_scale, alpha, beta)    
        output_filepath = os.path.join(output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        ref_wav2 = get_audio(ref_path, meter)
        wav2 = get_audio(output_filepath, meter)
        ref_spkr_embedding = extract_spkr_embedding(ref_wav2, 24000)
        syn_spkr_embedding = extract_spkr_embedding(wav2, 24000)
        sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)
        print('sss: {:.3f}'.format(sss))

        meta = {'syn-wav': output_filepath, 'text': text, 'ref-wav': ref_path, 'ref-text': ref_text, 'dur-ref': duration_ref,
                'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': diffusion_steps,
                'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'seen-speaker': False,
                'emo': emo, 'sss': float(sss), 'hostname': hostname, 'gpu': gpu_info, 'exp-id': exp_id, 'topic': 'style transfer'}
        output_jsonfile = os.path.join(output_path, output_filename.replace('.wav', '.json'))
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2)

        rtfs[i] = rtf

    rtf_avg = np.mean(rtfs)
    print('average RTF (style transfer, unseen speakers, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))    

    #%% Speech diversity

    # if alpha = 1 and beta = 1, the synthesized speech sounds the most dissimilar to the reference speaker, 
    # but it is also the most diverse (each time you synthesize a speech it will be totally different). 
    # if alpha = 0 and beta = 0, synthesized speech sounds the most siimlar to the reference speaker, 
    # but it is deterministic (i.e., the sampled style is not used for speech synthesis)

    exp_id = 7
    output_path = os.path.join(args.output_path, 'exp-{:02d}'.format(exp_id))
    os.makedirs(output_path, exist_ok=True)
    print('output path for exp {}: {}'.format(exp_id, output_path))

    ref_path = "Demo/reference_audio/1221-135767-0014.wav"
    ref_id = os.path.splitext(os.path.basename(ref_path))[0]
    ref_id = '_'.join(ref_id.split('-')[:-1])
    reference_filename = 'reference-{}.wav'.format(ref_id)
    reference_filepath = os.path.join(output_path, reference_filename)
    copyfile(ref_path, reference_filepath)
    print('copied reference file: {} -> {}'.format(ref_path, reference_filepath))

    ref_path = os.path.join(os.getcwd(), ref_path)
    ref_s = compute_style(model, ref_path, device=device)

    ref_wav, _ = librosa.load(ref_path, sr=24000)
    duration_ref = len(ref_wav) / 24000

    text = "How much variation is there?"
    nsamples = 5

    # default setting (alpha = 0.3, beta = 0.7)
    rtfs = [0 for _ in range(nsamples)]
    diffusion_steps = 10
    embedding_scale = 1
    alpha = 0.3
    beta = 0.7
    for i in range(nsamples):
        wav, rtf, (duration_proc, duration_out) = run_infer(text, ref_s, diffusion_steps, embedding_scale, alpha=alpha, beta=beta)
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

        output_filename = 'div3-{}-{}-{}-{}-{}-{}.wav'.format(i, ref_id, diffusion_steps, embedding_scale, alpha, beta)    
        output_filepath = os.path.join(output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        ref_wav2 = get_audio(ref_path, meter)
        wav2 = get_audio(output_filepath, meter)
        ref_spkr_embedding = extract_spkr_embedding(ref_wav2, 24000)
        syn_spkr_embedding = extract_spkr_embedding(wav2, 24000)
        sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)
        print('sss: {:.3f}'.format(sss))

        meta = {'syn-wav': output_filepath, 'text': text, 'ref-wav': ref_path, 'dur-ref': duration_ref,
                'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': diffusion_steps,
                'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'seen-speaker': False,
                'sss': float(sss), 'hostname': hostname, 'gpu': gpu_info, 'exp-id': exp_id, 'topic': 'speech diversity'}
        output_jsonfile = os.path.join(output_path, output_filename.replace('.wav', '.json'))
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2)

        rtfs[i] = rtf

    rtf_avg = np.mean(rtfs)
    print('average RTF (style transfer, unseen speakers, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))

    # less diverse (alpha = 0.1, beta = 0.3)
    rtfs = [0 for _ in range(nsamples)]
    diffusion_steps = 10
    embedding_scale = 1
    alpha = 0.1
    beta = 0.3
    for i in range(nsamples):
        wav, rtf, (duration_proc, duration_out) = run_infer(text, ref_s, diffusion_steps, embedding_scale, alpha=alpha, beta=beta)
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

        output_filename = 'div2-{}-{}-{}-{}-{}-{}.wav'.format(i, ref_id, diffusion_steps, embedding_scale, alpha, beta)    
        output_filepath = os.path.join(output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        ref_wav2 = get_audio(ref_path, meter)
        wav2 = get_audio(output_filepath, meter)
        ref_spkr_embedding = extract_spkr_embedding(ref_wav2, 24000)
        syn_spkr_embedding = extract_spkr_embedding(wav2, 24000)
        sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)
        print('sss: {:.3f}'.format(sss))

        meta = {'syn-wav': output_filepath, 'text': text, 'ref-wav': ref_path, 'dur-ref': duration_ref,
                'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': diffusion_steps,
                'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'seen-speaker': False,
                'sss': float(sss), 'hostname': hostname, 'gpu': gpu_info, 'exp-id': exp_id, 'topic': 'speech diversity'}
        output_jsonfile = os.path.join(output_path, output_filename.replace('.wav', '.json'))
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2)

        rtfs[i] = rtf
        
    rtf_avg = np.mean(rtfs)
    print('average RTF (style transfer, unseen speakers, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))

    # more diverse (alpha = 0.5, beta = 0.95)
    rtfs = [0 for _ in range(nsamples)]
    diffusion_steps = 10
    embedding_scale = 1
    alpha = 0.5
    beta = 0.95
    for i in range(nsamples):
        wav, rtf, (duration_proc, duration_out) = run_infer(text, ref_s, diffusion_steps, embedding_scale, alpha=alpha, beta=beta)
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

        output_filename = 'div4-{}-{}-{}-{}-{}-{}.wav'.format(i, ref_id, diffusion_steps, embedding_scale, alpha, beta)    
        output_filepath = os.path.join(output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        ref_wav2 = get_audio(ref_path, meter)
        wav2 = get_audio(output_filepath, meter)
        ref_spkr_embedding = extract_spkr_embedding(ref_wav2, 24000)
        syn_spkr_embedding = extract_spkr_embedding(wav2, 24000)
        sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)
        print('sss: {:.3f}'.format(sss))

        meta = {'syn-wav': output_filepath, 'text': text, 'ref-wav': ref_path, 'dur-ref': duration_ref,
                'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': diffusion_steps,
                'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'seen-speaker': False,
                'sss': float(sss), 'hostname': hostname, 'gpu': gpu_info, 'exp-id': exp_id, 'topic': 'speech diversity'}
        output_jsonfile = os.path.join(output_path, output_filename.replace('.wav', '.json'))
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2)

        rtfs[i] = rtf
        
    rtf_avg = np.mean(rtfs)
    print('average RTF (style transfer, unseen speakers, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))

    # most/extreme diverse (alpha = 1, beta = 1)
    rtfs = [0 for _ in range(nsamples)]
    diffusion_steps = 10
    embedding_scale = 1
    alpha = 1
    beta = 1
    for i in range(nsamples):
        wav, rtf, (duration_proc, duration_out) = run_infer(text, ref_s, diffusion_steps, embedding_scale, alpha=alpha, beta=beta)
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

        output_filename = 'div5-{}-{}-{}-{}-{}-{}.wav'.format(i, ref_id, diffusion_steps, embedding_scale, alpha, beta)    
        output_filepath = os.path.join(output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        ref_wav2 = get_audio(ref_path, meter)
        wav2 = get_audio(output_filepath, meter)
        ref_spkr_embedding = extract_spkr_embedding(ref_wav2, 24000)
        syn_spkr_embedding = extract_spkr_embedding(wav2, 24000)
        sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)
        print('sss: {:.3f}'.format(sss))

        meta = {'syn-wav': output_filepath, 'text': text, 'ref-wav': ref_path, 'dur-ref': duration_ref,
                'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': diffusion_steps,
                'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'seen-speaker': False,
                'sss': float(sss), 'hostname': hostname, 'gpu': gpu_info, 'exp-id': exp_id, 'topic': 'speech diversity'}
        output_jsonfile = os.path.join(output_path, output_filename.replace('.wav', '.json'))
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2)

        rtfs[i] = rtf
        
    rtf_avg = np.mean(rtfs)
    print('average RTF (style transfer, unseen speakers, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))

    # least diverse / no vairation (alpha = 0, beta = 0)
    rtfs = [0 for _ in range(nsamples)]
    diffusion_steps = 10
    embedding_scale = 1
    alpha = 0
    beta = 0
    for i in range(nsamples):
        wav, rtf, (duration_proc, duration_out) = run_infer(text, ref_s, diffusion_steps, embedding_scale, alpha=alpha, beta=beta)
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

        output_filename = 'div1-{}-{}-{}-{}-{}-{}.wav'.format(i, ref_id, diffusion_steps, embedding_scale, alpha, beta)    
        output_filepath = os.path.join(output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        ref_wav2 = get_audio(ref_path, meter)
        wav2 = get_audio(output_filepath, meter)
        ref_spkr_embedding = extract_spkr_embedding(ref_wav2, 24000)
        syn_spkr_embedding = extract_spkr_embedding(wav2, 24000)
        sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)
        print('sss: {:.3f}'.format(sss))

        meta = {'syn-wav': output_filepath, 'text': text, 'ref-wav': ref_path, 'dur-ref': duration_ref,
                'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': diffusion_steps,
                'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'seen-speaker': False,
                'sss': float(sss), 'hostname': hostname, 'gpu': gpu_info, 'exp-id': exp_id, 'topic': 'speech diversity'}
        output_jsonfile = os.path.join(output_path, output_filename.replace('.wav', '.json'))
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2)

        rtfs[i] = rtf
        
    rtf_avg = np.mean(rtfs)
    print('average RTF (style transfer, unseen speakers, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))

    #%% Extra fun

    # Authors' voice

    exp_id = 8
    output_path = os.path.join(args.output_path, 'exp-{:02d}'.format(exp_id))
    os.makedirs(output_path, exist_ok=True)
    print('output path for exp {}: {}'.format(exp_id, output_path))

    text = ''' StyleTTS 2 is a text to speech model that leverages style diffusion and adversarial training with large speech language models to achieve human level text to speech synthesis. '''

    reference_dicts = {}
    reference_dicts['Yinghao'] = "Demo/reference_audio/Yinghao.wav"
    reference_dicts['Gavin'] = "Demo/reference_audio/Gavin.wav"
    reference_dicts['Vinay'] = "Demo/reference_audio/Vinay.wav"
    reference_dicts['Nima'] = "Demo/reference_audio/Nima.wav"
    nsaples = len(reference_dicts)

    for i, ref_path in enumerate(reference_dicts.values()):
        ref_id = os.path.splitext(os.path.basename(ref_path))[0]
        reference_filename = 'reference-{}-{}.wav'.format(i, ref_id)
        reference_filepath = os.path.join(output_path, reference_filename)
        copyfile(ref_path, reference_filepath)
        print('copied reference file: {} -> {}'.format(ref_path, reference_filepath))

    rtfs = [0 for _ in range(nsamples)]
    diffusion_steps = 5
    embedding_scale = 1
    alpha=0.1
    beta=0.5
    for i, ref_path in enumerate(reference_dicts.values()):

        print('({}/{}) reference: {}'.format(i+1, nsamples, ref_path))

        # compute reference style from ref wav file
        ref_path = os.path.join(os.getcwd(), ref_path)
        ref_s = compute_style(model, ref_path, device=device)

        # get ref wav duration
        ref_wav, _ = librosa.load(ref_path, sr=24000)
        duration_ref = len(ref_wav) / 24000

        # get syn wav and rtf
        wav, rtf, (duration_proc, duration_out) = run_infer(text, ref_s, diffusion_steps, embedding_scale, alpha=alpha, beta=beta)
        print('author, unseen speakers, diffusion steps: {}, embedding scale: {}'.format(diffusion_steps, embedding_scale))
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

        # write output syn wav file
        ref_id = os.path.splitext(os.path.basename(ref_path))[0]
        output_filename = 'author-{}-{}-{}-{}-{}-{}.wav'.format(i, ref_id, diffusion_steps, embedding_scale, alpha, beta)    
        output_filepath = os.path.join(output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        # compute SSS (method 2, read from audio files)
        ref_wav2 = get_audio(ref_path, meter, sample_rate=24000)
        wav2 = get_audio(output_filepath, meter, sample_rate=24000)
        ref_spkr_embedding = extract_spkr_embedding(ref_wav2, 24000)
        syn_spkr_embedding = extract_spkr_embedding(wav2, 24000)
        sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)
        print('sss: {:.3f}'.format(sss))

        meta = {'syn-wav': output_filepath, 'text': text, 'ref-wav': ref_path, 'dur-ref': duration_ref,
                'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': diffusion_steps,
                'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'seen-speaker': True,
                'sss': float(sss), 'hostname': hostname, 'gpu': gpu_info, 'exp-id': exp_id, 'topic': 'author'}
        output_jsonfile = os.path.join(output_path, output_filename.replace('.wav', '.json'))
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2)

        rtfs[i] = rtf

    rtf_avg = np.mean(rtfs)
    print('average RTF (basic, seen speakers, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))

    #%% Cross-language zero-shot speaker adapatation 1 (Acoustic Environment Maintenance)

    # maintain the acoustic environment in the speaker (timbre) -> alpha = 0 (speaker as close to the refrence as possible)
    # change the prosody (beta) acording to the text

    exp_id = 11
    output_path = os.path.join(args.output_path, 'exp-{:02d}'.format(exp_id))
    os.makedirs(output_path, exist_ok=True)
    print('output path for exp {}: {}'.format(exp_id, output_path))

    recording_id = 'MARCHE_AssessmentTacticalEnvironment'
    voice = 'lada'
    stress = 'dictionary'
    wav_folder = '{}-{}'.format(voice, stress)
    ref_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts', 'outputs', 'sofw', 'espnet',
        recording_id, wav_folder)
    ref_wavpaths = sorted(glob.glob(os.path.join(ref_path, '*.wav')))
    keywords = ['.16000', '_paired', '_unpaired']
    ref_wavpaths = filter_path(ref_wavpaths, keywords)
    ref_wavpaths = ref_wavpaths[:5]

    texts = ["Hey guys, my name is Sam, and welcome to Prep Medic.",
             "This week's video, we are talking about the March algorithm.",
             "All right, guys, so the March algorithm is the assessment mode that we use in a tactical environment.",
             "A lot of other environments are starting to adapt it like civilian EMS, and essentially, it's a way of addressing life threats on our patients.",
             "So when we come up to a patient in a non-permissive environment such as a tactical situation, our number one priority is always going to be neutralizing the threat."]

    reference_dicts = {str(i+1): (ref_wavpaths[i], texts[i]) for i in range(5)}
    nsamples = len(reference_dicts)

    for i, v in enumerate(reference_dicts.values()):
        ref_path, text = v
        ref_id = os.path.splitext(os.path.basename(ref_path))[0]
        reference_filename = 'reference-{}-{}.wav'.format(i, ref_id)
        reference_filepath = os.path.join(output_path, reference_filename)
        ref_wav, sr = librosa.load(ref_path)
        if sr == 24000:
            copyfile(ref_path, reference_filepath)
            print('copied reference file: {} -> {}'.format(ref_path, reference_filepath))
        else:
            ref_wav2 = get_audio(ref_path, meter, sample_rate=24000)
            sf.write(reference_filepath, ref_wav2, 24000)
            print('resampled reference file: {} ({}) -> {} (24000)'.format(ref_path, sr, reference_filepath))

    rtfs = [0 for _ in range(nsamples)]
    diffusion_steps = 10
    embedding_scale = 1
    alpha = 0.0
    beta = 1.0
    for i, v in enumerate(reference_dicts.values()):
        ref_path, text = v

        ref_s = compute_style(model, ref_path, device=device)

        ref_wav, _ = librosa.load(ref_path, sr=24000)
        duration_ref = len(ref_wav) / 24000

        wav, rtf, (duration_proc, duration_out) = run_infer(text, ref_s, diffusion_steps, embedding_scale, alpha=alpha, beta=beta)
        print('acoustic environment maintenance, unseen speakers, diffusion steps: {}, embedding scale: {}, alpha: {}, beta: {}'.format(
            diffusion_steps, embedding_scale, alpha, beta))
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

        ref_id = os.path.splitext(os.path.basename(ref_path))[0]
        output_filename = 'aem-{}-{}-{}-{}-{}-{}.wav'.format(i, ref_id, diffusion_steps, embedding_scale, alpha, beta)
        output_filepath = os.path.join(output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        ref_wav2 = get_audio(ref_path, meter)
        wav2 = get_audio(output_filepath, meter)
        ref_spkr_embedding = extract_spkr_embedding(ref_wav2, 24000)
        syn_spkr_embedding = extract_spkr_embedding(wav2, 24000)
        sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)
        print('sss: {:.3f}'.format(sss))

        meta = {'syn-wav': output_filepath, 'text': text, 'ref-wav': ref_path, 'dur-ref': duration_ref,
                'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': diffusion_steps,
                'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'seen-speaker': False,
                'sss': float(sss), 'hostname': hostname, 'gpu': gpu_info, 'exp-id': exp_id, 'topic': 'acoustic environment maintenance'}
        output_jsonfile = os.path.join(output_path, output_filename.replace('.wav', '.json'))
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2)

        rtfs[i] = rtf

    rtf_avg = np.mean(rtfs)
    print('average RTF (adaption, unseen speakers, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))

    #%% Zero-shot speaker adapatation 2 (Speaker's Emotion Maintenance)

    # maintain speaker's emotion (prosody) -> beta = 0.1 to make the speaker as closer to the reference as possible
    # while having some diversity through the slight timbre change (small alpha, e.g. alpha=0.3)

    exp_id = 12
    output_path = os.path.join(args.output_path, 'exp-{:02d}'.format(exp_id))
    os.makedirs(output_path, exist_ok=True)
    print('output path for exp {}: {}'.format(exp_id, output_path))

    recording_id = 'MARCHE_AssessmentTacticalEnvironment'
    voice = 'lada'
    stress = 'dictionary'
    wav_folder = '{}-{}'.format(voice, stress)
    ref_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts', 'outputs', 'sofw', 'espnet',
        recording_id, wav_folder)
    ref_wavpaths = sorted(glob.glob(os.path.join(ref_path, '*.wav')))
    keywords = ['.16000', '_paired', '_unpaired']
    ref_wavpaths = filter_path(ref_wavpaths, keywords)
    ref_wavpaths = ref_wavpaths[:5]

    texts = ["Hey guys, my name is Sam, and welcome to Prep Medic.",
             "This week's video, we are talking about the March algorithm.",
             "All right, guys, so the March algorithm is the assessment mode that we use in a tactical environment.",
             "A lot of other environments are starting to adapt it like civilian EMS, and essentially, it's a way of addressing life threats on our patients.",
             "So when we come up to a patient in a non-permissive environment such as a tactical situation, our number one priority is always going to be neutralizing the threat."]

    reference_dicts = {str(i+1): (ref_wavpaths[i], texts[i]) for i in range(5)}
    nsamples = len(reference_dicts)

    for i, v in enumerate(reference_dicts.values()):
        ref_path, text = v
        ref_id = os.path.splitext(os.path.basename(ref_path))[0]
        reference_filename = 'reference-{}-{}.wav'.format(i, ref_id)
        reference_filepath = os.path.join(output_path, reference_filename)
        ref_wav, sr = librosa.load(ref_path)
        if sr == 24000:
            copyfile(ref_path, reference_filepath)
            print('copied reference file: {} -> {}'.format(ref_path, reference_filepath))
        else:
            ref_wav2 = get_audio(ref_path, meter, sample_rate=24000)
            sf.write(reference_filepath, ref_wav2, 24000)
            print('resampled reference file: {} ({}) -> {} (24000)'.format(ref_path, sr, reference_filepath))

    rtfs = [0 for _ in range(nsamples)]
    diffusion_steps = 10
    embedding_scale = 1
    alpha = 0.0
    beta = 0.0
    for i, (emo, v) in enumerate(reference_dicts.items()):
        ref_path, text = v

        ref_path = os.path.join(os.getcwd(), ref_path)
        ref_s = compute_style(model, ref_path, device=device)

        ref_wav, _ = librosa.load(ref_path, sr=24000)
        duration_ref = len(ref_wav) / 24000

        wav, rtf, (duration_proc, duration_out) = run_infer(text, ref_s, diffusion_steps, embedding_scale, alpha=alpha, beta=beta)
        print('speaker emotion maintenance, unseen speakers, diffusion steps: {}, embedding scale: {}, alpha: {}, beta: {}'.format(
            diffusion_steps, embedding_scale, alpha, beta))
        print("RTF: {:.4f} ({:.4f} / {:.4f})".format(rtf, duration_proc, duration_out))

        ref_id = os.path.splitext(os.path.basename(ref_path))[0]
        output_filename = '{}-{}-{}-{}-{}-{}-{}.wav'.format(emo.lower(), i, ref_id, diffusion_steps, embedding_scale, alpha, beta)
        output_filepath = os.path.join(output_path, output_filename)
        sf.write(output_filepath, wav, 24000)
        print('wrote output file: {}'.format(output_filepath))

        ref_wav2 = get_audio(ref_path, meter)
        wav2 = get_audio(output_filepath, meter)
        ref_spkr_embedding = extract_spkr_embedding(ref_wav2, 24000)
        syn_spkr_embedding = extract_spkr_embedding(wav2, 24000)
        sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)
        print('sss: {:.3f}'.format(sss))

        meta = {'syn-wav': output_filepath, 'text': text, 'ref-wav': ref_path, 'dur-ref': duration_ref,
                'dur-proc': duration_proc, 'dur-syn': duration_out, 'rtf': rtf, 'diffusion-steps': diffusion_steps,
                'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'seen-speaker': False,
                'emo': emo, 'sss': float(sss), 'hostname': hostname, 'gpu': gpu_info, 'exp-id': exp_id, 'topic': 'speaker emotion maintenance'}
        output_jsonfile = os.path.join(output_path, output_filename.replace('.wav', '.json'))
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2)

        rtfs[i] = rtf

    rtf_avg = np.mean(rtfs)
    print('average RTF (adaptation, unseen speakers, diffusion_steps: {}, embedding_scale: {}): {:.4f}'.format(
        diffusion_steps, embedding_scale, rtf_avg))