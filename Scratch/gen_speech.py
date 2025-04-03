# Generate TTS speech using StyleTTS2 based on the text generation scheme
#
# Experiment the quality and speed using word-level accumulation method to generate speech in a streaming setting
# 1st step: generate TTS speech samples 
#
# Zhenhao Ge, 2024-10-22

import os
from pathlib import Path
import torch
import glob
import subprocess
import shutil

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
import numpy as np

# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import librosa
import argparse
from nltk.tokenize import word_tokenize
import soundfile as sf
from shutil import copyfile
import json
import pyloudnorm as pyln
import phonemizer

from numpy.linalg import norm
from librosa.util import normalize

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
from audio import audioread, audiowrite

# from pyannote.audio import Inference
# spkr_embedding = Inference("pyannote/embedding", window="whole")

sr = 24000 # TTS audio sampling rate
punctuations = '.,:;?!'
meter = pyln.Meter(sr)

def gen_text_olw(text, win_size, step_size):
    """generate texts from text using overlapped windowing (gen_text_olw)"""

    words = text.split()
    nwords = len(words)
    
    texts = []
    for i in range(0,nwords,step_size):
        idx_start = i
        idx_end = i+win_size
        if idx_end < nwords:
            win_text = ' '.join(words[idx_start:idx_end])
            texts.append(win_text)

    return texts

def gen_text_acc(text, step_size=1):
    """generate texts from text accumulatively with step size"""

    words = text.split()
    nwords = len(words)

    texts = []
    cnt = 0
    for i in range(0, nwords, step_size):
        # print(i)
        texts.append(' '.join(words[:i+1]))
        cnt += 1
    nwords_last = len(texts[-1].split())
    if nwords_last < nwords:
        texts.append(text)

    return texts

def remove_punc(text, punctuations):

    words = text.split()
    nwords = len(words)
    
    words2 = []
    for i in range(nwords):
        if words[i] not in punctuations:
            words2.append(words[i])
    text2 = ' '.join(words2)
    
    return text2    

def inference(text, ref_s, ref_text='', s_prev=None, t=0.7, alpha=0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1, nframes_cut=50):

    # process text
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens0 = textclenaer(ps) # list (#tokens)
    tokens0.insert(0, 0) # list (#tokens+1)
    tokens1 = torch.LongTensor(tokens0).to(device) # torch.Tensor ([#tokens+1])
    tokens = tokens1.unsqueeze(0) # torch.Tensor ([1, #tokens+1])

    # process reference text if it exists
    ref_text = ref_text.strip()
    if len(ref_text) > 0:
        ps = global_phonemizer.phonemize([ref_text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        ref_tokens0 = textclenaer(ps)
        ref_tokens0.insert(0, 0)
        ref_tokens1 = torch.LongTensor(ref_tokens0).to(device)
        ref_tokens = ref_tokens1.unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 

        if len(ref_text) > 0: # with reference text
            s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device), 
                embedding=ref_bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s, # reference from the same speaker as the embedding
                num_steps=diffusion_steps).squeeze(1)            
        else: # without reference text
            s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device), 
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s, # reference from the same speaker as the embedding
                num_steps=diffusion_steps).squeeze(1)

        if s_prev is not None:
            # convex combination of previous and current style
            s_pred = t * s_prev + (1 - t) * s_pred

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
        out2 = out.squeeze().cpu().numpy()[..., :-nframes_cut] # weird pulse at the end of the model, need to be fixed later
        # out2 = out.squeeze().cpu().detach().numpy()[..., :-50]

    return out2, s_pred

def parse_args():

    usage = 'usage: generate TTS speech for word-level accumulation method'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--config-path', type=str, help='config path')
    parser.add_argument('--model-path', type=str, help='model path')
    parser.add_argument('--output-path', type=str, help='root output path')
    parser.add_argument('--data-path', type=str, help='data path')
    parser.add_argument('--ref-wav-rel-path', type=str, help='reference wav relative path')
    parser.add_argument('--exp-id', type=str, help='exp id')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cpu', help='gpu/cpu device')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # work_path = os.getcwd() # e.g., '/home/users/zge/code/repo/style-tts2'
    # args.config_path = os.path.join(work_path, 'Configs', 'config_libritts.yml')
    # args.model_path = os.path.join(work_path, 'Models', 'LibriTTS', 'epochs_2nd_00020.pth')
    # args.output_path = os.path.join(work_path, 'Outputs', 'Scratch', 'LibriTTS')
    # args.data_path = os.path.join(work_path, 'Datasets', 'GigaSpeech-Zhenhao')
    # args.ref_wav_rel_path = 'segment/youtube/P0000/YOU1000000038/YOU1000000038_S0000079.wav'
    # args.exp_id = 2
    # args.seed = 1
    # args.device = 'cuda:0' # 'cuda', 'cuda:x', or 'cpu'

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

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

    # set the output path
    if convertible_to_integer(args.exp_id):
        args.exp_id = int(args.exp_id)
        output_folder = f'exp-{args.exp_id:02d}'
    else:
        output_folder = f'gs-{args.exp_id}'
    output_path = os.path.join(args.output_path, output_folder) 
    if not os.path.isdir(output_path):
        print(f'creating new output dir: {output_path}')
        os.makedirs(output_path)
    else:
        print(f'using existing output dir: {output_path}')    
    # print('output path for exp {}: {}'.format(exp_id, output_path))

    # set the reference wav file
    ref_wav_file = os.path.join(args.data_path, args.ref_wav_rel_path)
    assert os.path.isfile(ref_wav_file), f'ref wav file {ref_wav_file} does not exist!'

    # set the reference text file (associated with the reference wav file)
    ref_txt_file = ref_wav_file.replace('.wav', '.txt')
    assert os.path.isfile(ref_txt_file), f'ref txt file {ref_txt_file} does not exist!'

    # get the reference text
    lines = open(ref_txt_file, 'r').readlines()
    assert len(lines) == 1, f'more than 1 line in the text file: {ref_txt_file}'
    ref_text = lines[0].strip()

    # copy the refrence wav and text files to the output dir
    filename_wav = os.path.basename(ref_wav_file).replace('.wav', '_reference.wav')
    ref_wav_file2 = os.path.join(output_path, filename_wav)
    shutil.copyfile(ref_wav_file, ref_wav_file2)
    print(f'{ref_wav_file} -> {ref_wav_file2}')

    filename_txt = os.path.basename(ref_txt_file).replace('.txt', '_reference.txt')
    ref_txt_file2 = os.path.join(output_path, filename_txt)
    shutil.copyfile(ref_txt_file, ref_txt_file2)
    print(f'{ref_txt_file} -> {ref_txt_file2}')

    ref_id = os.path.splitext(os.path.basename(ref_wav_file))[0]

    # compute reference style from ref wav file
    ref_s = compute_style(model, ref_wav_file, device=device)

    # get windowed texts
    # win_size, step_size = 5, 1
    # texts = gen_text_olw(text, win_size, step_size)
    texts = gen_text_acc(ref_text, step_size=2)
    ntexts = len(texts)
    ndigits = len(str(ntexts))
    print(f'# of texts: {ntexts}')

    durations_proc = [0 for _ in range(ntexts)]
    durations_out = [0 for _ in range(ntexts)]
    rtfs = [0 for _ in range(ntexts)]
    wavs = ['' for _ in range(ntexts)]
    s_preds = ['' for _ in range(ntexts)]

    # perform inference and get processing durations
    ref_text = ''
    s_prev = None
    t = 0.7
    alpha = 0.3
    beta = 0.7
    diffusion_steps = 5
    embedding_scale = 1
    nframes_cut = 50
    for i, text in enumerate(texts):
        
        # print the text
        print(text)

        start_time = time.time()
        wavs[i], s_preds[i] = inference(text, ref_s, 
                        ref_text=ref_text,
                        s_prev=s_prev,
                        t=t,
                        alpha=alpha,
                        beta=beta,
                        diffusion_steps=diffusion_steps,
                        embedding_scale=embedding_scale,
                        nframes_cut=nframes_cut)
        end_time = time.time()
        durations_proc[i] = end_time - start_time

    # compute the output durations and rtfs
    for i in range(ntexts):
        durations_out[i] = len(wavs[i]) / sr
        rtfs[i] = durations_proc[i] / durations_out[i]

    # construct the output audio, text (unpunctuated), and meta json files
    out_wavfiles = ['' for _ in range(ntexts)]
    out_txtfiles = ['' for _ in range(ntexts)]
    out_jsonfiles = ['' for _ in range(ntexts)]
    for i in range(ntexts):

        # construct the audio files 
        filename_wav = '{}-{}-{}-{}-{}-{}.wav'.format(ref_id, i, diffusion_steps, embedding_scale, alpha, beta)
        out_wavfiles[i] = os.path.join(output_path, filename_wav)

        # construct the text files
        filename_txt = filename_wav.replace('.wav', '.txt')
        out_txtfiles[i] = os.path.join(output_path, filename_txt)

        # construct the json files
        filename_json = filename_wav.replace('.wav', '.json')
        out_jsonfiles[i] = os.path.join(output_path, filename_json)

    # generaete the audio, text (unpunctuated), and meta json files
    for i in range(ntexts):

        # write the wav file
        sf.write(out_wavfiles[i], wavs[i], 24000)

        # get unpunctuated text
        text_nopunct = remove_punc(texts[i], punctuations)
        # write unpunctuated text to the text file
        open(out_txtfiles[i], 'w').writelines(text_nopunct + '\n')

        meta = {'ref-id': ref_id, 'text-nopunct': text_nopunct, 'diffusion-steps': diffusion_steps,
                'embedding-scale': embedding_scale, 'alpha': alpha, 'beta': beta, 'hostname': hostname,
                'gpu': gpu_info, 'exp-id': args.exp_id, 'dur-proc': durations_proc[i], 'dur-out': durations_out[i],
                'rtf': rtfs[i]}
        with open(out_jsonfiles[i], 'w') as fp:
            json.dump(meta, fp, indent=2) 
