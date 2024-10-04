# Inference demo for SOFW abbreviations
#
# This script was modified based on Demo/Inference_LibriTTS.py and Scratch/test1.py
#
# To run with Ipython interactively with certain CUDA device, please specify CUDA_VISIBLE_DEVICES, e.g.,
# CUDA_VISIBLE_DEVICES=1 python -m IPython --no-autoindent
# Then, use 'cuda:0' in the script 
#
# Zhenhao Ge, 2024-10-01

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
import phonemizer
from numpy.linalg import norm
from librosa.util import normalize

# set dirs
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()
from Utils.PLBERT.util import load_plbert

from infer_utils import length_to_mask
from infer_utils import compute_style_from_path as compute_style

punctuations = '.,:;?!'

def separate_punc(words, punctuations='.,:;?!'):
    """seperate punctuations as single words. This is helpful to identify the abbreviation
       at the end of the sentence, when it is appeneded with a punctuation, such as 'AWS' in 'AWS.'"""

    words2 = []
    for word in words:
        if word[-1] in punctuations:
            words2.append(word[:-1])
            words2.append(word[-1])
        else:
            words2.append(word) 
    return words2

def abbr_in_word(abbrs, word):
    """check if one of the abbreviations is in word, where abbreviation can be a partial word,
       e.g., abbrs is 'AT' and word is 'AT&L', then return True. This is helpful when the abbr
       is in a partial form, e.g., 'AT' is abbr, and it will recognize word with 'AT', such as 'AT&L'
       as an abbreviation"""

    for abbr in abbrs:
        if abbr in word:
            return True
    return False

def abbr2ps(abbr):
    """convert abbreviation to phone sequence letter by letter"""

    nletters = len(abbr)
    pss = ['' for _ in range(nletters)]
    for i in range(nletters):
        pss[i] = global_phonemizer.phonemize([abbr[i]])[0]
    pss = ''.join(pss)
    return pss 

def get_ps(text, abbr='', punctuations='.,:;?!'):
    """convert text to phone sequence (work with single abbreviation)"""

    text = text.strip()

    if abbr != '':

        # split text to words (treat punctuations as words)
        words = text.split()
        words = separate_punc(words, punctuations)

        # expand the abbrevation to the whole word containing the abbrevation (e.g. AT -> AT&L)
        for word in words:
            if abbr in word:
                abbr = word
                break

        idx = words.index(abbr)
        text_before = ' '.join(words[:idx])
        text_after = ' '.join(words[idx+1:])

        ps_before = global_phonemizer.phonemize([text_before])[0]
        ps_after = global_phonemizer.phonemize([text_after])[0]

        # # use alternative form for abbr (connect letter by '-' or '_')
        # ps_abbr = global_phonemizer.phonemize([abbr])[0]
        # ps = ps_before + ps_abbr + ps_after

        # feed in the abbreviation letter by letter
        ps = abbr2ps(abbr)

        # concatenate phone seqs
        ps = ps_before + pss + ps_after

    else:

        ps = global_phonemizer.phonemize([text])[0]

    ps = word_tokenize(ps)
    ps = ' '.join(ps)
    
    return ps

def get_ps2(text, abbrs=[], punctuations='.,:;?!'):
    """convert text to phone sequence (work with single and multiple abbreviations)"""

    text = text.strip()

    num_abbrs = len(abbrs)
    if num_abbrs > 0:
        
        # split text to words (treat punctuations as words)
        words = text.split()
        words = separate_punc(words, punctuations)

        # get list of segment(attribute, text)
        segments = []
        words2 = []
        for word in words:
            is_abbr = abbr_in_word(abbrs, word)
            if not is_abbr:
                words2.append(word)
            else:
                if len(words2) > 0:
                    text2 = ' '.join(words2)
                    segments.append(('regular', text2))
                segments.append(('abbr', word))
                words2 = []
        if len(words2) > 0:
            text2 = ' '.join(words2)
            segments.append(('regular', text2))

        # get phone sequence from segments based on their attributes (either 'regular', or 'abbr')
        num_segments = len(segments)
        pss = ['' for _ in range(num_segments)]
        for i, segment in enumerate(segments):
            if segment[0] == 'regular':
                pss[i] = global_phonemizer.phonemize([segment[1]])[0]
            elif segment[0] == 'abbr':
                pss[i] =  abbr2ps(segment[1])
        ps = ''.join(pss)

    else:

        ps = global_phonemizer.phonemize([text])[0]

    ps = word_tokenize(ps)
    ps = ' '.join(ps)
    
    return ps

def inference(text, ref_s, ps='', abbrs=[], ref_text='', s_prev=None, t=0.7, alpha=0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1, nframes_cut=50):

    # get phone seq (ps) from text if ps=='', otherwise, using existing ps, and ignore text
    if ps == '':
        # text = text.strip()
        # ps = global_phonemizer.phonemize([text])
        # ps = word_tokenize(ps[0])
        # ps = ' '.join(ps)
        ps = get_ps2(text, abbrs)

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

def parse_txt(txt_file):
    """parse the text file, which contains the segment id and the corresponding text per line"""
    fid2text = {}
    lines = open(txt_file, 'r').readlines()
    for i, line in enumerate(lines):
        parts = line.strip().split()
        fid = parts[0]
        text = ' '.join(parts[1:])
        fid2text[fid] = text
    return fid2text

def parse_args():

    usage = 'usage: inference SOFW sentences with abbreviations'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--config-file', type=str, help='config file')
    parser.add_argument('--txt-file', type=str, help='input text file with abbrevations')
    parser.add_argument('--output-dir', type-str, help='output dir')
    parser.add_argument('--device', type=str, default='cpu', help='gpu/cpu device')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # work_dir = os.getcwd()
    # args.config_file = os.path.join(work_dir, 'Configs', 'config_libritts.yml')
    # args.model_file = os.path.join(work_dir, 'Models', 'LibriTTS', 'epochs_2nd_00020.pth')
    # args.txt_file = os.path.join(work_dir, 'Data', 'SOFW_sentences_for_tts.sentids')
    # args.output_dir = os.path.join(work_dir, 'Outputs', 'Demo', 'SOFW')
    # args.data_dir = os.path.join(work_dir, 'Datasets', 'GigaSpeech-Zhenhao')
    # args.device = 'cuda:0' # 'cuda', 'cuda:x', or 'cpu'

    # set and create output dir (if needed)
    set_path(args.output_dir)

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

    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us', preserve_punctuation=True,  with_stress=True)

    # load config
    config = yaml.safe_load(open(args.config_file))

    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    # load BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    params_whole = torch.load(args.model_file, map_location='cpu')
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
    _ = [model[key].eval() for key in model]

    from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
  
    sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
    )

    ref_wav_rel_path = 'segment/youtube/P0000/YOU1000000038/YOU1000000038_S0000079.wav'
    ref_wav_file = os.path.join(args.data_dir, ref_wav_rel_path)
    assert os.path.isfile(ref_wav_file), f'ref wav file {ref_wav_file} does not exist!'

    ref_id = os.path.splitext(os.path.basename(ref_wav_file))[0]

    # compute reference style from ref wav file
    ref_s = compute_style(model, ref_wav_file, device=device)

    fid2text = parse_txt(args.txt_file)
    fids = sorted(fid2text.keys())

    # get the number of fids
    num_fids = len(fids)
    print('# of fids: {}'.format(num_fids))

    ref_text = ''
    s_prev = None
    t = 0.7
    alpha = 0.3
    beta = 0.7
    diffusion_steps = 5
    embedding_scale = 1
    nframes_cut = 50

    noutputs = num_fids
    for i in range(noutputs):

        fid = fids[i]

        out_wavfile = os.path.join(args.output_dir, f'{i:04d}_{fid}.wav')
        out_txtfile = os.path.join(args.output_dir, f'{i:04d}_{fid}.txt')

        if os.path.isfile(out_wavfile) and os.path.isfile(out_txtfile):
            print(f'{i}/{noutputs}: already exist, skip!')
            continue

        print(f'{i}/{noutputs}: text: "{text}"')

        # fid = 'DICT_A_02154_AGM_ARRW'
        # fid = 'DICT_U_00158_USD_AT'

        # get text from fid2text
        text = fid2text[fid]

        # text processing steps
        text = text.replace('-', ' ')

        # get abbreviations from fid
        abbrs = fid.split('_')[3:]
        num_abbrs = len(abbrs)

        # text = text.replace('AABFS', 'A A B F S')

        # text = text.replace('ABAA', 'A_B_A_A')
        # abbr = '_'.join([l for l in abbr])

        # ps = get_ps(text, abbrs[0]) # work on single abbreviation 
        ps = get_ps2(text, abbrs) # work on multiple abbreviations

        wav, _ = inference(text, ref_s,
            ps=ps,
            abbrs=abbrs,
            ref_text=ref_text,
            s_prev=s_prev,
            t=t,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
            nframes_cut=nframes_cut)

        # write the sentence wav file
        sf.write(out_wavfile, wav, 24000)
        # print(f'{i}/{noutputs}: wrote the sentence wav file {out_wavfile}')

        # write the sentence text file
        open(out_txtfile, 'w').writelines(text)
        # print(f'{i}/{noutputs}: wrote the sentence txt file {out_txtfile}')
