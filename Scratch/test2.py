import os
from pathlib import Path
import torch
import glob
import subprocess

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
align_path = os.path.join(home_path, 'code', 'repo', 'gentle')

from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()

from infer_utils import length_to_mask
from infer_utils import compute_style_from_path as compute_style
from audio import audioread, audiowrite

# from pyannote.audio import Inference
# spkr_embedding = Inference("pyannote/embedding", window="whole")

punctuations = '.,:;?!'
meter = pyln.Meter(24000)

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

def get_float_precision(number):
    # Convert the float to string
    float_str = str(number)
    # Find the decimal point
    if '.' in float_str:
        # Split on the decimal point and count the digits after it
        return len(float_str.split('.')[1])
    else:
        # No decimal point means precision is 0
        return 0 

def round_snap(raw_time, snap_dur=0.0125):
    """round toward multiples of snap_dur"""
    precision = get_float_precision(snap_dur)
    rounded_time = round(round(raw_time/snap_dur) * snap_dur, precision)
    return rounded_time

def extract_word_ts(tgfile, idx_word_start=0, idx_word_end=-1, round_method='regular', round_precision=2, snap_dur=0.125):
    """extract word and timestamp tuples from the TextGrid file in json format"""
    # read the TextGrid file with alignment
    with open(tgfile, 'r') as json_file:
        data_dict = json.load(json_file)
        # get the # of words    
        nwords = len(data_dict['words'])
        # update the end word idx
        if idx_word_end == -1:
            idx_word_end = nwords

    # wavfile = tgfile.replace('.TextGrid', '.wav')
    # duration = librosa.get_duration(filename=wavfile)

    # extract the word timestamps
    word_ts = []
    for i in range(idx_word_start, idx_word_end):
        word = data_dict['words'][i]
        alignedWord = word['alignedWord']
        start_time_raw = float(word['start'])
        end_time_raw = float(word['end'])
        if round_method == 'regular':
            start_time = round(start_time_raw, round_precision)
            end_time = round(end_time_raw, round_precision)
        elif round_method == 'snap':
            start_time = round_snap(start_time_raw, snap_dur)
            end_time = round_snap(end_time_raw, snap_dur)
        word_ts.append([alignedWord, start_time, end_time])

    # adjust the start time of the first word based on the gap to the previous word
    if idx_word_start > 0:
        word_previous = data_dict['words'][idx_word_start-1]
        end_time_previous_raw = float(word_previous['end'])
        end_time_previous = round(end_time_previous_raw, round_precision)
        start_time_first = word_ts[0][1]
        start_time_first_updated_raw = (start_time_first + end_time_previous) / 2
        if round_method == 'regular':
            start_time_first_updated = round(start_time_first_updated_raw, round_precision)
        elif round_method == 'snap':
            start_time_first_updated = round_snap(start_time_first_updated_raw, snap_dur)
            word_ts[0][1] = start_time_first_updated
    
    # adjust the end time of the last word based on the gap of the next word
    if idx_word_end < nwords:
        word_next = data_dict['words'][idx_word_end]
        start_time_next = round(float(word_next['start']), round_precision)
        end_time_last = word_ts[-1][2]
        end_time_last_updated_raw = (start_time_next + end_time_last) / 2
        if round_method == 'regular':
            end_time_last_updated = round(end_time_last_updated_raw, round_precision)
        elif round_method == 'snap':
            end_time_last_updated = round_snap(end_time_last_updated_raw, snap_dur)
        word_ts[-1][2] = end_time_last_updated

    return word_ts

def get_tgfile_tuples(texts, tgfiles, nwords_future):

    ntexts = len(texts)
    ntgfiles = len(tgfiles)
    assert ntexts == ntgfiles, 'len(texts) and len(tgfiles) mis-match!'

    # get the start idx of texts (the first text with nwords > nwords_future)
    idx_start = [i for i, text in enumerate(texts) if len(text.split())>nwords_future][0]

    words_upto_current = [[] for _ in range(ntexts-idx_start)]
    for i in range(idx_start, ntexts):
        words_upto_current[i-idx_start] = texts[i].split()[:-nwords_future]

    # # sanity check: print out words upto the current
    # for i, words in enumerate(words_upto_current):
    #     print(f'{i}/{ntexts-idx_start}: ' + ' '.join(words))

    words_current = ['' for _ in range(ntexts-idx_start)]
    words_current[0] = words_upto_current[0]

    # fid_idxs_tuples = ['' for _ in range(ntexts-idx_start)]
    # fid = os.path.splitext(os.path.basename(out_wavfiles[idx_start]))[0]
    # idx_word_start = 0
    # idx_word_end = len(words_upto_current[0])
    # fid_idxs_tuples[0] = (fid, idx_word_start, idx_word_end)

    for i in range(idx_start+1, ntexts):

        idx_word_start = len(words_upto_current[i-idx_start-1])
        idx_word_end = len(words_upto_current[i-idx_start])
        # print(f'i:{i}, idx: {idx_word_start}, {idx_word_end}')
        words_current[i-idx_start] = words_upto_current[i-idx_start][idx_word_start:idx_word_end]

        # fid = os.path.splitext(os.path.basename(out_wavfiles[i]))[0]
        # fid_idxs_tuples[i-idx_start] = (fid, idx_word_start, idx_word_end)

    # # sanity check: print out the final current words
    # for i, words in enumerate(words_current):
    #     print(f'{i}/{ntexts-idx_start}: ' + ' '.join(words))

    # get tgfile tuples (tgfile, words, idx_word_start, idx_word_end)
    ntexts2 = ntexts - idx_start
    assert len(words_current) == ntexts2, 'check words_current!'
    tgfile_tuples = [() for _ in range(ntexts2)]
    for i in range(ntexts2):
        if i == 0:
            idx_word_start = 0
            idx_word_end = len(words_upto_current[i])
        else:
            idx_word_start = len(words_upto_current[i-1])
            idx_word_end = len(words_upto_current[i])
        tgfile_tuples[i] = [tgfiles[i+idx_start], words_current[i], idx_word_start, idx_word_end]

    return tgfile_tuples         

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

def update_crossfade(dur0, dur1, crossfade):
    dur = min(dur0, dur1)
    cf = min(int(dur/2), crossfade)
    return cf

def crossfade_segment(audio0, audio1, crossfade, type='regular'):

    # update crossfade duration based on the durations of aduio0 and audio1
    dur0 = int(audio0.duration_seconds * 1000)
    dur1 = int(audio1.duration_seconds * 1000)
    cf = update_crossfade(dur0, dur1, args.crossfade)
    if type == 'regular':
        combined = audio0.append(audio1, crossfade=cf)
    elif type == 'samedur':
        crossfade_part = audio0[-cf:].append(audio1[:cf], crossfade=cf)
        audio0_nonoverlapped = audio0[:-cf]
        audio1_nonoverlapped = audio1[cf:]
        # # combine 3 parts (has data format issue)
        # combined = audio0_nonoverlapped + crossfade_part + audio1_nonoverlapped
        # combine 3 parts (data format issue fixed)
        crossfade_part2 = AudioSegment(crossfade_part._data,
            frame_rate=audio0.frame_rate, sample_width=audio0.sample_width, channels=audio0.channels)
        if isinstance(audio0_nonoverlapped._data, np.ndarray):
            audio0_nonoverlapped2 = AudioSegment(audio0_nonoverlapped._data.tobytes(),
                frame_rate=audio0.frame_rate, sample_width=audio0.sample_width, channels=audio0.channels)
        else:
           audio0_nonoverlapped2 = AudioSegment(audio0_nonoverlapped._data,
                frame_rate=audio0.frame_rate, sample_width=audio0.sample_width, channels=audio0.channels)
        audio1_nonoverlapped2 = AudioSegment(audio1_nonoverlapped._data.tobytes(),
            frame_rate=audio1.frame_rate, sample_width=audio1.sample_width, channels=audio1.channels)
        combined = audio0_nonoverlapped2 + crossfade_part2 + audio1_nonoverlapped2
        # # combine 3 parts (alternative method, does not work)
        # crossfade_part_samples = crossfade_part.get_array_of_samples()
        # crossfade_part_array = np.array(crossfade_part_samples)
        # crossfade_part2 = crossfade_part
        # crossfade_part2._data = crossfade_part_array
        # combined = audio0_nonoverlapped + crossfade_part2 + audio1_nonoverlapped
        # dur_combined = int(combined.duration_seconds * 1000)
        # print(f'duration after crossfade ({type}): {dur_combined} ms')
    return combined

def parse_args():
    usage = 'usage: test2'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--config-path', type=str, help='config path')
    parser.add_argument('--model-path', type=str, help='model path')
    parser.add_argument('--output-path', type=str, help='output path')
    parser.add_argument('--data-path', type=str, help='data path')
    parser.add_argument('--crossfade', type=int, help='crossfade duration in ms')
    parser.add_argument('--device', type=str, default='cpu', help='gpu/cpu device')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    args = argparse.ArgumentParser()

    work_path = os.getcwd() # e.g., '/home/users/zge/code/repo/style-tts2'
    args.config_path = os.path.join(work_path, 'Configs', 'config_libritts.yml')
    args.model_path = os.path.join(work_path, 'Models', 'LibriTTS', 'epochs_2nd_00020.pth')
    args.output_path = os.path.join(work_path, 'Outputs', 'Scratch', 'LibriTTS')
    args.data_path = os.path.join(work_path, 'Datasets', 'GigaSpeech-Zhenhao')
    args.crossfade = 50
    args.device = 'cuda:0' # 'cuda', 'cuda:x', or 'cpu'

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

    exp_id = 1
    output_path = os.path.join(args.output_path, f'exp-{exp_id:02d}')
    os.makedirs(output_path, exist_ok=True)
    print('output path for exp {}: {}'.format(exp_id, output_path))

    # text = 'StyleTTS 2 is a text to speech model that leverages style diffusion and adversarial training with large speech language models to achieve human level text to speech synthesis.'

    ref_wav_rel_path = 'segment/youtube/P0000/YOU1000000038/YOU1000000038_S0000079.wav'
    ref_wav_file = os.path.join(args.data_path, ref_wav_rel_path)
    assert os.path.isfile(ref_wav_file), f'ref wav file {ref_wav_file} does not exist!'

    ref_txt_file = ref_wav_file.replace('.wav', '.txt')
    assert os.path.isfile(ref_txt_file), f'ref txt file {ref_txt_file} does not exist!'

    lines = open(ref_txt_file, 'r').readlines()
    assert len(lines) == 1, f'more than 1 line in the text file: {ref_txt_file}'
    ref_text = lines[0].strip()

    ref_id = os.path.splitext(os.path.basename(ref_wav_file))[0]

    # compute reference style from ref wav file
    ref_s = compute_style(model, ref_wav_file, device=device)

    # get windowed texts
    # win_size, step_size = 5, 1
    # texts = gen_text_olw(text, win_size, step_size)
    texts = gen_text_acc(ref_text, step_size=2)
    ntexts = len(texts)

    durations_proc = [0 for _ in range(ntexts)]
    durations_out = [0 for _ in range(ntexts)]
    rtfs = [0 for _ in range(ntexts)]
    wavs = ['' for _ in range(ntexts)]
    s_preds = ['' for _ in range(ntexts)]

    ref_text = ''
    s_prev = None
    t = 0.7
    alpha = 0.3
    beta = 0.7
    diffusion_steps = 5
    embedding_scale = 1
    nframes_cut = 50
    for i, text in enumerate(texts):
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
        durations_out[i] =  len(wavs[i])
        rtfs[i] = durations_proc[i] / durations_out[i]

    # generaete the audio and text (unpunctuated) files
    out_wavfiles = ['' for _ in range(ntexts)]
    out_txtfiles = ['' for _ in range(ntexts)]
    for i in range(ntexts):

        # write the wav file
        filename = 'basic-{}-{}-{}-{}-{}-{}.wav'.format(ref_id, i, diffusion_steps, embedding_scale, alpha, beta)
        out_wavfiles[i] = os.path.join(output_path, filename)
        sf.write(out_wavfiles[i], wavs[i], 24000)
        
        # write the txt file
        filename = filename.replace('.wav', '.txt')
        out_txtfiles[i] = os.path.join(output_path, filename)
        # get unpunctuated text
        text_nopunct = remove_punc(texts[i], punctuations)
        # write unpunctuated text
        open(out_txtfiles[i], 'w').writelines(text_nopunct + '\n')

    out_tgfiles = ['' for _ in range(ntexts)]
    for i in range(ntexts):

        # write the TextGrid alignment file
        out_tgfiles[i] = out_wavfiles[i].replace('.wav', '.TextGrid')
        command = ['python', os.path.join(align_path, 'align.py'), '--output', out_tgfiles[i], out_wavfiles[i], out_txtfiles[i]]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check if the command was successful
        if result.returncode == 0:
            print("Command executed successfully")
            print(result.stdout)
        else:
            print("Error occurred")
            print(result.stderr)

    # get tuple list for (.TextGrid file, current words, idx_word_start, idx_word_end)
    nwords_future = 2 # number of words in the future (avoid the distortion on the ending words)
    tgfile_tuples = get_tgfile_tuples(texts, out_tgfiles, nwords_future)

    # include future words for the last tgfile_tuples
    tgfile_last = tgfile_tuples[-1][0]
    idx_word_start = tgfile_tuples[-1][2]
    words_ts_last = extract_word_ts(tgfile_last, idx_word_start=idx_word_start, idx_word_end=-1, round_method='regular', round_precision=2)
    words_last = [sublst[0] for sublst in words_ts_last]
    tgfile_tuples[-1][1] = words_last
    tgfile_tuples[-1][3] = idx_word_start + len(words_last)

    # experiment with crossfade and tolerance (overide crossfade)
    args.crossfade = 5
    tolerance = args.crossfade/2
    # tolerance = args.crossfade
    # tolerance = 0

    idx_start = [i for i, text in enumerate(texts) if len(text.split())>nwords_future][0]
    ntexts2 = ntexts - idx_start

    # extract the words from tgfiles based on tgfile_tuples
    seg_list = [{} for _ in range(ntexts2)]

    for i in range(ntexts2):
        tgfile, words, idx_word_start, idx_word_end = tgfile_tuples[i]
        # words_ts = extract_word_ts(tgfile, idx_word_start=0, idx_word_end=-1)
        # words_ts_sel = word_ts[idx_word_start:idx_word_end]
        words_ts_sel = extract_word_ts(tgfile, idx_word_start, idx_word_end, round_method='regular', round_precision=2)
        # words_ts_sel = extract_word_ts(tgfile, idx_word_start, idx_word_end, round_method='snap', snap_dur=0.0125)

        start_time = max(0, words_ts_sel[0][1] - tolerance/1000)
        end_time = words_ts_sel[-1][2] + tolerance/1000
        duration = end_time - start_time

        wavfile = tgfile.replace('.TextGrid', '.wav')
        assert os.path.isfile(wavfile), f'wav file: {wavfile} does not exist!'

        # method 1: read/write using audioread/audiowrite
        data, params = audioread(wavfile, start_time, duration)
        seg_list[i] = {'data': data, 'params': params}
        seg_wavfile1 = wavfile.replace('.wav', f'_{idx_word_start}-{idx_word_end}_t{tolerance}_M1.wav')
        audiowrite(seg_wavfile1, data, params)
        print(f'{i}/{ntexts2}: wrote {seg_wavfile1}')

        # method 2: read/write using lobrosa.load and sf.write
        y, _ = librosa.load(wavfile, sr=24000, offset=start_time, duration=duration)
        seg_list[i]['y'] = y
        seg_wavfile2 = wavfile.replace('.wav', f'_{idx_word_start}-{idx_word_end}_t{tolerance}_M2.wav')
        sf.write(seg_wavfile2, y, 24000)
        print(f'{i}/{ntexts2}: wrote {seg_wavfile2}')

    # method 1: write the concatenated audio (no crossfading) using audiowrite
    wavfile_concat1 = os.path.join(output_path, f'basic-{ref_id}_concat_t{tolerance}_M1.wav')
    data_list = [seg['data'] for seg in seg_list]
    nframes_list = [seg['params'][3] for seg in seg_list]
    data = np.concatenate(data_list)
    nframes = sum(nframes_list)
    params[3] = nframes
    audiowrite(wavfile_concat1, data, params)
    print(f'wrote {wavfile_concat1}')

    # method 2: write the concatenated audio (no crossfading) using soundfile.write
    wavfile_concat2 = os.path.join(output_path, f'basic-{ref_id}_concat_t{tolerance}_M2.wav')
    ys = [seg['y'] for seg in seg_list]
    y = np.concatenate(ys)
    sf.write(wavfile_concat2, y, 24000)
    print(f'wrote {wavfile_concat2}')

    # set crossfade type
    cf_type = 'regular' # 'regular' or 'samedur'

    # method 1: create concatenated audio with crossfade from audiowrite output
    channels, sample_width, frame_rate = params[:3]
    combined = AudioSegment(data=data_list[0].tobytes(), sample_width=sample_width, frame_rate=frame_rate, channels=channels)
    for i, data in enumerate(data_list[1:]):
        seg = AudioSegment(data=data.tobytes(), sample_width=sample_width, frame_rate=frame_rate, channels=channels)
        combined = crossfade_segment(combined, seg, args.crossfade, type=cf_type)
    wavfile_concat_cf = os.path.join(output_path, f'basic-{ref_id}_concat_cf{args.crossfade}_t{tolerance}_M1.wav')
    combined.export(wavfile_concat_cf, format="wav") 
