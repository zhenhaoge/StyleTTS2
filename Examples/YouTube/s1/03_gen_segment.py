# generate english audio segments using style-tts2 English model
#
# prepare the master reference file for speaker style first if needed
#
# Zhenhao Ge, 2024-06-27

import os
from pathlib import Path
import torch
import glob
import argparse
import pyloudnorm as pyln
from nltk.tokenize import word_tokenize
import librosa
import soundfile as sf
import numpy as np
import json

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

# set paths
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

from utils import str2bool
from audio import load_wavs
from Examples.dub_utils import get_ts, count_letters, split_text

from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()

from infer_utils import length_to_mask
from infer_utils import compute_style_from_path as compute_style
from infer_utils import compute_style_from_two_paths, compute_style_from_two_wavs

sr0 = 24000 # model sample rate

def expand_audio(audiofiles, idx, dur_min=3.0):
    """expand (i.e., include more) audio files from current index (to left, then to right),
       until the total duration exceed duration min requirement"""

    nfiles = len(audiofiles)

    # expand audio by including adjacent index (left index first, then right index)
    idx_offset = 0
    idx_left = idx - idx_offset
    idx_right = idx + idx_offset
    cond1 = idx_left >= 0
    cond2 = idx_right < nfiles
    idx_list = []
    dur_list = []
    while cond1 or cond2:
        if cond1:
            idx_list.append(idx_left)
            dur = librosa.get_duration(filename=audiofiles[idx_left])
            dur_list.append(round(dur,2))
            if sum(dur_list) > dur_min:
                break
        if cond2 and idx_right != idx_left:
            idx_list.append(idx_right)
            dur = librosa.get_duration(filename=audiofiles[idx_left])
            dur_list.append(round(dur,2))
            if sum(dur_list) > dur_min:
                break
        idx_offset += 1
        idx_left = idx - idx_offset
        idx_right = idx + idx_offset
        cond1 = idx_left >= 0
        cond2 = idx_right < nfiles

    # get idx list and the corresponding duration list w.r.t. the sorted order of idx list
    ii = sorted(range(len(idx_list)), key=lambda k: idx_list[k])
    idx_list_sorted = [idx_list[i] for i in ii]
    dur_list_sorted = [dur_list[i] for i in ii]
    
    audiofiles_sel = [audiofiles[i] for i in idx_list_sorted]
    dur_total = sum(dur_list_sorted)

    return audiofiles_sel, dur_total

def get_audio(speaker_file, meter, sample_rate=16000, mono_channel=True):

    audio, sr = sf.read(speaker_file) # load audio (shape: samples, channels)
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

def inference(text, ref_s, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens0 = textclenaer(ps) # list (#tokens)
    tokens0.insert(0, 0) # list (#tokens+1)
    tokens1 = torch.LongTensor(tokens0).to(device) # torch.Tensor ([#tokens+1])
    tokens = tokens1.unsqueeze(0) # torch.Tensor ([1, #tokens+1])
    
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

        ref = s_pred[:, :128]
        s = s_pred[:, 128:]

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

        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        out2 = out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later
        # out2 = out.squeeze().cpu().detach().numpy()[..., :-50] 

    return out2

def parse_args():
    usage = 'usage: generate inferenced segments using style-tts2 model'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--config-file', type=str, help='config file')
    parser.add_argument('--model-file', type=str, help='model file')
    parser.add_argument('--meta-file', type=str, help='meta file contains timestamps and english texts')
    parser.add_argument('--in-dir', type=str, help='input dir of the reference chinese audio segments')
    parser.add_argument('--out-dir', type=str, help='output dir of the inferenced english audio segments')
    parser.add_argument('--use-master-style', type=str2bool, nargs='?', const=True,
        default=False, help="true if use master style")
    parser.add_argument('--master-audio-file', type=str,
        help='master audio file with the concatenation of reliable segments')    
    parser.add_argument('--device', type=str, default='cpu', help='cpu/gpu device')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # args.config_file = os.path.join(work_dir, 'Models', 'LibriTTS', 'config.yml')
    # args.model_file = os.path.join(work_dir, 'Models', 'LibriTTS', 'epochs_2nd_00020.pth')
    # data_dir = os.path.join(work_dir, 'Datasets', 'YouTube')
    # account_id = 'dr-wang'
    # recording_id = '20210915'
    # dur_id = 'full'
    # args.meta_file = os.path.join(work_dir, 'Outputs', 'YouTube', account_id, recording_id, dur_id, 'meta', f'{recording_id}_L2_ts-text_v3.manual.csv')
    # args.in_dir = os.path.join(work_dir, 'Outputs', 'YouTube', account_id, recording_id, dur_id, 'v3.corrected')
    # args.out_dir = args.in_dir.replace('v3.corrected', 'v4.translated')
    # args.use_master_style = True
    # args.master_audio_file = os.path.join(data_dir, account_id, recording_id, dur_id, f'{recording_id}_L1_spk.wav')
    # args.device = 'cuda:0'

    # check file/dir existence
    assert os.path.isfile(args.config_file), f'config file: {args.config_file} does not exist!'
    assert os.path.isfile(args.model_file), f'model file: {args.model_file} does not exist!'
    assert os.path.isfile(args.meta_file), f'meta file: {args.meta_file} does not exist!'
    assert os.path.isfile(args.master_audio_file), f'master audio file: {args.master_audio_file} does not exist!'
    assert os.path.isdir(args.in_dir), f'input dir: {args.in_dir} does not exist!'

    # set and create output dir (if needed)
    set_path(args.out_dir, verbose=True)
    empty_dir(args.out_dir)

    # localize arguments
    config_file = args.config_file
    model_file = args.model_file
    meta_file = args.meta_file
    use_master_style = args.use_master_style
    master_audio_file = args.master_audio_file
    in_dir = args.in_dir
    out_dir = args.out_dir
    device = args.device

    # print arguments
    print(f'config file: {config_file}')
    print(f'model file: {model_file}')
    print(f'meta file: {meta_file}')
    print(f'master audio file: {master_audio_file}')
    print(f'input dir: {in_dir}')
    print(f'output dir: {out_dir}')
    print(f'device: {device}')

    # set GPU/CPU device
    if 'cuda' in device:
        if not torch.cuda.is_available():
            print ('device set to {}, but cuda is not available, so swithed to cpu'.format(args.device))
            device = 'cpu'
        else:
            device = device
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

    import phonemizer
    meter = pyln.Meter(sr0)
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us', preserve_punctuation=True,  with_stress=True)

    # load config
    config = yaml.safe_load(open(config_file))

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

    from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

    sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
    )

    # get reference audio files
    ref_audiofiles = sorted(glob.glob(os.path.join(in_dir, '*.wav')))

    # get segment info (idx, start_time, end_time, text)
    tuple_list = get_ts(meta_file, keep_ori_fid=False)
    assert len(ref_audiofiles) == len(tuple_list), \
        '# of segments in the reference audio files and text file mis-match!'
    nsegments = len(ref_audiofiles)
    print(f'# of segments: {nsegments}')

    if use_master_style:
        dur_master_lim = 15 # use first 15 seconds in the master audio file (the earlier seconds, the cleaner)
        wave_s_master, _ = librosa.load(master_audio_file, sr=sr0)
        audio_s_master, _ = librosa.effects.trim(wave_s_master, top_db=30)
        dur_master_total = len(audio_s_master)/sr0
        audio_s_master = audio_s_master[:int(min(dur_master_lim, dur_master_total)*sr0)]

    diffusion_steps = 5
    embedding_scale = 1
    alpha = 0.0
    beta = 0.3
    dur_min_s = 5.0 # min dur for the reference audio of style
    dur_min_p = 3.0 # min dur for the reference audio of predictor
    params_str = f'{diffusion_steps}-{embedding_scale}-{alpha}-{beta}'
    for i in range(nsegments):

        ref_audiofile = ref_audiofiles[i]
        
        # get style audio
        if use_master_style:
            audio_s = audio_s_master
        else:
            dur_ref_s = librosa.get_duration(filename=ref_audiofile)
            if dur_ref_s < dur_min_s:
                ref_audiofiles_included, dur_total = expand_audio(ref_audiofiles, i, dur_min=dur_min_s)
                wave_s = load_wavs(ref_audiofiles_included, sr0)
                # out_wav_file = os.path.join(work_dir, 'output.wav')
                # sf.write(out_wav_file, wave_s, sr0)
            else:
                wave_s, _ = librosa.load(ref_audiofile, sr=sr0)
            audio_s, _ = librosa.effects.trim(wave_s, top_db=30)

        # get predictor audio
        dur_ref_p = librosa.get_duration(filename=ref_audiofile)
        if dur_ref_p < dur_min_p:
            ref_audiofiles_included, dur_total = expand_audio(ref_audiofiles, i, dur_min=dur_min_p)
            wave_p = load_wavs(ref_audiofiles_included, sr0)
        else:
            wave_p, _ = librosa.load(ref_audiofile, sr=sr0)
        audio_p, _ = librosa.effects.trim(wave_p, top_db=30) 

        # get reference style from two sources
        ref_s = compute_style_from_two_wavs(model, audio_s, audio_p, device=device)

        # get timestamps and text
        idx, start_time, end_time, text = tuple_list[i]

        # count # of letters in text
        nletters = count_letters(text)

        # inference
        if nletters > 400: # too long, break into 2 parts
            text1, text2 = split_text(text)
            wav1 = inference(text1, ref_s, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
            wav2 = inference(text2, ref_s, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
            wav = np.concatenate((wav1, wav2))
        else:  
            wav = inference(text, ref_s, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
        dur = len(wav)/sr0

        # write the inferenced tts wav to file
        fid = f'{idx:04d}_{start_time:.2f}_{end_time:.2f}_{params_str}'
        syn_audiofilename = f'{fid}.wav'
        syn_audiofile = os.path.join(out_dir, syn_audiofilename)
        sf.write(syn_audiofile, wav, sr0)
        print(syn_audiofile)

        # get reference meta
        ref_jsonfile = ref_audiofile.replace('.wav', '.json')
        with open(ref_jsonfile) as f:
            meta0 = json.load(f)
 
        # write json file
        json_file = os.path.join(out_dir, f'{fid}.json')
        meta = {'fid': fid,
                'idx': idx,
                'start-time-l1': round(start_time, 2),
                'end-time-l1': round(start_time, 2),
                'duration-l1': round(end_time-start_time, 2),
                'duration-l2': round(dur, 2),
                'text-l1': meta0['text'],
                'text-l2': text}
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
