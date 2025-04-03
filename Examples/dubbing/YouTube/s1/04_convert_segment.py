# convert english tts audio segments to the target speaker's voice using free-vc
#
# Zhenhao Ge, 2024-06-28

import os, sys
from pathlib import Path
import torch
import argparse
import glob
import librosa
import soundfile as sf
import json

# set paths
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'free-vc')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

# add current work dir with high priority
sys.path.insert(0, work_dir)

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from speaker_encoder.voice_encoder import SpeakerEncoder
import logging
logging.getLogger('numba').setLevel(logging.WARNING)
from scripts.utils import set_path, empty_dir

def parse_args():
    usage = "usage: convert audio segments to the target speaker's voice using free-vc"
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--config-file', type=str, help='config file')
    parser.add_argument('--model-file', type=str, help='model file')
    parser.add_argument('--src-dir', type=str, help='source dir for the content and the voice to be converted from')
    parser.add_argument('--tgt-dir', type=str, help='target dir for the voice to be converted to')
    parser.add_argument('--out-dir', type=str, help='output dir for the content with converted voice')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # args.config_file = os.path.join(work_dir, 'configs', 'freevc-24.json')
    # args.model_file = os.path.join(work_dir, 'checkpoints', '24kHz', 'freevc-24.pth')
    # account_id = 'laoming'
    # recording_id = '20220212'
    # tts_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
    # args.src_dir = os.path.join(tts_dir, 'Outputs', 'YouTube', account_id, recording_id, 'v4.translated')
    # args.tgt_dir = os.path.join(tts_dir, 'Outputs', 'YouTube', account_id, recording_id, 'v3.corrected')
    # args.out_dir = os.path.join(tts_dir, 'Outputs', 'YouTube', account_id, recording_id, 'v5.converted')

    # check file/dir existence
    assert os.path.isfile(args.config_file), f'config file: {args.config_file} does not exist!'
    assert os.path.isfile(args.model_file), f'model file: {args.model_file} does not exist!'
    assert os.path.isdir(args.src_dir), f'source dir: {args.src_dir} does not exist!'
    assert os.path.isdir(args.tgt_dir), f'target dir: {args.tgt_dir} does not exist!'

    # create the output dir
    set_path(args.out_dir, verbose=True)
    empty_dir(args.out_dir)

    # localize arguments
    config_file = args.config_file
    model_file = args.model_file
    src_dir = args.src_dir
    tgt_dir = args.tgt_dir
    out_dir = args.out_dir

    # print out arguments
    print(f'config file: {config_file}')
    print(f'model file: {model_file}')
    print(f'source dir: {src_dir}')
    print(f'target dir: {tgt_dir}')
    print(f'output dir: {out_dir}')

    hps = utils.get_hparams_from_file(config_file)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(model_file, net_g, None)

    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(0)

    if hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder(os.path.join(work_dir, 'speaker_encoder/ckpt/pretrained_bak_5805000.pt'))

    src_audiofiles = sorted(glob.glob(os.path.join(src_dir, '*.wav')))
    tgt_audiofiles = sorted(glob.glob(os.path.join(tgt_dir, '*.wav')))
    assert len(src_audiofiles) == len(tgt_audiofiles), '# of audio files mis-match in the source and target dirs!'
    nsegments = len(src_audiofiles)
    print(f'# of audio segments: {nsegments}')

    for i, (src_file, tgt_file) in enumerate(zip(src_audiofiles, tgt_audiofiles)):

        # get idx, start time and end time from source filename
        parts = os.path.splitext(os.path.basename(src_file))[0].split('_')
        idx_src = int(parts[0])
        start_time_src = round(float(parts[1]), 2)
        end_time_src = round(float(parts[2]), 2)

        # get idx, start time and end time from target filename
        parts = os.path.splitext(os.path.basename(tgt_file))[0].split('_')
        idx_tgt = int(parts[0])
        start_time_tgt = round(float(parts[1]), 2)
        end_time_tgt = round(float(parts[2]), 2)

        assert idx_src == idx_tgt and start_time_src == start_time_tgt and end_time_src == end_time_tgt, \
            'timestamps in the source and target segments mismatch!'
        idx = idx_src
        start_time = start_time_src
        end_time = end_time_src
        del idx_src, idx_tgt, start_time_src, start_time_tgt, end_time_src, end_time_tgt

        # target
        wav_tgt, _ = librosa.load(tgt_file, sr=hps.data.sampling_rate)
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
        if hps.model.use_spk:
            g_tgt = smodel.embed_utterance(wav_tgt)
            g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
        else:
            wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
            mel_tgt = mel_spectrogram_torch(
                wav_tgt, 
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax)

        # source
        wav_src, _ = librosa.load(src_file, sr=hps.data.sampling_rate)
        wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
        c = utils.get_content(cmodel, wav_src)

        if hps.model.use_spk:
            audio = net_g.infer(c, g=g_tgt)
        else:
            audio = net_g.infer(c, mel=mel_tgt)
        audio = audio[0][0].data.cpu().float().numpy()
        duration = len(audio)/24000

        # write audio to file
        fid = f'{idx:04d}_{start_time:.2f}_{end_time:.2f}'
        out_file = os.path.join(out_dir, f'{fid}.wav')
        sf.write(out_file, audio, 24000)
        print(out_file)

        # read meta from the source json file
        src_json_file = src_file.replace('.wav', '.json')
        with open(src_json_file, encoding='utf-8') as f:
            meta0 = json.load(f)

        # write meta info to json file
        json_file = os.path.join(out_dir, f'{fid}.json')
        meta = {'fid': fid,
                'idx': idx,
                'start-time-zh': round(start_time, 2),
                'end-time-zh': round(end_time, 2),
                'duration-zh': round(end_time-start_time, 2),
                'duration-en': round(duration, 2),
                'text-zh': meta0['text-zh'],
                'text-en': meta0['text-en']}
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)        
