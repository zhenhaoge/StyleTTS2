# convert audio files in GigaSpeech from opus to wav format
#
# # audio files: 40368 (only 38131 are included in the master json file)
#
# reference: https://github.com/SpeechColab/GigaSpeech/blob/main/utils/opus_to_wav.py
#
# Zhenhao Ge, 2024-06-17

import os, sys
from pathlib import Path
import glob
import opuspy
import librosa
import json
import soundfile as sf
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

# from audio import convert_opus2wav

def convert_opus2wav(opus_file, wav_file, target_sr=16000, rm_opus=False):
    cmd = f'ffmpeg -y -i {opus_file} -ac 1 -ar {target_sr} {wav_file}'
    try:
      os.system(cmd)
    except:
      sys.exit(f'Failed to run the cmd: {cmd}')
    if rm_opus is True:
      os.remove(opus_file)

def convert_audio(file_pair, refresh=False):
    opus_file, wav_file = file_pair
    wav_dir = os.path.dirname(wav_file)
    os.makedirs(wav_dir, exist_ok=True)
    if not os.path.isfile(wav_file) or refresh:
        convert_opus2wav(opus_file, wav_file, target_sr=16000)

# set opus path
opus_path = os.path.join(work_path, 'Datasets', 'GigaSpeech', 'audio')
assert os.path.isdir(opus_path), f'opus path: {opus_path} does not exist!'

# set wav path
wav_path = os.path.join(work_path, 'Datasets', 'GigaSpeech-Zhenhao', 'audio')
if os.path.isdir(wav_path):
    print(f'use existing dir: {wav_path}')
else:
    os.makedirs(wav_path)
    print(f'created new dir: {wav_path}')

# get list of opus files
opus_files = sorted(glob.glob(os.path.join(opus_path, '**', "*.opus"), recursive=True))
num_opus_files = len(opus_files)
print(f'# of opus files: {num_opus_files}') # 40368 opus files (only 38131 are included in master json file)

# get the corresponding list of wav files
wav_files = [f.replace(opus_path, wav_path).replace('.opus', '.wav') for f in opus_files]

# for i, (opus_file, wav_file) in tqdm(enumerate(zip(opus_files, wav_files))):
#     # print(f'{i+1}/{num_opus_files}: {opus_file} -> {wav_file}')
#     wav_dir = os.path.dirname(wav_file)
#     os.makedirs(wav_dir, exist_ok=True)
#     if not os.path.isfile(wav_file):
#         convert_opus2wav(opus_file, wav_file, target_sr=16000)

pool = Pool()
pool.map(convert_audio, zip(opus_files, wav_files))

# check if the generated wav file has the duration match the duration in the meta file

# set meta path
meta_path = os.path.join(work_path, 'Datasets', 'GigaSpeech-Zhenhao', 'metadata')
assert os.path.isdir(meta_path), f'meta dir: {meta_path} does not exist!'

# get meta files (smaller number compared with opus and wav audio files)
meta_files = []
for i, wav_file in enumerate(wav_files):
    meta_file = wav_file.replace('/audio/', '/metadata/').replace('.wav', '.json')
    if os.path.isfile(meta_file):
        meta_files.append(meta_file)
    # else:
    #     print(f'{i}/{num_opus_file}: no corresponding {os.path.basename(meta_file)}, skip!')
num_meta_files = len(meta_files)
print(f'# of meta files: {num_meta_files}, while # of opus/wav files: {num_opus_files}')

durs_audio = [0 for _ in range(num_meta_files)]
durs_meta = [0 for _ in range(num_meta_files)]
for i, meta_file in enumerate(meta_files):

    wav_file = meta_file.replace('/metadata/', '/audio/').replace('.json', '.wav')

    # get duration from the audio file only when it is not empty
    wav_size = os.path.getsize(wav_file)
    if wav_size > 0:
        durs_audio[i] = librosa.get_duration(filename=wav_file)

    # get duration from the meta file
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    durs_meta[i] = meta['duration']

    # re-convert the audio if the duration abs difference is larger than 1 second
    abs_diff = abs(durs_meta[i]-durs_audio[i])
    if abs_diff > 1:
        rel_meta_path = meta_file.replace(meta_path, '').lstrip(os.sep)
        rel_id = os.path.splitext(rel_meta_path)[0]
        msg = f'{i}/{num_meta_files}: {rel_id}, dur (audio): {round(durs_audio[i],2)}, ' + \
            f'but dur (meta): {round(durs_meta[i],2)}'
        print(msg)
        rel_opus_path = rel_meta_path.replace('.json', '.opus')
        opus_file = os.path.join(opus_path, rel_opus_path)
        convert_audio((opus_file, wav_file), refresh=True)

    # re-check the duration
    durs_audio[i] = librosa.get_duration(filename=wav_file)
    abs_diff = abs(durs_meta[i]-durs_audio[i])
    msg = f'{i}/{num_meta_files}: {rel_id}, dur-mismatch after fix, dur (audio): {round(durs_audio[i],2)}, ' + \
            f'but dur (meta): {round(durs_meta[i],2)}'
    assert abs_diff <= 1, msg

# get some statistics
dur_audio_total_hrs = sum(durs_audio)/3600
dur_meta_total_hrs = sum(durs_meta)/3600
# check if the total duration diff (between audio and meta) is less than 3 min
assert abs(dur_audio_total_hrs-dur_meta_total_hrs) < 0.05, 'total duration between audio and meta is larger than 3 min!'
dur_total_hrs = min(dur_audio_total_hrs, dur_meta_total_hrs)
dur_mean_min = dur_total_hrs / num_meta_files * 60
# show mean and total duration (38131 recordings with avg dur 39.77 min and total dur 25274.66 hrs.)
print(f'{num_meta_files} recordings with avg dur {dur_mean_min:.2f} min and total dur {dur_total_hrs:.2f} hrs.')
