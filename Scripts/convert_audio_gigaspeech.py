# convert audio files in GigaSpeech from opus to wav format
#
# reference: https://github.com/SpeechColab/GigaSpeech/blob/main/utils/opus_to_wav.py
#
# Zhenhao Ge, 2024-06-17

import os
from pathlib import Path
import glob
import opuspy
import librosa
import soundfile as sf
from tqdm import tqdm
from multiprocessing import Pool

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

from audio import convert_opus2wav

def convert_audio(file_pair):
    opus_file, wav_file = file_pair
    wav_dir = os.path.dirname(wav_file)
    os.makedirs(wav_dir, exist_ok=True)
    if not os.path.isfile(wav_file):
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
print(f'# of opus files: {num_opus_files}') # 40368 opus files

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
