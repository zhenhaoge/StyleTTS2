# check and fix duration of the audio file in the file list, and also 
# check if the duration matched the duration in the meta json file
#
# training reported error that audio file has length equal to 0 in Style-TTS2 training with GigaSpeech
# (ValueError: Input signal length=0 is too small to resample from 16000->24000)
#
# root course: multi-processing opus2wav conversion leads to some partial audio file, so the segments cannot
# be extracted correctly if the duration range is outside the duration of the partial audio file
#
# Zhenhao Ge, 2024-07-15

import os, sys
from pathlib import Path
import argparse
import librosa
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import psutil
import time
import numpy as np
import json
import soundfile as sf

# set paths
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

# set global variables
sample_rate = 16000

def convert_opus2wav(opus_file, wav_file, target_sr=16000, rm_opus=False):
    cmd = f'ffmpeg -y -i {opus_file} -ac 1 -ar {target_sr} {wav_file}'
    try:
      os.system(cmd)
    except:
      sys.exit(f'Failed to run the cmd: {cmd}')
    if rm_opus is True:
      os.remove(opus_file)

def read_manifest(manifest_file):

    lines = open(manifest_file, 'r').readlines()
    nlines = len(lines)
    tuple_list = [() for _ in range(nlines)]
    for i, line in enumerate(lines):
        tuple_list[i] = tuple(line.strip().split('|'))
    return tuple_list

def get_durations_from_audio(segment_files, bs=100, verbose=False):
    num_segment_files = len(segment_files)
    durations = [0 for _ in range(num_segment_files)]
    for i, segment_file in tqdm(enumerate(segment_files)):
        if verbose and i % bs == 0:
            print(f'processing {i} ~ {min(num_segment_files, i+bs)} files ...')
        durations[i] = librosa.get_duration(filename=segment_file)
    return durations

def get_durations_from_json(json_files, bs=100, verbose=False):
    num_json_files = len(json_files)
    durations = [0 for _ in range(num_json_files)]
    for i, json_file in tqdm(enumerate(json_files)):
        if verbose and i % bs == 0:
            print(f'processing {i} ~ {min(num_segment_files, i+bs)} files ...')
        with open(json_file, 'r') as f:
            meta = json.load(f)
        durations[i] = meta['duration']
    return durations       

def worker(segment_files, pid, return_dict):
    return_dict[pid] = get_durations_from_audio(segment_files)

def worker2(json_files, pid, return_dict):
    return_dict[pid] = get_durations_from_json(json_files)    

def check_dur(segment_files_group, return_dict, pid, idx):
    dur = librosa.get_duration(filename=segment_files_group[pid][idx])
    assert dur == return_dict[pid][idx], f'duration mis-match: pid={pid}, idx={idx}!'
    print(f'duration matched: pid={pid}, idx={idx}')

def check_dur2(segment_files, durations, idx):
    dur = librosa.get_duration(filename=segment_files[idx])
    assert dur == durations[idx], f'duration mis-match: idx={idx}!'
    print(f'duration matched: idx={idx}')

def group_duration(return_dict):
    pids = sorted(return_dict.keys())
    nprocs = len(pids)
    durations = []
    for i in range(nprocs):
        durations += return_dict[i]
    return durations    

def parse_args():
    usage = 'usage: check duration of the files listed in the manifest file'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--manifest-file', type=str, help='manifest file')
    parser.add_argument('--data-dir', type=str, help='root data dir')
    parser.add_argument('--ori-data-dir', type=str, help='original root data dir')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # interactive mode
    args = argparse.ArgumentParser()
    args.manifest_file = os.path.join(work_dir, 'Data', 'GigaSpeech', 'train_list_10p_1234.txt')
    args.data_dir = os.path.join(work_dir, 'Datasets', 'GigaSpeech-Zhenhao')
    args.ori_data_dir = os.path.join(work_dir, 'Datasets', 'GigaSpeech')

    # localize arguments
    manifest_file = args.manifest_file
    data_dir = args.data_dir
    ori_data_dir = args.ori_data_dir

    # check dir/file existence
    assert os.path.isfile(manifest_file), f'manifest file: {manifest_file} does not exist!'
    assert os.path.isdir(data_dir), f'data dir: {data_dir} does not exist!'
    assert os.path.isdir(ori_data_dir), f'original data dir: {ori_data_dir} does not exist!'

    tuple_list = read_manifest(manifest_file)
    segment_files = [os.path.join(data_dir, tuple_element[0]) for tuple_element in tuple_list]
    num_segment_files = len(segment_files)
    print(f'# of audio files: {num_segment_files}')

    json_files = [segment_file.replace('.wav', '.json') for segment_file in segment_files]

    # # get durations from audio (single-thread for loop)
    # durations = get_durations(segment_files)

    # get durations from audio (multi-threads)
    nprocs = psutil.cpu_count()
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    segment_files_group = np.array_split(segment_files, nprocs)
    # Ls = [len(segment_files_split) for segment_files_split in segment_files_group]
    for i in range(nprocs):
        p = multiprocessing.Process(target=worker, args=(segment_files_group[i], i, return_dict))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

    # # sanity check: duration
    # pid = 40 # [0,103]
    # idx = 80 # [0, 478]
    # check_dur(segment_files_group, return_dict, pid, idx)     

    # group durations (match the order in segment_files)
    durations = group_duration(return_dict)

    # # sanity check 2: duration
    # idx = 200
    # check_dur2(segment_files, durations, idx)

    # get durations from json files (multi-threads)
    manager = multiprocessing.Manager()
    return_dict2 = manager.dict()
    jobs = []
    json_files_group = np.array_split(json_files, nprocs)
    for i in range(nprocs):
        p = multiprocessing.Process(target=worker2, args=(json_files_group[i], i, return_dict2))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

    # group durations (match the order in json_files)
    durations2 = group_duration(return_dict2)

    # get idxs with zero durations 
    idxs_zero = [i for i, dur in enumerate(durations) if dur==0.0]
    nidxs_zero = len(idxs_zero)
    print(f'# of files with zero duration: {nidxs_zero} ({nidxs_zero/num_segment_files*100:.2f}%)')

    # get idxs with mis-matched duration
    idxs_mismatch = [i for i, (dur, dur2) in enumerate(zip(durations, durations2)) if round(dur,2)!=round(dur2,2)]
    nidxs_mismatch = len(idxs_mismatch)
    print(f'# of files with duration mis-match: {nidxs_mismatch} ({nidxs_mismatch/num_segment_files*100:.2f}%)')

    # for i, segment_file in enumerate(segment_files):
    #     segment_filename = os.path.basename(segment_file)
    #     sid = os.path.splitext(segment_filename)[0]
    #     if sid == 'POD0000014439_S0000223':
    #         print(f'{i}: {sid}')
    #         break

    # fix duration (duration mismatch )    
    for i, idx in enumerate(idxs_mismatch):

        dur, dur2 = durations[idx], durations2[idx]
        
        # get segment audio file
        segment_file = segment_files[idx]
        print(f'{i}/{nidxs_mismatch}: {segment_file}, duration should be {round(dur2,2)}, but {round(dur,2)}')

        # get segment json file
        json_file = segment_file.replace('.wav', '.json')
        with open(json_file, 'r') as f:
            meta = json.load(f)
        begin_time = meta['begin_time']
        end_time = meta['end_time']  
        duration = meta['duration']
        assert np.abs(duration - (end_time-begin_time)) < 0.01, f'{idx}: {json_file} duration mis-match!'

        # get recording audio file
        parts = segment_file.split(os.sep)
        sid = os.path.splitext(parts[-1])
        cat, pid, aid = parts[-4:-1]
        audio_file = os.path.join(data_dir, 'audio', cat, pid, f'{aid}.wav')
        assert os.path.isfile(audio_file), f'audio file: {audio_file} does not exist!'

        # get recording json file
        meta_file = audio_file.replace('/audio/', '/metadata/').replace('.wav', '.json')
        assert os.path.isfile(meta_file), f'meta file: {meta_file} does not exist!'

        # get total duration of the audio file
        dur_total = librosa.get_duration(filename=audio_file)

        # re-convert audio file (opus -> wav)
        if dur_total < end_time:
            opus_file = os.path.join(ori_data_dir, 'audio', cat, pid, f'{aid}.opus')
            assert os.path.isfile(opus_file), f'original audio file: {opus_file} does not exist!'
            convert_opus2wav(opus_file, audio_file, target_sr=16000, rm_opus=False)
            dur_total2 = librosa.get_duration(filename=audio_file)
            if dur_total2 >= end_time:
                print(f'{audio_file}: duration fixed, total dur: {round(dur_total2,2)}, segment dur: [{round(begin_time,2)}, {round(end_time,2)}]')
            else:
                raise Exception(f'{audio_file} duration not fixed, total dur: {round(dur_total2,2)}, segment dur: [{round(begin_time,2)}, {round(end_time,2)}]')    

        # read in the audio segment
        y, sr = librosa.load(audio_file, sr=sample_rate, offset=begin_time, duration=duration)
        dur = len(y)/sr
        assert round(dur,2) == round(dur2,2), f'duration still not fixed, should be {round(dur2,2)}, but {round(dur,2)}'

        # re-write the segment audio file
        segment_dir = os.path.dirname(segment_file)
        os.makedirs(segment_dir, exist_ok=True)
        sf.write(segment_file, y, sr)
        print(f're-wrote segment audio file: {segment_file}')

    # get durations from audio again after fix (multi-threads)
    nprocs = psutil.cpu_count()
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    segment_files_group = np.array_split(segment_files, nprocs)
    # Ls = [len(segment_files_split) for segment_files_split in segment_files_group]
    for i in range(nprocs):
        p = multiprocessing.Process(target=worker, args=(segment_files_group[i], i, return_dict))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

    # # sanity check: duration
    # pid = 40 # [0,103]
    # idx = 80 # [0, 478]
    # check_dur(segment_files_group, return_dict, pid, idx)     

    # group durations (match the order in segment_files)
    durations = group_duration(return_dict)

    # get idxs with mis-matched duration
    idxs_mismatch = [i for i, (dur, dur2) in enumerate(zip(durations, durations2)) if round(dur,2)!=round(dur2,2)]
    nidxs_mismatch = len(idxs_mismatch)
    print(f'# of files with duration mis-match: {nidxs_mismatch} ({nidxs_mismatch/num_segment_files*100:.2f}%)') # should be 0 now