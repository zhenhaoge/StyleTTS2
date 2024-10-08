# segment audio into segments for GigaSpeech
#  - generate audio-wise metadata under folder GigaSpeech/metadata
#  - check tags in the texts (found out there are tags other than punctuations, such as NOISE, MUSIC, SIL, OTHER)
#  - get some overall meta data (duration, #segments, etc.)
#  - get sid2path dict to enable finding wav path from sid (sid is unique)
#  - extract audio segments
#
# Zhenhao Ge, 2024-06-18

import os
from pathlib import Path
import json
import glob
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

# set paths
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current path: {}'.format(os.getcwd()))

from utils import set_path

punc_map = {'<COMMA>':',', '<PERIOD>':'.', '<QUESTIONMARK>':'?', '<EXCLAMATIONPOINT>':'!'}
punc_list = list(punc_map.values())

def process_text(text, punc_map):
    for k,v in punc_map.items():
        text = text.replace(k, v)
    text = text.lower()
    return text

def extract_meta(args):

    audio_dict, meta_dir, verbose = args

    opus_rel_path = audio_dict['path']
    # meta_rel_path = opus_rel_path.replace('audio', '').replace('.opus', '.json').strip('/')
    meta_rel_path = opus_rel_path.replace('audio/', '').replace('.opus', '.json')
    meta_file = os.path.join(meta_dir, meta_rel_path)
    meta_dir = os.path.dirname(meta_file)
    set_path(meta_dir)

    if not os.path.isfile(meta_file):
        with open(meta_file, 'w') as f:
            json.dump(audio_dict, f, indent=2)
        if verbose:    
            print(f'wrote meta file: {meta_file}')
    else:
        if verbose:
            print(f'meta file: {meta_file} already exist')

def get_texts(meta_file):
    """get the normalized text from meta file"""
    with open(meta_file) as f:
        audio_dict = json.load(f)
    texts = [segment['text_tn'] for segment in audio_dict['segments']]
    # print('\n'.join(texts))
    return texts

def get_tag(texts):
    """get tag:count dict"""

    tag_list = []
    for i, text in enumerate(texts):
        idx_start, idx_end = -1, -1
        for j, char in enumerate(text):
            if char == '<':
                idx_start = j
            elif char == '>':
                idx_end = j
                if idx_start != -1 and idx_start < idx_end:
                    tag = text[idx_start+1:idx_end]
                    tag_list.append(tag)
                    idx_start, idx_end = -1, -1
                else:
                    raise Exception(f'check text {i} with idx_start:{idx_start} and idx_end:{idx_end}!')

    tag_dict = {}
    for tag in tag_list:
        if tag in tag_dict.keys():
            tag_dict[tag] += 1
        else:
            tag_dict[tag] = 1

    return tag_dict

def get_acc_tag(tag_acc_dict, tag_dict):
    """append the current tag dict to the accumulative tag dict"""
    for k in tag_dict.keys():
        if k in tag_acc_dict.keys():
            tag_acc_dict[k] += tag_dict[k]
        else:
            tag_acc_dict[k] = tag_dict[k]
    return tag_acc_dict            

def segment_audio(args):
    """segment audio recording into audio segments given audio meta data in dict and input/output dirs"""

    # parse arguments
    # data_dir: '/home/users/zge/code/repo/style-tts2/Datasets/GigaSpeech-Zhenhao'
    # output_dir: '/home/users/zge/code/repo/style-tts2/Datasets/GigaSpeech-Zhenhao/segment'
    audio_dict, data_dir, output_dir = args

    # get the relative path for opus audio file
    opus_rel_path = audio_dict['path']
    # get category (podcast, youtube, audiobook, etc.) and pid (e.g., P0001)
    cat, pid = opus_rel_path.split('/')[1:3]
    # get audio id (e.g., 'POD0000000001')
    aid =  audio_dict['aid']
    sample_rate = audio_dict['sample_rate']
    segments = audio_dict['segments']
    num_segments = len(segments)

    # # show progress
    # print(f'extracting segments for audio {aid} ...')

    # get the wav path based on opus path (generated in convert_audio_gigaspeech.py)
    wav_rel_path = opus_rel_path.replace('.opus', '.wav')
    wav_path = os.path.join(data_dir, wav_rel_path)
    assert os.path.isfile(wav_path), f'wav path {wav_path} does not exist!'

    # create segment dir
    seg_dir = os.path.join(output_dir, cat, pid, aid)
    os.makedirs(seg_dir, exist_ok=True)

    for j in range(num_segments):
        segment = segments[j]
        sid = segment['sid']

        # get the duration (used to valid the wav file is not empty)
        begin_time = segment['begin_time']
        end_time = segment['end_time']
        duration = round(end_time-begin_time, 2)

        # set output files for the current segment
        seg_path = os.path.join(seg_dir, f'{sid}.wav')
        txt_path = os.path.join(seg_dir, f'{sid}.txt')
        json_path = os.path.join(seg_dir, f'{sid}.json')

        # check if the segment wav path
        cond11 = os.path.isfile(seg_path)
        if cond11:
            dur = librosa.get_duration(filename=seg_path)
            cond12 = dur > 0.0 and dur <= duration + 0.01
        else:
            cond12 = False
        cond1 = cond11 and cond12

        # check if the text path is valid
        cond21 = os.path.isfile(txt_path)
        if cond21:
            lines = open(txt_path, 'r').readlines()
            nlines = len(lines)
            cond22 = nlines == 1
        else:
            cond22 = False
        cond2 = cond21 and cond22

        # check if the json path is valid
        cond31 = os.path.isfile(json_path)
        if cond31:
            nlines_json = len(open(json_path, 'r').readlines())
            cond32 = nlines_json > 0
        else:
            cond32 = False
        cond3 = cond31 and cond32

        if cond1 and cond2 and cond3:
            # print(f'{sid}: already generated, skip ...')
            continue
        else:
            print(f'{sid}: extracting ...')
            text_tn = segment['text_tn']
            text_tn = process_text(text_tn, punc_map)
            y, sr = librosa.load(wav_path, sr=sample_rate, offset=begin_time, duration=duration)

            # save segment wav file
            sf.write(seg_path, y, sr)

            # save segment text file
            with open(txt_path, 'w') as f:
                f.write(f'{text_tn}\n')

            # save segment json file
            meta = {'title': audio_dict['title'], 'pid': pid, 'aid': aid, 'sid': sid, 'speaker': segment['speaker'],
                    'begin_time': begin_time, 'end_time': end_time, 'duration': duration, 'subsets': segment['subsets']}
            with open(json_path, 'w') as f:
                json.dump(meta, f, indent=2)

# set paths
data_ori_dir = os.path.join(work_dir, 'Datasets', 'GigaSpeech')
data_dir = os.path.join(work_dir, 'Datasets', 'GigaSpeech-Zhenhao')
meta_dir = os.path.join(data_dir, 'metadata')
output_dir = os.path.join(data_dir, 'segment')
set_path(meta_dir, verbose=True)
set_path(output_dir, verbose=True)

verbose = True

# get master metadata jsonfile
master_jsonfile = os.path.join(data_ori_dir, 'GigaSpeech.json')
assert os.path.isfile(master_jsonfile), \
    'master jsonfile: {} does not exist!'.format(master_jsonfile)

# loda metadata from the master jsonfile
with open(master_jsonfile) as f:
    data = json.load(f)

# get audio list
audio_list = data['audios']
num_audios = len(audio_list)
print('# of audios: {}'.format(num_audios)) # 38131 audios

#%% audio-wise meta data extraction

# generate audio-wise meta data in json format
# (after this is done, the # of meta json file should be equal to the # of audio recordings)
meta_files = sorted(glob.glob(os.path.join(meta_dir, '**', '*.json'), recursive=True))
num_meta_files = len(meta_files)
print(f'# of meta files: {num_meta_files}')

if num_meta_files < num_audios:

    # get audio-wise meta data using for-loop
    for i in range(num_audios):

        # get current audio info
        audio_dict = audio_list[i]

        # # show meta data (except segment info) for the current audio
        # for k,v in audio_dict.items():
        #     if k != 'segments':
        #         print('{}: {}'.format(k, v))

        # show progress
        aid = audio_dict['aid']
        print(f'{i+1}/{num_audios}: extracting meta for audio {aid} ...')

        args = (audio_dict, meta_dir, verbose)
        extract_meta(args)

    # #  get audio-wise meta data using using multi-processing
    # meta_dirs = [meta_dir for _ in range(num_audios)]
    # verboses = [verbose for _ in range(num_audios)]
    # pool = Pool()
    # pool.map(extract_meta, zip(audio_list, meta_dirs, verboses))

else:

    print('all audio-wise meta data are extracted already, skip meta data extraction.')

# sanity check: meta files are not empty
for i in tqdm(range(num_meta_files)):
    lines = open(meta_files[i], 'r').readlines()
    if len(lines) == 0:
        raise Exception(f'{meta_files[i]} is empty!')

#%%  check texts and tags inside

tag_acc_dict = {}
itvl = 1000
for i in range(num_audios):

    # show progress
    if i % itvl == 0:
        print(f'checking texts from {i} to {min(num_audios, i+itvl)} (total {num_audios}) ...')

    # get current audio info
    audio_dict = audio_list[i]

    opus_rel_path = audio_dict['path']
    meta_rel_path = opus_rel_path.replace('audio/', '').replace('.opus', '.json')
    meta_file = os.path.join(meta_dir, meta_rel_path)
    texts = get_texts(meta_file)
    tag_dict = get_tag(texts)
    tag_acc_dict = get_acc_tag(tag_acc_dict, tag_dict)

# print out tags and their counts
print('tags and their counts:')
for k, v in tag_acc_dict.items():
    print(f'{k}: {v}')

#%% get some overall meta data (total duration, # of segments, etc.)

# get duration stats (sum, mean, etc.)
durations = [[] for _ in range(num_audios)]
num_segments = [0 for _ in range(num_audios)]
for i in range(num_audios):

    audio_dict = audio_list[i]
    segments = audio_dict['segments']
    num_segments[i] = len(segments)
    # print(f'({i}/{num_audios}) # of segments: {num_segments[i]}')

    durations[i] = [0 for _ in range(num_segments[i])]
    for j in range(num_segments[i]):
        durations[i][j] = segments[j]['end_time'] - segments[j]['begin_time']

num_segments_total = int(sum(num_segments))
num_segments_mean = num_segments_total/num_audios
durations_audio = [sum(durs) for durs in durations]
duration_total = sum(durations_audio)
duration_audio_mean = np.mean(durations_audio)
duration_segment_mean = duration_total / num_segments_total
print(f'total #segments in GigaSpeech: {num_segments_total}') # 8315357 segments
print('mean #segments in GigaSpeech per recording: {:.2f}'.format(num_segments_mean)) # 218.07 #segments
print('total duration of GigaSpeech: {:.2f} hrs.'.format(duration_total/3600)) # 10050.65 hrs.
print('mean duration of GigaSpeech per audio recording: {:.2f} min.'.format(duration_audio_mean/60)) # 15.81 min.
print('mean duration of GigaSpeech per audio segment: {:.2f} sec.'.format(duration_segment_mean)) # 4.35 sec.

#%% get sid2path dict (sid: wav relative path)

sid2path = {}
for i in range(num_audios):

    audio_dict = audio_list[i]

    opus_rel_path = audio_dict['path']
    cat, pid = opus_rel_path.split('/')[1:3]
    aid = audio_dict['aid']
    segments = audio_dict['segments']
    num_segments = len(segments)

    # # show progress
    # print(f'extracting segments for audio {aid} ...')

    # set segment dir
    seg_dir = os.path.join(output_dir, cat, pid, aid)
   
    for j in range(num_segments):
        segment = segments[j]
        sid = segment['sid']
        seg_path = os.path.join(seg_dir, f'{sid}.wav')
        seg_rel_path = seg_path.replace('{}/'.format(data_dir), '')
        sid2path[sid] = seg_rel_path

num_sids = len(sid2path)
print(f'# of segment ids: {num_sids}')
assert num_sids == num_segments_total, \
    f'# sids ({num_sids}) must be equal to the total #segments ({num_segments_total})!'

sid2path_file = os.path.join(data_dir, 'sid2path.txt')
itvl = 100000
with open(sid2path_file, 'w') as f:
    for i, (k,v) in enumerate(sid2path.items()):
        if i % itvl == 0:
            print(f'writing line {i} ~ {min(num_sids, i+itvl)} (total {num_sids}) ...')
        f.write('{}|{}\n'.format(k,v))
    f.write('\n')

#%% extract audio segments

# dummy replications for the parallel runs
data_dirs = [data_dir for _ in range(num_audios)]
output_dirs = [output_dir for _ in range(num_audios)]

# # extract segments using for-loop (~40 hours)
# for i, args in enumerate(tqdm(zip(audio_list, data_dirs, output_dirs))):

#     audio_dict, data_dir, output_dir = args

#     # show progress
#     aid = audio_dict['aid']
#     print(f'{i}/{num_audios}: extracting segments for audio {aid} ...')

#     segment_audio(args)

# extract segments using multi-processing
pool = Pool()
pool.map(segment_audio, zip(audio_list, data_dirs, output_dirs))
