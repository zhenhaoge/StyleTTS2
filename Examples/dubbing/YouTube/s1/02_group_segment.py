# group adjacent audio segments based on the timestamps, which can ensure that
# the same sentence is not split into multiple segments
#
# Zhenhao Ge, 2024-06-25

import os
from pathlib import Path
import argparse
from pydub import AudioSegment
import glob
import json
import librosa
import soundfile as sf

# set dirs
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

from utils import set_path, empty_dir, get_value_from_json, tuple2csv
from Examples.dub_utils import get_ts_from_filename, get_ts, group_ts, extract_grouped_segment

def parse_args():
    usage = 'usage: group segments that are close to each other'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--audio-file', type=str, help='audio file to extract segments from')
    parser.add_argument('--in-dir', type=str, help='input dir of the ungrouped segments')
    parser.add_argument('--out-dir', type=str, help='output dir of the grouped segments')
    parser.add_argument('--meta-dir', type=str, help='dir of the meta data')
    parser.add_argument('--in-ts-file', type=str, default='', help='input timestamp file with segment index and text')
    parser.add_argument('--out-ts-file', type=str, default='', help='output timestamp file with segment index and text')
    parser.add_argument('--out-ver', type=str, help='output version, e.g. v{x}')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # data_dir = os.path.join(home_dir, 'data1', 'datasets', 'YouTube')
    # account_id = 'dr-wang'
    # recording_id = '20210915'
    # dur_id = 'full'
    # args = argparse.ArgumentParser()
    # args.audio_file = os.path.join(data_dir, account_id, recording_id, dur_id, f'{recording_id}_L1_vocals_mono.wav')
    # args.meta_dir = os.path.join(work_dir, 'Outputs', 'YouTube', account_id, recording_id, dur_id, 'meta')
    
    # # pass 1 (using timestamp from filenames in v1.original)
    # args.in_dir = os.path.join(work_dir, 'Outputs', 'YouTube', account_id, recording_id, dur_id, 'v1.original')
    # args.out_dir = args.in_dir.replace('v1.original', 'v2.grouped')
    # args.in_ts_file = ''
    # args.out_ts_file = os.path.join(args.meta_dir, f'{recording_id}_L1_ts-text_v2.csv')
    # args.out_ver = 'v2'

    # # pass 2 (using ts-text file with manual correction in v2.grouped)
    # args.in_dir = os.path.join(work_dir, 'Outputs', 'YouTube', account_id, recording_id, dur_id, 'v2.grouped')
    # args.out_dir = args.in_dir.replace('v2.grouped', 'v3.corrected')
    # args.in_ts_file = os.path.join(args.meta_dir, f'{recording_id}_L1_ts-text_v2.corrected.csv')
    # args.out_ts_file = os.path.join(args.meta_dir, f'{recording_id}_L1_ts-text_v3.csv')
    # args.out_ver = 'v3'

    # localize arguments
    audio_file = args.audio_file
    in_dir = args.in_dir
    out_dir = args.out_dir
    meta_dir = args.meta_dir
    in_ts_file = args.in_ts_file
    out_ts_file = args.out_ts_file
    out_ver = args.out_ver

    # check file/dir existence
    assert os.path.isfile(audio_file), f'audio file: {audio_file} does not exist!'
    assert os.path.isdir(in_dir), f'input dir: {in_dir} does not exist!'

    # set output dir
    set_path(out_dir, verbose=True)
    empty_dir(out_dir)

    # set meta dir
    set_path(meta_dir, verbose=True)

    # get recording id
    recording_id = os.path.basename(audio_file).split('_')[0]

    # print out arguments
    print(f'recording id: {recording_id}')
    print(f'audio file: {audio_file}')
    print(f'input dir: {in_dir}')
    print(f'output dir: {out_dir}')
    print(f'meta dir: {meta_dir}')
    print(f'input timestamp csv file: {in_ts_file}')
    print(f'output timestamp csv file: {out_ts_file}')
    print(f'output version: {out_ver}')

    # convert audio file to mono-channel (if needed)
    wav = AudioSegment.from_file(audio_file)
    if wav.channels > 1:
        audio_file2 = audio_file.replace('.wav', '_mono.wav')
        if os.path.isfile(audio_file2):
            print(f'use existed mono-channel audio file instead: {audio_file2}')
        else:
            extract_wav_channel(audio_file, audio_file2, channel=0, verbose=True)
            print(f'use newly generated mono-channel audio file: {audio_file2}')    
        audio_file = audio_file2
    else:
        print(f'{audio_file} is already a mono-channel aduio file')

    # read the input wav and json files
    in_wav_files = sorted(glob.glob(os.path.join(in_dir, '*.wav')))
    in_json_files = sorted(glob.glob(os.path.join(in_dir, '*.json')))
    num_wavs = len(in_wav_files)
    num_jsons = len(in_json_files)
    assert num_wavs == num_jsons, "input wav and json files mis-match!"
    num_segments = num_wavs
    del num_wavs, num_jsons

    # get the initial segment time stamps
    if not in_ts_file:
        print('getting timestamps from the input wav files ...')
        tuple_list = get_ts_from_filename(in_wav_files)
        num_segments = len(tuple_list)
        print(f'# of initial segments: {num_segments}')

        # get the grouped segment time stamps with old fid range
        tuple_list_grouped = group_ts(tuple_list)

    else:
        print(f'getting timestamps from the input timestamp csv file {in_ts_file} ...')
        assert os.path.isfile(in_ts_file), f'input timestamp file {in_ts_file} does not exist!'
        tuple_list_grouped = get_ts(in_ts_file)

    num_segments_grouped = len(tuple_list_grouped)
    print(f'# of grouped segments: {num_segments_grouped}')

    # extract segments
    extract_grouped_segment(audio_file, tuple_list_grouped, in_json_files, out_dir, refresh=False, verbose=1)

    # get the output wav files
    out_wav_files = sorted(glob.glob(os.path.join(out_dir, '*.wav')))
    assert len(out_wav_files) == num_segments_grouped, \
        '# of output wav files != # of grouped segments!'

    # get the output json files
    out_json_files = sorted(glob.glob(os.path.join(out_dir, '*.json')))
    assert len(out_json_files) == num_segments_grouped, \
        '# of output json files != # of grouped segments!'

    # get the header for the output meta csv
    with open(out_json_files[0]) as f:
        meta = json.load(f)
    header = list(meta.keys())
    header = ['wav-file'] + header

    # construct rows
    rows = [() for _ in range(num_segments_grouped)]
    for i in range(num_segments_grouped):

         # load meta for the current segment
        with open(out_json_files[i]) as f:
            meta = json.load(f)

        # get current row
        entry = [out_wav_files[i]]
        for k in header[1:]:
            entry += [meta[k]]
        rows[i] = tuple(entry)    

        # rows[i] = (out_wav_files[i],
        #            meta['fid'],
        #            meta['start-time'],
        #            meta['end-time'],
        #            meta['duration'],
        #            meta['start-time-ori'],
        #            meta['end-time-ori'],
        #            meta['duration-ori'],
        #            meta['idx'],
        #            meta['idx-ori-start'],
        #            meta['idx-ori-end'],
        #            meta['text'])

    # write meta to csv
    out_csv_file = os.path.join(meta_dir, f'{recording_id}_meta_{out_ver}.csv')
    tuple2csv(rows, out_csv_file, delimiter='|', header=header, verbose=True)

    # write timestamps with idx and text for mannual correction
    rows = [() for _ in range(num_segments_grouped)]
    for i in range(num_segments_grouped):

        with open(out_json_files[i]) as f:
            meta = json.load(f)

        rows[i] = (meta['idx'],
                   meta['start-time'],
                   meta['end-time'],
                   meta['text'])

    # write timestamps with idx and text to a .csv file
    header = ['idx', 'start-time', 'end-time', 'text']
    tuple2csv(rows, out_ts_file, delimiter='|', header=header, verbose=True)

    # write text to a .txt file
    out_txt_file = os.path.join(meta_dir, f'{recording_id}_L1_text_{out_ver}.txt')
    texts = [row[-1] for row in rows]
    with open(out_txt_file, 'w') as f:
        f.writelines('\n'.join(texts))
    print(f'wrote the texts (no time stamp) to {out_txt_file}')

    # # (obsoleted) execute when the corrected texts are manually created in {recording_id}_L1_text.corrected.txt, after v2 run
    # out_ts_file_corrected = os.path.join(meta_dir, f'{recording_id}_L1_ts-text_{out_ver}.corrected.csv')
    # if os.path.isfile(out_ts_file_corrected):
    #     # read the corrected texts
    #     lines = open(out_ts_file_corrected, 'r').readlines()
    #     lines = lines[1:]
    #     nlines = len(lines)
    #     texts = ['' for _ in range(nlines)]
    #     for i in range(nlines):
    #         texts[i] = lines[i].strip().split('|')[-1]
    #     print(f'read the corrected texts with time stamps from {out_ts_file_corrected}')    

    #     # write corrected texts to a .txt file
    #     out_txt_file = os.path.join(meta_dir, f'{recording_id}_L1_text_{out_ver}.corrected.txt')
    #     with open(out_txt_file, 'w') as f:
    #         f.writelines('\n'.join(texts))
    #     print(f'wrote the corrected texts (no time stamp) to {out_txt_file}')
    # else:
    #     print(f'{out_ts_file_corrected} does not exist yet, ' + \
    #         f'consider manually correct the texts in {out_csv_file} and save the corrected texts in {out_ts_file_corrected}') 
