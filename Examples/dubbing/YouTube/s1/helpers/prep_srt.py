# prepare the subtitle file in the srt format
# 
# Zhenhao Ge, 2024-06-29

import os, sys
from pathlib import Path
import argparse
# import time
# import datetime

# set paths
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

from Examples.dub_utils import sec2hms

def parse_args():
    usage = 'usage: prepare the subtitle file in the srt format'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--meta-file', type=str, help='meta file containing timestamps and texts')
    parser.add_argument('--srt-file', type=str, help='subtitle file in srt format')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()
    # account_id = 'laoming'
    # recording_id = '20220212'
    # dur_id = 'full'
    # meta_dir = os.path.join(work_dir, 'Outputs', 'YouTube', account_id, recording_id, dur_id, 'meta')
    # args.meta_file = os.path.join(meta_dir, f'{recording_id}_meta_v6.csv')
    # args.srt_file = os.path.join(meta_dir, f'{recording_id}_L2.srt')

    # check file/dir existence
    assert os.path.isfile(args.meta_file), f'meta file: {args.meta_file} does not exist!'

    # localize arguments
    meta_file = args.meta_file
    srt_file = args.srt_file

    # print arguments
    print(f'meta file: {meta_file}')
    print(f'srt file: {srt_file}')

    # read meta file
    lines = open(meta_file, 'r').readlines()
    header = lines[0].strip().split('|')
    lines = lines[1:]
    nsegments = len(lines)
    print(f'# of segments: {nsegments}')

    # get the index of start time, end time, and text
    colname_lst = ['start-time-l2', 'end-time-l2', 'text-l2']
    idx_dct = {k:header.index(k) for k in colname_lst}

    # find tuple list of (start time, end time, text)
    tuple_lst = [() for _ in range(nsegments)]
    for i in range(nsegments):
        parts = lines[i].strip().split('|')
        start_time_l2 = round(float(parts[idx_dct['start-time-l2']]), 2)
        end_time_l2 = round(float(parts[idx_dct['end-time-l2']]), 2)
        text_l2 = parts[idx_dct['text-l2']]
        tuple_lst[i] = (start_time_l2, end_time_l2, text_l2)

    # construct rows in the srt file
    rows = []
    for i in range(nsegments):
        rows.append(str(i))
        start_time_l2, end_time_l2, text_l2 = tuple_lst[i]
        start_hms = sec2hms(start_time_l2)
        end_hms = sec2hms(end_time_l2)
        rows.append(f'{start_hms} --> {end_hms}')
        rows.append(text_l2)
        rows.append('')

    # write out the srt file
    with open(srt_file, 'w') as f:
        f.writelines('\n'.join(rows) + '\n')
    print(f'wrote the srt file: {srt_file}')    
