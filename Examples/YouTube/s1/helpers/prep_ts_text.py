# prepare the csv file with timestamps and translated texts
#
# reason for developing this script:
#   - given chinese texts with time stamps and english texts without time stamps, prepare
#     the english texts with time stamps
#   - it is because we only feed in pure chinese texts (without time stamps) to DeepL for
#     translation, which outputs purely the english texts, so we need to attach the time
#     stamps to the texts 
#
# the prepared timestamps with english text is to be used in 03_gen_segment.py, so do this
# before running that script
#
# Zhenhao Ge, 2024-06-27

import os
from pathlib import Path
import argparse

# set dirs
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

from utils import tuple2csv

def get_ts(ts_file):

    lines = open(ts_file, 'r').readlines()
    header = lines[0].strip().split('|')
    assert header == ['idx', 'start-time', 'end-time', 'text'], 'header mis-match!'
    lines = lines[1:]
    nsegments = len(lines)
    tuple_list = [() for _ in range(nsegments)]
    for i in range(nsegments):
        parts = lines[i].strip().split('|')
        idx = int(parts[0])
        start_time = round(float(parts[1]), 2)
        end_time = round(float(parts[2]), 2)
        text = parts[3]
        tuple_list[i] = (idx, start_time, end_time, text)
    return tuple_list

def parse_args():
    usage = 'usage: prepare csv file with time stamps with translated texts'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--zh-ts-file', type=str, help='meta file with timestamps and texts for L1')
    parser.add_argument('--en-text-file', type=str, help='translated texts file for L2')
    parser.add_argument('--en-ts-file', type=str, help='meta file with timestamp and translated texts for L2')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # account_id = 'laoming'
    # recording_id = '20220212'
    # dur_id = 'full'
    # args =  argparse.ArgumentParser()
    # meta_dir = os.path.join(work_dir, 'Outputs', 'YouTube', account_id, recording_id, dur_id, 'meta')
    # args.zh_ts_file = os.path.join(meta_dir, f'{recording_id}_L1_ts-text_v3.csv')
    # args.en_text_file = os.path.join(meta_dir, f'{recording_id}_L2_text_v3.manual.txt')
    # args.en_ts_file = os.path.join(meta_dir, f'{recording_id}_L2_ts-text_v3.manual.csv')

    # check input file/dir existence
    assert os.path.isfile(args.zh_ts_file), f'chinese timestamp file {args.zh_ts_file} does not exist!'
    assert os.path.isfile(args.en_text_file), f'english text file {args.en_text_file} does not exist!' 

    # localize arguments
    zh_ts_file = args.zh_ts_file
    en_text_file = args.en_text_file
    en_ts_file = args.en_ts_file

    # print out arguments
    print(f'chinese timestamp file: {zh_ts_file}')
    print(f'english text file: {en_text_file}')
    print(f'english timestamp file: {en_ts_file}')

    # get chinese tuple list
    zh_tuple_list = get_ts(zh_ts_file)
    nsegments = len(zh_tuple_list)
    print(f'# of segments in chinese timestamp file: {nsegments}')

    # get english texts
    lines = open(en_text_file, 'r').readlines()
    nlines = len(lines)
    en_texts = [line.strip() for line in lines]

    # get english tuple list
    en_tuple_list = [() for _ in range(nsegments)]
    for i, (zh_tuple, en_text) in enumerate(zip(zh_tuple_list, en_texts)):
        idx, start_time, end_time, zh_text = zh_tuple
        en_tuple_list[i] = (idx, start_time, end_time, en_text)

    # write english tuple list to csv
    header = ['idx', 'start-time', 'end-time', 'text']
    tuple2csv(en_tuple_list, en_ts_file, delimiter='|', header=header, verbose=True)
