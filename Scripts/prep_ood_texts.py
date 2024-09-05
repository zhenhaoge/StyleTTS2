# prepare the OOD text file with one additional column of text
#
# the original OOD test file contains 3 columns: audio file path, ipa phones, speaker ID,
# and this script will add one additional column: text, to make ipa phones more interligible

import os
from pathlib import Path
import argparse
from tqdm import tqdm
import multiprocessing
import psutil
import time 

# set dirs
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

from utils import split, flatten

def get_entries(txt_file):
    lines = open(txt_file, 'r').readlines()
    entries = [line.rstrip().split('|') for line in lines]
    return entries

def update_entry(entry, data_dir):
    rel_audio_file, phones, sid = entry
    parts = rel_audio_file.split(os.sep)
    dataset, speaker_id, utterance_id, wavname = parts[-4:]
    wavfile = os.path.join(data_dir, dataset, speaker_id, utterance_id, wavname)
    if dataset == 'train-clean-360' and (not os.path.isfile(wavfile)):
        dataset = 'train-clean-100'
        wavfile = os.path.join(data_dir, dataset, speaker_id, utterance_id, wavname)
    assert os.path.isfile(wavfile), f'wav file: {wavfile} does not exist!'
    txtfile = wavfile.replace('.wav', '.normalized.txt')
    text = open(txtfile, 'r').readlines()[0]
    rel_audio_file2 = f'{os.sep}'.join(parts[-4:])
    entry2 = (rel_audio_file2, text, phones, sid)
    return entry2

def worker(entries_split, data_dir, return_dict, pid):
    L = len(entries_split)
    entries2 = [() for _ in range(L)]
    for i, entry in enumerate(entries_split):
        entries2[i] = update_entry(entry, data_dir)
    return_dict[pid] = entries2    

def parse_args():
    usage = 'usage: prepare the OOD texts'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--in-txt-file', type=str, help='input text file with 3 columns')
    parser.add_argument('--out-txt-file', type=str, help='output text file with 4 columns')
    parser.add_argument('--data-dir', type=str, help='data dir to the libritts train-clean-360')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # interactive mode
    args = argparse.ArgumentParser()

    args.in_txt_file = os.path.join(work_dir, 'Data', 'OOD_texts.ori.txt')
    args.out_txt_file = os.path.join(work_dir, 'Data', 'OOD_texts.txt')
    args.data_dir = '/home/data/LibriTTS/wav24k'

    # localize arguments
    in_txt_file = args.in_txt_file
    out_txt_file = args.out_txt_file
    data_dir = args.data_dir

    # check file existence
    assert os.path.isfile(in_txt_file), f'input text file: {in_txt_file} does not exist!'

    # read contents from the input text file
    entries = get_entries(in_txt_file)
    num_entries = len(entries)
    print(f'# of entries in {in_txt_file}: {num_entries}') # 141434

    # update entries to entries2 (single thread)
    entries2 = [() for _ in range(num_entries)]
    # bs = 100
    for i, entry in enumerate(entries):
        # if i % bs == 0:
        #     print(f'processing entries {i} to {min(i+bs, num_entries)} ({num_entries} total) ...')
        entry2 = update_entry(entry, data_dir)
        entries2[i] = entry2

    # # update entries to entries2 (multithread)
    # start_time = time.time()
    # nprocs = psutil.cpu_count()
    # manager = multiprocessing.Manager()
    # return_dict = manager.dict()
    # jobs = []
    # entries_group = split(entries, nprocs)
    # for pid in range(nprocs):
    #     p = multiprocessing.Process(target=worker, args=(entries_group[pid], data_dir, return_dict, pid))
    #     jobs.append(p)
    #     p.start()
    # for proc in jobs:
    #     proc.join()
    # entries2 = [[] for _ in range(nprocs)]
    # for pid in range(nprocs):
    #     entries2[i] = return_dict[pid]
    # entries2 = flatten(entries2)    
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f'duration of processing entries with multithreads: {elapsed_time:.2f} seconds')

    lines = ['' for _ in range(num_entries)]
    for i, entry in enumerate(entries2):
        lines[i] = '|'.join(entry)
    open(out_txt_file, 'w').writelines('\n'.join(lines) + '\n')
    print(f'wrote output text file: {out_txt_file}')
