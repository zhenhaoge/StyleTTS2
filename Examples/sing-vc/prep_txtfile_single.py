# prepare the text file which contains a single entry of title|src-voice-wav(take content from)|tgt-voice-wav (take voice from)
#
# Zhenhao Ge, 2024-11-11

import os
import argparse
from pathlib import Path

# set paths
home_path = str(Path.home())

def parse_args():
    usage = 'prepare the text file which contains a single entry of 3 parts: title, source-wav-path, target-wav-path'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--src-file', type=str, help='source audio file (take content from)')
    parser.add_argument('--tgt-file', type=str, help='target audio file (take voice from)')
    parser.add_argument('--txt-file', type=str, help='output text file')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # data_dir = '/home/users/zge/code/repo/style-tts2/Datasets/YouTube'
    # account_id = 'pillow'
    # recording_id = 'adele.someone-like-you'
    # args.src_file = os.path.join(data_dir, account_id, f'{recording_id}_vocals.wav')
    # reference_id = 'voice_yufang_fssx2'
    # args.tgt_file = os.path.join(data_dir, account_id, f'{reference_id}_vocal.wav')
    # freevc_dir = os.path.join(home_path, 'code', 'repo', 'free-vc')
    # args.txt_file = os.path.join(freevc_dir, 'txtfiles', f'{recording_id}_{reference_id}.txt')

    # sanity check: file existence
    assert os.path.isfile(args.src_file), f'source audio file: {args.src_file} does not exist!'
    assert os.path.isfile(args.tgt_file), f'target audio file: {args.tgt_file} does not exist!'

    with open(args.txt_file, 'w') as f:
        f.write(f'dummy|{args.src_file}|{args.tgt_file}\n')
    print(f'wrote text file: {args.txt_file}')    
