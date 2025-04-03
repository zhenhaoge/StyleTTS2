# align speech to get the word-level timestamps
#
# Experiment the quality and speed using word-level accumulation method to generate speech in a streaming setting
# 2nd step: align TTS speech samples to get the word-level timestamps 
#
# Zhenhao Ge, 2024-10-22

import os
from pathlib import Path
import argparse
import glob
import subprocess

home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))
align_path = os.path.join(home_path, 'code', 'repo', 'gentle')

from utils import convertible_to_integer

def filter_file(filelist):

    filelist2 = [f for f in filelist if '_t' not in f]
    filelist2 = [f for f in filelist2 if '_concat_' not in f]
    filelist2 = [f for f in filelist2 if '_reference' not in f]

    return filelist2

def parse_args():

    usage = 'usage: align TTS speech to get the word-level timestamps'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--output-path', type=str, help='root output path')
    parser.add_argument('--exp-id', type=str, help='exp id')
    parser.add_argument('--ref-id', type=str, help='ref id')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args =  argparse.ArgumentParser()

    # work_path = os.getcwd() # e.g., '/home/users/zge/code/repo/style-tts2'
    # args.output_path = os.path.join(work_path, 'Outputs', 'Scratch', 'LibriTTS')
    # args.exp_id = 2
    # args.ref_id = 'YOU1000000038_S0000079'

    # print arguments
    print(f'output path: {args.output_path}')
    print(f'exp id: {args.exp_id}')
    print(f'ref id: {args.ref_id}')

    # set the output path
    if convertible_to_integer(args.exp_id):
        args.exp_id = int(args.exp_id)
        output_folder = f'exp-{args.exp_id:02d}'
    else:
        output_folder = f'gs-{args.exp_id}'
    output_path = os.path.join(args.output_path, output_folder)
    if not os.path.isdir(output_path):
        print(f'creating new output dir: {output_path}')
        os.makedirs(output_path)
    else:
        print(f'using existing output dir: {output_path}')

    # get the sorted wav files
    out_wavfiles = sorted(glob.glob(os.path.join(output_path, f'{args.ref_id}*.wav')))
    out_wavfiles = filter_file(out_wavfiles)
    idxs = [int(os.path.basename(f).split('-')[1]) for f in out_wavfiles]
    out_wavfiles = [f for _, f in sorted(zip(idxs, out_wavfiles))]

    # get the text and TextGrid files
    out_txtfiles = [f.replace('.wav', '.txt') for f in out_wavfiles]
    out_tgfiles = [f.replace('.wav', '.TextGrid') for f in out_wavfiles]

    # get the number of texts
    ntexts = len(out_txtfiles)
    print(f'# of texts: {ntexts}')
        
    # write the TextGrid alignment file
    for i in range(ntexts):
        
        # run the force alignment command in subprocess
        command = ['python', os.path.join(align_path, 'align.py'), '--output', out_tgfiles[i], out_wavfiles[i], out_txtfiles[i]]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check if the command was successful
        if result.returncode == 0:
            print("Command executed successfully")
            print(result.stdout)
        else:
            print("Error occurred")
            print(result.stderr)
