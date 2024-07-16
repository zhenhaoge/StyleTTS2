# subset the training list (e.g. for GigaSpeech) for trial training runs
#
# Zhenhao Ge, 2024-07-12

import os
from pathlib import Path
import argparse
import random

# set paths
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

def parse_args():
    usage = 'usage: subset the training list with a percentage'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--in-listfile', type=str, help='input list file')
    parser.add_argument('--out-listfile', type=str, help='output list file')
    parser.add_argument('--percent', type=float, help='percentage to subset')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    return parser.parse_args()

if __main__ == '__main__':

    # runtime mode
    args = parse_args()

    # interactive mode
    args = argparse.ArgumentParser()

    args.percent = 0.1
    args.seed = 1234
    str_pct = f'{int(args.percent*100)}p'
    args.in_listfile = os.path.join(work_dir, 'Data', 'GigaSpeech', 'train_list.txt')
    args.out_listfile = args.in_listfile.replace('.txt', f'_{str_pct}_{args.seed}.txt')

    # check dir/file existence
    assert os.path.isfile(args.in_listfile), f'input list file: {args.in_listfile} does not exist!'

    # localize arguments
    percent = args.percent
    seed = args.seed
    in_listfile = args.in_listfile
    out_listfile = args.out_listfile

    # print arguments
    print(f'percent: {percent}')
    print(f'seed: {seed}')
    print(f'in_listfile: {in_listfile}')
    print(f'out_listfile: {out_listfile}')

    # read the input list file
    with open(in_listfile, 'r') as f:
        lines = f.readlines()

    # get the number of entries
    nlines = len(lines)
    print(f'{nlines} in {in_listfile}')

    # get the manifest tuple list
    tuple_list = [() for _ in range(nlines)]
    for i in range(nlines):
        tuple_list[i] = tuple(lines[i].strip().split('|'))

    # save the original tuple list
    tuple_list_ori = tuple_list.copy()   

    # shuffle the tuple list with a fixed seed
    random.Random(seed).shuffle(tuple_list)
    # print(tuple_list[:3])

    # # sanity check: output of the same seed should be always the same
    # tuple_list = tuple_list_ori
    # random.Random(seed).shuffle(tuple_list)
    # print(tuple_list[:3])

    # subset with the specified percentage
    idx_pct = int(nlines*percent)
    tuple_list_sel = tuple_list[:idx_pct]
    nlines_subset = len(tuple_list_sel)

    # prepare lines to write
    lines_subset = ['' for _ in range(nlines_subset)]
    for i, entry in enumerate(tuple_list_sel):
        lines_subset[i] = '|'.join(entry)

    # write lines to the output the subset list file
    with open(out_listfile, 'w') as f:
        f.writelines('\n'.join(lines_subset) + '\n')
    print(f'wrote {out_listfile}')

        # write lines to the output the subset list file
    with open(out_listfile, 'w') as f:
        f.write('\n'.join(lines_subset) + '\n')
    print(f'wrote {out_listfile}')