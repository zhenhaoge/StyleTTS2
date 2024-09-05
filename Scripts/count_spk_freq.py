# count the speaker frequncy in the file list
#
# Zhenhao Ge, 2024-08-21

import os
from pathlib import Path

# set paths
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

# set file list
list_file = os.path.join(work_dir, 'Data', 'GigaSpeech', 'train_list_nodur.txt')
# list_file = os.path.join(work_dir, 'Data', 'OOD_texts.txt')
assert os.path.join(list_file), f'train path: {list_file} does not exist!'

# get sids
lines = open(list_file, 'r').readlines()
sids = [int(l.rstrip().split('|')[-1]) for l in lines]
num_sids = len(sids)
print(f'# of sids: {num_sids}')

# get sid count dict
sid_count = {}
for sid in sids:
    if sid in sid_count:
        sid_count[sid] += 1
    else:
        sid_count[sid] = 1

# get # of uniq sids
sids_uniq = sorted(set(sids))
num_sids_uniq = len(sids_uniq)
print(f'# of uniq sids: {num_sids_uniq}')

# get # of single sids
sids_single = [sid for sid in sids_uniq if sid_count[sid]==1]
num_sids_single = len(sids_single)
print(f'# of sids with single audio file (count==1): {num_sids_single}')

# get max sid
sid_max = max(sids_uniq)

# update the lines with new sids
lines_new = ['' for _ in range(num_sids)]
for i, line in enumerate(lines):
    parts = line.rstrip().split('|')
    wavfile, text, phone_str, sid = parts
    if int(sid) in sids_single:
        sid = sid_max + 1
    else:
        sid = int(sid)
    lines_new[i] = '|'.join([wavfile, text, phone_str, str(sid)])

# write the updated file list
list_file2 = list_file.replace('.txt', '.new.txt')
open(list_file2, 'w').writelines('\n'.join(lines_new) + '\n')
