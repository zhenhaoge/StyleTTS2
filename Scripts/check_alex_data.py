import os
from pathlib import Path
import json

# set dirs
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

data_dir_alex = os.path.join(work_dir, 'Datasets', 'GigaSpeech-Alex')
data_dir_zhenhao = os.path.join(work_dir, 'Datasets', 'GigaSpeech-Zhenhao')
assert os.path.isdir(data_dir_alex), f'data dir: {data_dir_alex} does not exist!'
assert os.path.isdir(data_dir_zhenhao), f'data dir: {data_dir_zhenhao} does not exist!'

meta_file1 = os.path.join(data_dir_alex, 'dev.json')
with open(meta_file1) as f:
    meta1 = json.load(f)
fids1 = sorted([os.path.basename(k) for k in meta1.keys()])

meta_file2 = os.path.join(data_dir_zhenhao, 'fid2meta_val.json')
with open(meta_file2) as f:
    meta2 = json.load(f)
fids2 = sorted([os.path.basename(k) for k in meta2.keys()])

assert fids1 == fids2, 'two files contains different fids!'