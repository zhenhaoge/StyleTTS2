# Prepare the manifest files with different text lengths for RTF tests
#
# Zhenhao Ge, 2024-06-12

import os
from pathlib import Path
import glob
import numpy as np
import random

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

from audio import wav_duration

def select_wav(wavpaths, dur=3.0):

    durs = [0 for _ in range(len(wavpaths))]
    for i, f in enumerate(wavpaths):
        try:
            durs[i] = wav_duration(f)
        except:
            continue
    wavpaths = [f for d,f in zip(durs, wavpaths) if d==dur] 

    return wavpaths

def count_word_from_texts(texts):
    nwords = [len(text.split()) for text in texts]
    return nwords

def write_manifest(filepath, wavpaths, texts, seed=0):

    assert len(wavpaths) == len(texts), 'wav paths and texts mis-match!'

    rows = ['{} | {}'.format(f, text) for (f,text) in zip(wavpaths, texts)]
    random.seed(seed)
    random.shuffle(rows)
    # print('random rows:\n{}'.format('\n'.join(rows)))
    
    with open(filepath, 'w') as f:
        f.writelines('\n'.join(rows))
    print('wrote file: {}'.format(filepath))    

# get reference wav paths with duration 3.0
reference_path = os.path.join(work_path, 'Demo', 'reference_audio')
reference_wavpaths = glob.glob(os.path.join(reference_path, '*.wav'))
reference_wavpaths = select_wav(reference_wavpaths, dur=3.0)
num_refs = len(reference_wavpaths)
print('# of reference wav paths: {}'.format(num_refs))

# get texts from LJSpeech val list
listfile = os.path.join(work_path, 'Data', 'LJSpeech', 'val_list.txt')
lines = open(listfile, 'r').readlines()
texts = [line.split('|')[1] for line in lines]
num_texts = len(texts)
print('# of texts: {}'.format(num_texts))

# get #words list and sort
nwords = count_word_from_texts(texts)
idx = np.argsort(nwords)

# get the sorted texts
texts_sorted = [texts[i] for i in idx]

# get texts with different lennth (in #words)
texts_short = texts_sorted[:num_refs]
texts_long = texts_sorted[-num_refs:]
texts_mid = texts_sorted[int((num_texts-num_refs)/2):int((num_texts+num_refs)/2)]

# get #words in different text-length level
nword_short = np.mean(count_word_from_texts(texts_short))
nword_long = np.mean(count_word_from_texts(texts_long))
nword_mid = np.mean(count_word_from_texts(texts_mid))
print('#words: {:.2f} (short), {:.2f} (mid), {:.2f} (long)'.format(
    nword_short, nword_mid, nword_long))

# set output dir
outpath = os.path.join(work_path, 'Outputs', 'RTF', 'manifests')
assert os.path.isdir(outpath), 'output dir: {} does not exist!'.format(output)

# set output files
outfile_short = os.path.join(outpath, 'manifest_short.txt')
outfile_long = os.path.join(outpath, 'manifest_long.txt')
outfile_mid = os.path.join(outpath, 'manifest_mid.txt')

# write output files
write_manifest(outfile_short, reference_wavpaths, texts_short)
write_manifest(outfile_long, reference_wavpaths, texts_long)
write_manifest(outfile_mid, reference_wavpaths, texts_mid)
