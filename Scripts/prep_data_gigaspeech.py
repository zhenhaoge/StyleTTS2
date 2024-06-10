# prepare data for GigaSpeech
#
# it uses the same train/val split as MQTTS
#
# Zhenhao Ge, 2024-05-30

import os
from pathlib import Path
import glob
import argparse

# import phonemizer
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(
    language='en-us', preserve_punctuation=True,  with_stress=True)

# import word tokenizer
from nltk.tokenize import word_tokenize

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

# load local modules
from text_utils import clean_text

def parse_args():
    usage = 'usage: prepare the GigaSpeech data for the StyleTTS training'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--data-path', type=str, help='data path which contains wavs')
    parser.add_argument('--meta-filepath', type=str, help='file path to the meta csv file, which contain the texts')
    parser.add_argument('--id-filepath', type=str, help='reference file to get ids')
    parser.add_argument('--out-filepath', type=str, help='output manifest file path')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # interactive mode
    args = argparse.ArgumentParser()

    work_path = os.getcwd()
    args.data_path = os.path.join(work_path, 'Datasets', 'GigaSpeech-Alex')
    args.meta_filepath = os.path.join(work_path, '')