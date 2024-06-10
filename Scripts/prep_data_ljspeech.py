# prepare data for LJSpeech
# 
# it takes the splits used in StyleTTS2 as a reference, to ensure it generate the same file
# so it just a walk-through to generate the same data files as the LJSpeech data used in StyleTTS2,
# but with an extra column which is the original text
#
# Zhenhao Ge, 2024-05-27

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

# import string aligner
from string2string.alignment import NeedlemanWunsch
nw = NeedlemanWunsch()

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

# load local modules
# from text_utils import TextCleaner
# textclenaer = TextCleaner()
from text_utils import clean_text
from utils import get_fid, get_fid2wav, get_fid2text, get_fid2ps
from utils import tuple2csv
from Text.cleaners import expand_abbreviations, expand_numbers

# import importlib
# importlib.reload(Text.cleaners.english_cleaners)

def get_manifest(fids, fid2text, spkr_id):
    """fid in fids is rel file path, while fid in fid2text is the unique id, 
       which may not be the rel path"""

    num_fids = len(fids)
    tuple_list = [() for _ in range(num_fids)]
    for i in range(num_fids):

        fid = fids[i]
        fid0 = os.path.splitext(os.path.basename(fid))[0]
        text = fid2text[fid0]
        ps = global_phonemizer.phonemize([text])[0]
        ps1 = word_tokenize(ps)
        ps2 = ' '.join(ps1)

        # # convert phone seqs to tokens
        # tokens = textclenaer(ps2)

        # # insert value at index to tokens
        # index = 0
        # value = 0
        # tokens.insert(index, value)

        tuple_list[i] = (fid, text, ps2, spkr_id)
        # tuple_list[i] = (fid, ps2, spkr_id)
    
    return tuple_list

def compare_ps(ps0_list, ps1_list):

    # get the length of ps list
    L0, L1 = len(ps0_list), len(ps1_list)
    assert L0 == L1, '#phones mismatch: ({} vs {})!'.format(L0, L1)
    L = L0
    del L0, L1

    diff_dct = {}
    for (p0, p1) in zip(ps0_list, ps1_list):
        if p0 != p1:
            if (p0, p1) in diff_dct.keys(): 
                diff_dct[(p0, p1)] += 1
            else:
                diff_dct[(p0, p1)] = 1

    return diff_dct

def append_dct(diff_acc_dct, diff_dct):
    for k in diff_dct.keys():
        if k in diff_acc_dct.keys():
            diff_acc_dct[k] = diff_acc_dct[k] + diff_dct[k]
        else:
            diff_acc_dct[k] = diff_dct[k]
    return diff_acc_dct

def get_aligned_ps(ps0, ps1, ph='0'):
    ps0_list = [*ps0]
    ps1_list = [*ps1]
    ps0_aligned, ps1_aligned = nw.get_alignment(ps0_list, ps1_list, return_score_matrix=False)
    ps0_aligned = ps0_aligned.replace('-', ph).replace(' | ', '').strip()
    ps1_aligned = ps1_aligned.replace('-', ph).replace(' | ', '').strip()
    return ps0_aligned, ps1_aligned

# def get_grouped_list(ps, ph='0'):
#     """get phoneme grouped list where each element is one phoneme with various length,
#         e.g., 'ɔː' is a phoneme with length 2, and ʌ is a phoneme with length 1"""

#     ps_nospace = ps.replace(' ', '')
#     L = len(ps_nospace)

#     ps_list = []
#     i = 0
#     while i < L:
#         idx_start = i
#         idx_end = None
#         if ps_nospace[i] == ph:
#             idx_start += 1
#             i += 1
#         elif i < L-1 and ps_nospace[i+1]=='ː':
#             idx_end = i + 2
#             i += 2
#         else:
#             idx_end = i + 1
#             i += 1
#         if idx_end != None:  
#             ps_list.append(ps_nospace[idx_start:idx_end])

#     return ps_list

def get_grouped_lists(ps0, ps1, ph='0'):

    # get ps0 and ps1 without space
    ps0_nospace = ps0.replace(' ', '')
    ps1_nospace = ps1.replace(' ', '')

    # get the length of ps0_nospace and ps1_nospace
    L0 = len(ps0_nospace)
    L1 = len(ps1_nospace)
    assert L0 == L1, 'ps0_nospace & ps1_nospace length mismatch: {} vs. {}!'.format(L0, L1)
    L = L0
    del L0, L1

    ps0_list, ps1_list = [], []
    i = 0
    while i < L:
        idx_start = i
        idx_end = None
        if ps0_nospace[i] == ph or ps1_nospace[i] == ph:
            if i > 0 and ps0_nospace[i-1] != ps1_nospace[i-1]:
                ps0_list[-1] = ps0_list[-1] + ps0_nospace[i]
                ps1_list[-1] = ps1_list[-1] + ps1_nospace[i]
                idx_start += 1
                i += 1
                continue
        if i < L - 1 and (ps0_nospace[i+1]=='ː' or ps1_nospace[i+1]=='ː'):
            idx_end = i + 2
            i += 2
        else:
            idx_end = i + 1
            i += 1
        ps0_list.append(ps0_nospace[idx_start:idx_end].replace(ph, ''))
        ps1_list.append(ps1_nospace[idx_start:idx_end].replace(ph, ''))
    return ps0_list, ps1_list, L

def parse_args():
    usage = 'usage: prepare the LJSpeech data for the StyleTTS training'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--data-path', type=str, help='data path which contains wavs')
    parser.add_argument('--meta-filepath', type=str, help='file path to the meta csv file, which contain the texts')
    parser.add_argument('--id-filepath', type=str, help='reference file to get ids')
    parser.add_argument('--out-filepath', type=str, help='output manifest file path')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # work_path = os.getcwd()
    # args.data_path = os.path.join(home_path, 'data1/datasets/LJSpeech/wavs')
    # args.meta_filepath = os.path.join(home_path, 'data1/datasets/LJSpeech/metadata.csv')
    # args.id_filepath = os.path.join(work_path, 'Data', 'train_list.txt')
    # args.out_filepath = os.path.join(work_path, 'Data', 'LJSpeech', 'train_list.txt')

    print('data path: {}'.format(args.data_path))
    print('meta file path: {}'.format(args.meta_filepath))
    print('id file path: {}'.format(args.id_filepath))
    print('output file path: {}'.format(args.out_filepath))

    # create the output path if it does not exist 
    out_path = os.path.dirname(args.out_filepath)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
        print('created output dir: {}'.format(args.out_path))
    else:
        print('using existing output dir: {}'.format(out_path))

    # get wav files from data path
    wavfiles = sorted(glob.glob(os.path.join(args.data_path, '*.wav')))
    num_wavfiles = len(wavfiles)
    print('# of wav files in {}: {}'.format(args.data_path, num_wavfiles))

    # get fid2wav dict
    fid2wav = get_fid2wav(wavfiles, args.data_path)

    # get wav ids from the reference id file path
    fids = get_fid(args.id_filepath)
    num_fids = len(fids)
    print('# of fids in {}: {}'.format(args.id_filepath, num_fids))

    # get fid2text dict from meta file
    fid2text = get_fid2text(args.meta_filepath)

    # clean texts in fid2text
    fid2text2 = {fid: clean_text(text, flag_ascii=False, flag_lowercase=False) for fid, text in fid2text.items()}

    # fid2text2 = {}
    # for i, (fid, text) in enumerate(fid2text.items()):
    #     text2 = clean_text(text, flag_ascii=False, flag_lowercase=False)
    #     fid2text2[fid] = text2

    # get the manifest tuple list
    tuple_list = get_manifest(fids, fid2text2, spkr_id=0)
     
    # write output manifest file
    tuple2csv(tuple_list, args.out_filepath, delimiter='|')

    # compare with the author's ipa phone sequence (optional)
    manifest_filepath0 = args.out_filepath
    print('getting fid2ps from {} ...'.format(manifest_filepath0))
    fid2ps0 = get_fid2ps(manifest_filepath0, idx_fid=0, idx_ps=2, delimiter='|')

    manifest_filepath1 = args.id_filepath
    print('getting fid2ps from {} ...'.format(manifest_filepath1))
    fid2ps1 = get_fid2ps(manifest_filepath1, idx_fid=0, idx_ps=1, delimiter='|')

    # get the skip list for outliers
    if 'train' in os.path.basename(args.id_filepath):
        skip_list = [9633] # one name with roman that the author's version has not detected
    elif 'val' in os.path.basename(args.id_filepath):
        skip_list = [18] # author's version has extara words
    else:
        skip_list = []

    # get the count of diff pairs
    diff_acc_dct = {}
    ph = '0'    
    Ls = []
    for i in range(num_fids):

        fid = fids[i]
        # print('processing {}/{}: {}'.format(i, num_fids, fid))

        fid0 = os.path.splitext(os.path.basename(fid))[0]
        text_ori = fid2text[fid0]
        text_exp0 = expand_abbreviations(text_ori)
        text_exp1 = expand_numbers(text_exp0)

        cond0 = '"' in text_ori or 'first place' in text_ori
        cond1 = i in skip_list
        cond2 = text_exp0 != text_ori
        cond3 = text_exp1 != text_ori
        if cond0 or cond1 or cond2 or cond3:
            continue
    
        ps0 = fid2ps0[fid]
        ps1 = fid2ps1[fid]

        ps0_aligned, ps1_aligned = get_aligned_ps(ps0, ps1, ph=ph)
        # ps0_list = get_grouped_list(ps0_aligned, ph=ph)
        # ps1_list = get_grouped_list(ps1_aligned, ph=ph)
        ps0_list, ps1_list, L = get_grouped_lists(ps0_aligned, ps1_aligned, ph=ph)
        Ls.append(L)

        diff_dct = compare_ps(ps0_list, ps1_list)
        diff_acc_dct = append_dct(diff_acc_dct, diff_dct)

    # get percentage of different phones (val: 1.13%, train: 1.17)
    num_diff_phones = sum(diff_acc_dct.values())
    num_all_phones = sum(Ls)
    percent_diff = num_diff_phones / num_all_phones
    message = "there are {:.2f}% phones different" + \
        " compared with the author's ipa phone sequences in {}"
    print(message.format(percent_diff*100, manifest_filepath1))
