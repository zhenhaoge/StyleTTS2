# testing examples of using DeepL API for translation
# (try to see how to get the same or closer results compared to the online translation, by copying texts on https://www.deepl.com/en/translator)
#
# API reference: 
#   - documentation: https://developers.deepl.com/docs/api-reference/translate
#   - python library: https://github.com/DeepLcom/deepl-python?tab=readme-ov-file
#   - doc on translating text: https://developers.deepl.com/docs/api-reference/translate/openapi-spec-for-text-translation
#
# Zhenhao Ge, 2024-07-29

import os
from pathlib import Path
import argparse
import deepl

AUTH_KEY = 'bc3857ea-9e4e-456e-861b-f71bdfc0da2f:fx'
translator = deepl.Translator(AUTH_KEY)

# set dirs
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

def translate(texts, bs, source_lang, target_lang, preserve_formatting, formality):
    print(f'batch size: {bs}')
    print(f'configs: preserve formatting={preserve_formatting}, formality={formality}')
    texts2 = []
    for i in range(0, ntexts, bs):
        idx_start, idx_end = i, min(i+bs,ntexts)
        print(f'translating sentences [{idx_start}, {idx_end}), {ntexts} total ...')
        texts_batch = texts[idx_start:idx_end]
        result = translator.translate_text(texts_batch,
            source_lang=source_lang,
            target_lang=target_lang,
            preserve_formatting=preserve_formatting,
            formality=formality)
        bs2 = len(result)
        assert bs2 == bs, f'batch size should be {bs}, but now {bs2}!'
        for j in range(bs):
            texts2.append(result[j].text)
    return texts2

def translate_with_context(texts, cs, source_lang, target_lang, preserve_formatting=False):
    print(f'context size: {cs}')
    ntexts = len(texts)
    texts2 = ['' for _ in range(ntexts)]
    for i, text in enumerate(texts):
        print(f'translating sentence {i}/{ntexts} ...')
        idx_start = max(0, i-cs)
        idx_end = min(i+cs, ntexts)
        context = ','.join(texts[idx_start:idx_end])
        # context = ','.join(texts)
        result = translator.translate_text([text], source_lang=source_lang, target_lang=target_lang, context=context)
        texts2[i] = result[0].text
    return texts2    

def parse_args():
    usage = 'usage: translate text from L1 to L2 using DeepL'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--in-txt-file', type=str, help='input text file in L1 language')
    parser.add_argument('--out-txt-file', type=str, help='output text file in L2 language')
    parser.add_argument('--source-lang', type=str, help='source language (L1)')
    parser.add_argument('--target-lang', type=str, help='target language (L2)')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()
    
    # interactive mode
    data_dir = os.path.join(home_dir, 'data1', 'datasets', 'YouTube')
    account_id = 'laoming'
    recording_id = '20220212'
    dur_id = 'full'
    meta_dir = os.path.join(work_dir, 'Outputs', 'YouTube', account_id, recording_id, dur_id, 'meta')
    args = argparse.ArgumentParser()
    args.in_txt_file = os.path.join(meta_dir, f'{recording_id}_L1_text_v3.txt')
    args.out_txt_file = os.path.join(meta_dir, f'{recording_id}_L2_text_v3.txt')
    args.source_lang = 'ZH'
    args.target_lang = 'EN-US'

    # loacalize arguments
    in_txt_file = args.in_txt_file
    out_txt_file = args.out_txt_file
    source_lang = args.source_lang
    target_lang = args.target_lang

    # check file/dir existence
    assert os.path.isfile(in_txt_file), f'input text file: {in_txt_file} does not exist!'

    # read in the texts in the source language
    lines = open(in_txt_file, 'r').readlines()
    texts = [line.strip() for line in lines]
    ntexts = len(texts)
    print(f'# of texts: {ntexts}')

    #%% basic testing

    # option 1: translate with batch size 1 and overlap 0 (line by line without overlap)
    texts2 = ['' for _ in range(ntexts)]
    for i, text in enumerate(texts):
        print(f'translating sentence {i}/{ntexts} ...')
        result = translator.translate_text(text, source_lang=source_lang, target_lang=target_lang)
        texts2[i] = result.text

    # option 2: translate with max allowed batch size 50 and overlap 0 (50 lines at a time without overlap)
    # (no difference compared with option 1)
    bs = 50
    preserve_formatting=True # True or False
    formality = 'prefer_less' # default, more, less, prefer_more, prefer_less
    text2 = translate(texts, bs, source_lang, target_lang, preserve_formatting, formality)

    # write out the translated texts
    open(out_txt_file, 'w').writelines('\n'.join(texts2) + '\n')
    print(f'wrote translated texts to {out_txt_file}')

    #%% more testing
    nlines = 10
    texts_short = texts[:nlines]
    print('\n'.join(texts_short))

    result = translator.translate_text(texts_short, target_lang="EN-US", context=','.join(texts_short)) # adding context gets closer to the online results

    # various options, but same output (1. after trillions of years: no, humans: yes)
    result = translator.translate_text(texts_short, target_lang="EN-US") 
    result = translator.translate_text(texts_short, target_lang="EN-US", split_sentences='nonewlines') 

    # various options, but same output
    # (split_sentences, and preserve_formatting does not change anything, formality does not work for EN-US)
    result = translator.translate_text(texts_short, target_lang="EN-US", context=','.join(texts_short), split_sentences='nonewlines')
    result = translator.translate_text(texts_short, target_lang="EN-US", context=','.join(texts_short), split_sentences='0')
    result = translator.translate_text(texts_short, target_lang="EN-US", context=','.join(texts_short), split_sentences='1') # split_sentences default: 1
    result = translator.translate_text(texts_short, target_lang="EN-US", context=','.join(texts_short), preserve_formatting=True)
    result = translator.translate_text(texts_short, target_lang="EN-US", context=','.join(texts_short), preserve_formatting=False) # preserve_formatting default: False
    result = translator.translate_text(texts_short, target_lang="EN-US", context=','.join(texts_short), tag_handling='html')
    result = translator.translate_text(texts_short, target_lang="EN-US", context=','.join(texts_short), tag_handling='xml')
    result = translator.translate_text(texts_short, source_lang='ZH', target_lang="EN-US", context=','.join(texts_short))

    # experiment with different context size (take all texts as context does not help, it is over kill)
    cs = 5 # context size 2 means take 2 more lines before and 2 more lines after, plus itself, totally 5 lines
    texts_translated = translate_with_context(texts[:20], cs, source_lang, target_lang)

    # test how much context is enough
    context = ','.join(texts[:15])
    result = translator.translate_text(texts[:50], source_lang=source_lang, target_lang=target_lang, context=context, split_sentences='0')
    texts_translated = [result[i].text for i in range(nlines)]

    # show translated results
    print('\n'.join(texts_translated))
