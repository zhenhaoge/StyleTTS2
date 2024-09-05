# translate texts using DeepL API for translation
#
# API reference: 
#   - documentation: https://developers.deepl.com/docs/api-reference/translate
#   - python library: https://github.com/DeepLcom/deepl-python?tab=readme-ov-file
#   - doc on translating text: https://developers.deepl.com/docs/api-reference/translate/openapi-spec-for-text-translation
#
# Zhenhao Ge, 2024-08-07

import os
from pathlib import Path
import argparse
import deepl

# set dirs
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

def get_key(api_key_file, account):

    lines = open(api_key_file, 'r').readlines()
    account2key = {}
    for line in lines:
        parts = line.rstrip().split('|')
        if parts[0] == account:
            return parts[1]
    return ''

def get_context(texts_context, lang='EN', verbose=0):

    # define punctuation set based on lang
    if lang == 'EN' or lang == 'EN-US':
        punctuations = ',.?!:;'
    elif lang == 'ZH':
        punctuations = '，。？!：；'
    else:
        raise Exception('support punctuations in EN and ZH only right now!')

    # get number sentences in the context
    N = len(texts_context)

    # concatenate context texts: get first sentence
    if verbose >= 2:
        print(f'0/{N}: {texts_context[0]}')
    context = texts_context[0]

    # concatenate context texts: get the sentences after the first sentence
    for i in range(1,N):
        if verbose >= 2:
            print(f'{i}/{N}: {texts_context[i]}')
        if texts_context[i-1][-1] in punctuations:
            # print(f'{texts_context[i-1][-1]} in punctuations')
            context += texts_context[i]
        else:
            # print(f'{texts_context[i-1][-1]} NOT in punctuations')
            context = context + punctuations[0] + ' ' + texts_context[i]
    if verbose >= 1:
        print(f'context: {context}')

    return context           

def translate_with_context(texts, cs, source_lang, target_lang, preserve_formatting=False, verbose=0):
    print(f'context size: {cs}')
    ntexts = len(texts)
    texts2 = ['' for _ in range(ntexts)]
    for i, text in enumerate(texts):
        idx_start = max(0, i-cs)
        idx_end = min(i+cs+1, ntexts)
        texts_context = texts[idx_start:idx_end]
        # context = ','.join(texts_context)
        context = get_context(texts_context, lang=source_lang, verbose=verbose)
        print(f'translating sentence {i}/{ntexts} (context size: {context_size} X 2 + 1) ...')
        # print(f'{i}/{ntexts}: {context}')
        result = translator.translate_text([text],
            source_lang=source_lang,
            target_lang=target_lang,
            preserve_formatting=preserve_formatting,
            context=context)
        text2 = result[0].text
        texts2[i] = text2
    return texts2

def parse_args():
    usage = 'usage: translate text from L1 to L2 using DeepL'
    parser.add_argument('--in-txt-file', type=str, help='input text file in L1 language')
    parser.add_argument('--out-txt-file', type=str, help='output text file in L2 language')
    parser.add_argument('--source-lang', type=str, help='source language (L1)')
    parser.add_argument('--target-lang', type=str, help='target language (L2)')
    parser.add_argument('--context-size', type=int, help='# of sentences before and after')
    parser.add_argument('--api-key-file',type=str, help='deepl api key')
    parser.add_argument('--account', type=str, help='account used to find the api key in the api key file')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # data_dir =  os.path.join(home_dir, 'data1', 'datasets', 'YouTube')
    # account_id = 'dr-wang'
    # recording_id = '20240805'
    # dur_id = 'full'
    # meta_dir = os.path.join(work_dir, 'Outputs', 'YouTube', account_id, recording_id, dur_id, 'meta')
    # args = argparse.ArgumentParser()
    # args.in_txt_file = os.path.join(meta_dir, f'{recording_id}_L1_text_v3.txt')
    # args.out_txt_file = os.path.join(meta_dir, f'{recording_id}_L2_text_v3.txt')
    # args.context_size = 2
    # args.source_lang = 'ZH'
    # args.target_lang = 'EN-US'
    # args.api_key_file = os.path.join(work_dir, 'Examples', 'deepl_api_keys.txt')
    # args.account = 'gezhenhaonuaa@gmail.com'

    # localize arguments
    in_txt_file = args.in_txt_file
    out_txt_file = args.out_txt_file
    source_lang = args.source_lang
    target_lang = args.target_lang
    context_size = args.context_size
    api_key_file = args.api_key_file
    account = args.account

    # print arguments
    print(f'input text file (L1): {in_txt_file}')
    print(f'output text file (L2): {out_txt_file}')
    print(f'source lang: {source_lang}')
    print(f'target lang: {target_lang}')
    print(f'context size: {context_size}')
    print(f'aip key file: {api_key_file}')
    print(f'account: {account}')

    # get the api key for deepl
    AUTH_KEY = get_key(api_key_file, account)
    translator = deepl.Translator(AUTH_KEY)

    # # test translator
    # texts = ['你好！']
    # result = translator.translate_text(texts, source_lang=source_lang, target_lang=target_lang)
    # print(f'{source_lang}: {texts[0]} -> {target_lang}: {result[0].text}')

    # check file/dir existence
    assert os.path.isfile(in_txt_file), f'input text file: {in_txt_file} does not exist!'

    # read in the texts in the source language
    lines = open(in_txt_file, 'r').readlines()
    texts = [line.strip() for line in lines]
    ntexts = len(texts)
    print(f'# of texts: {ntexts}')

    # translate the texts
    texts2 = translate_with_context(texts, context_size, source_lang, target_lang)

    # # show the first 10 translated sentences
    # print('\n'.join(texts2[:10]))

    # save the translated texts to the output file
    open(out_txt_file, 'w').writelines('\n'.join(texts2) + '\n')
    print(f'wrote the translated texts to: {out_txt_file}')
