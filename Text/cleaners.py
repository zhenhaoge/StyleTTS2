""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
import inflect
p = inflect.engine()
from unidecode import unidecode
from .numbers import normalize_numbers
from .roman import romanToDecimal


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def replace_roman_numeral(text):
    words = text.split()
    nwords = len(words)
    words2 = []
    for i, word in enumerate(words):
        word_roman = ''.join([c for c in word if c.isalpha()])
        is_roman = bool(re.search(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", word_roman))
        if is_roman:
            cond0a = i > 0
            if i < nwords - 1:
                cond0b = words[i+1][0].isupper()
            else:
                cond0b = False
            if cond0a and cond0b: # exist next word and the first letter of the next word is capitalized
                is_mid_name_initial = True
            else:
                is_mid_name_initial = False           
            cond1 = word_roman == word[:len(word_roman)]
            cond2 = len(word) > len(word_roman) and word[len(word_roman)] == '.'
            if (not is_mid_name_initial) and cond1 and cond2:
                decimal = romanToDecimal(word_roman)
                word_new = p.ordinal(decimal) + word[len(word_roman)+1:]
                words2.append('the')
                words2.append(word_new)
            else:
                words2.append(word)        
        else:
            words2.append(word)
    text2 = ' '.join(words2)        
    return text2               


def english_cleaners(text, flag_lowercase=True, flag_ascii=True):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    if flag_ascii: text = convert_to_ascii(text)
    text = replace_roman_numeral(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    if flag_lowercase: text = lowercase(text)
    return text