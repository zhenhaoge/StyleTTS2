# IPA Phonemizer: https://github.com/bootphon/phonemizer

from Text.cleaners import english_cleaners

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

punc_list = ['.', ',', '?', '!']

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
        print(len(dicts))
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes

def clean_text(text, replace_word=False, flag_ascii=True, flag_lowercase=True):
    """clean text, refer to `Inference_LJSpeech.py`"""

    text = text.strip()
    text = text.replace('"', '') # remove '"'
    text = text.replace('-', ' ') # replace '-' with space
    text = english_cleaners(text, replace_word=replace_word, flag_ascii=flag_ascii, flag_lowercase=flag_lowercase)
    text = text.replace('-', ' ') # replace '-' with space
    return text

def attach_punc(text, punc_list=punc_list):

    words = text.split()
    nwords = len(words)
    words2 = []
    for i, word in enumerate(words):
        if word in punc_list and i > 0:
            words2[-1] += word
        else:
            words2.append(word)
    text2 = ' '.join(words2)
    return text2
