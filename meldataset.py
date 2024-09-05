#coding: utf-8
import os
import os.path as osp
import time
import random
import numpy as np
import random
import soundfile as sf
import librosa

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import pandas as pd

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes

np.random.seed(1)
random.seed(1)
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
}

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def find_phone_col_idx(entry, min_num_char=5):
    parts = entry.strip().split('|')
    idx = 0
    for i, part in enumerate(parts):
        part_nospace = part.replace(' ', '')
        L = min(len(part_nospace), min_num_char)
        cnt = 0
        for c in part_nospace[:L]:
            if c in _letters_ipa:
                cnt += 1
        if cnt/L > 0:
            idx = i
            break
    return idx

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 root_path,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 OOD_data="Data/OOD_texts.txt",
                 min_length=50,
                 ):

        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS

        # split data list
        _data_list = [l.strip().split('|') for l in data_list]

        # add dummy speaker id 0 for the single speaker case
        # dataset_data_list = [data if data[-1].isdigit() else (*data, 0) for data in _data_list]
        self.data_list = [data if data[-1].isdigit() else (*data, 0) for data in _data_list]

        # dataset_text_cleaner = TextCleaner()
        self.text_cleaner = TextCleaner()
        self.sr = sr

        # dataset_df = pd.DataFrame(dataset_data_list)
        self.df = pd.DataFrame(self.data_list)

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        
        self.min_length = min_length

        # read OOD data
        with open(OOD_data, 'r', encoding='utf-8') as f:
            tl = f.readlines()

        # get the idx of ipa phone column
        # (checking if the first few chars in each column contains ipa phones)
        idx = find_phone_col_idx(tl[0])

        # get phone texts
        self.ptexts = [t.split('|')[idx] for t in tl]
        
        # dataset_root_path = root_path
        self.root_path = root_path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        # data = dataset_data_list[idx]
        data = self.data_list[idx]
        path = data[0]
        
        wave, token, speaker_id = self._load_tensor(data)
        
        mel_tensor = preprocess(wave).squeeze()
        
        acoustic_feature = mel_tensor.squeeze()
        # truncate 0 or 1 column (2nd dim) at the end to make the 2nd dim a even number
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]
        
        # get a random in-domain reference audio sample
        # idx_speaker = dataset_df.shape[1] - 1
        idx_speaker = self.df.shape[1] - 1
        # print(f'idx of speaker: {idx_speaker}')
        # subset_speaker = dataset_df[dataset_df[idx_speaker] == str(speaker_id)]
        subset_speaker = self.df[self.df[idx_speaker] == str(speaker_id)]
        # print(f'subset shape: {subset_speaker.shape}')
        ref_data = (subset_speaker).sample(n=1).iloc[0].tolist()
        # get the reference mel and ref speaker label
        # ref_mel_tensor, ref_label = dataset._load_data(ref_data)
        ref_mel_tensor, ref_label = self._load_data(ref_data)
        
        # get a random out-of-domain (OOD) reference text sample
        
        ps = ""
        
        while len(ps) < self.min_length:
            rand_idx = np.random.randint(0, len(self.ptexts) - 1)
            ps = self.ptexts[rand_idx]
            
            ref_token = self.text_cleaner(ps)
            ref_token.insert(0, 0)
            ref_token.append(0)

            ref_token = torch.LongTensor(ref_token)
        
        return speaker_id, acoustic_feature, token, ref_token, ref_mel_tensor, ref_label, path, wave

    def _load_tensor(self, data):
        """data can be either 3 parts (rel path, text, ptext, speaker_id) or 4 parts"""
        ncols = len(data)
        if ncols == 3:
            wave_path, ptext, speaker_id = data
        elif ncols == 4:
            wave_path, _, ptext, speaker_id = data
        else:
            raise Exception('load tensor: check manifest file, #cols={}!'.format(ncols))
        speaker_id = int(speaker_id)
        # wave, sr = sf.read(osp.join(dataset_root_path, wave_path))
        wave, sr = sf.read(osp.join(self.root_path, wave_path))

        # convert to mono if needed
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()

        if sr != self.sr:
            # wave = librosa.resample(wave, orig_sr=sr, target_sr=dataset.sr)
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.sr)
            # print('resampled {} from sr:{} to sr:{}'.format(wave_path, sr, self.sr))

        # append 5000 samples of 0s to the beginning and to the end
        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
        
        # convert symbol to token
        # token = dataset_text_cleaner(ptext)
        token = self.text_cleaner(ptext)
        
        # append 0 (pad/silence) token to both end
        token.insert(0, 0)
        token.append(0)
        
        token = torch.LongTensor(token)

        return wave, token, speaker_id

    def _load_data(self, data):

        # wave, _, speaker_id = dataset._load_tensor(data)
        wave, _, speaker_id = self._load_tensor(data)

        mel_tensor = preprocess(wave).squeeze()

        # randomly get a segment with max_mel_length in mel_tensor
        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, speaker_id


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave
        

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rtext_length = max([b[3].shape[0] for b in batch])

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        paths = ['' for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]
        
        for bid, (label, mel, text, ref_text, ref_mel, ref_label, path, wave) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            ref_labels[bid] = ref_label
            waves[bid] = wave

        return waves, texts, input_lengths, ref_texts, ref_lengths, mels, output_lengths, ref_mels

def build_dataloader(path_list, root_path, sr,
                     validation=False,
                     OOD_data="Data/OOD_texts.txt",
                     min_length=50,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):
    
    dataset = FilePathDataset(path_list, root_path, sr,
                              OOD_data=OOD_data,
                              min_length=min_length,
                              validation=validation,
                              **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader

