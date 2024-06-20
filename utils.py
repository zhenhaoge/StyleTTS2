import os
from monotonic_align import maximum_path
from monotonic_align import mask_from_lens
from monotonic_align.core import maximum_path_c
import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
from munch import Munch
import subprocess
import re
import csv

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent =  np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
  path =  np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

  t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
  t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)

def set_path(path, verbose=False):
    if os.path.isdir(path):
        if verbose:
            print('use existed path: {}'.format(path))
    else:
        os.makedirs(path)
        if verbose:
            print('created path: {}'.format(path))

def get_hostname():
    hostname = subprocess.check_output('hostname').decode('ascii').rstrip()
    return hostname

def get_gpu_info(device=-1):
    """get gpu info, device is the index of GPU device, e.g., 0, 1, 2, or 3"""
    line_as_bytes = subprocess.check_output("nvidia-smi -L", shell=True)
    lines = line_as_bytes.decode("ascii").split('\n')
    lines = [line for line in lines if line != '']
    nlines = len(lines)
    if device == -1:
        lines = [re.sub("\(.*?\)","()", line).replace('()','').strip() for line in lines]
        string = '\n'.join(lines)
        # print(string)
    elif device >= 0 and device < nlines:
        line = lines[device]
        string = re.sub("\(.*?\)","()", line).replace('()','').strip()
    else:
        string = ''
        raise Exception('device: {} out of range (0~{})'.format(device,nlines-1))
    return string

def read_manifest(manifest_file):
    lines = open(manifest_file, 'r').readlines()
    reference_dicts = {}
    for i, line in enumerate(lines):
        parts = line.split('|')
        ref_path = parts[0].strip()
        ref_path = os.path.join(os.getcwd(), ref_path)
        text = parts[1].strip()
        reference_dicts[i] = (ref_path, text)
    return reference_dicts

def get_fid(id_filepath):
    lines = open(id_filepath, 'r').readlines()
    # fids = [os.path.splitext(line.split('|')[0])[0] for line in lines]
    fids = [line.split('|')[0] for line in lines]
    return fids

def get_fid2wav(wavfiles, data_path):
    """get fid2wav dict (using the rel path as fid)"""
    # fid2wav = {os.path.splitext(os.path.basename(wavfile))[0]:wavfile for wavfile in wavfiles}
    fid2wav = {os.path.relpath(wavfile, data_path):wavfile for wavfile in wavfiles}
    return fid2wav

def get_fid2text(meta_filepath):
    """get fid2text dict, this fid is real fid without filename extension,
       since the fid in the first column of the meta csv file does not contain extension"""
    lines =  open(meta_filepath, 'r').readlines()
    fid2text = {}
    for line in lines:
        parts = line.strip().split('|')
        fid, text = parts[0], parts[1]
        # text = clean_text(text,
        #     convert_to_ascii=convert_to_ascii, convert_to_lowercase=convert_to_lowercase)
        fid2text[fid] = text
    return fid2text

def get_fid2ps(manifest_filepath, idx_fid, idx_ps, delimiter='|'):
    lines = open(manifest_filepath, 'r').readlines()
    fid2ps = {}
    for line in lines:
        parts = line.strip().split(delimiter)
        fid, ps = parts[idx_fid], parts[idx_ps]
        # fid = os.path.splitext(fid)[0]
        fid2ps[fid] = ps
    return fid2ps

def tuple2csv(tuple_list, csvname, delimiter=',', verbose=True):
    with open(csvname, 'w', newline='') as f:
        csv_out = csv.writer(f, delimiter=delimiter)
        n = len(tuple_list)
        for i in range(n):
            csv_out.writerow(list(tuple_list[i]))
    if verbose:
        print('{} saved!'.format(csvname))

