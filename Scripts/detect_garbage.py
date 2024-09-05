# Dectect the garbage synthesized speech
#
# Zhenhao Ge, 2024-06-13

import os
from pathlib import Path
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
import argparse
import shutil

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

import utils

def get_duration(wavpaths):

    durs = [0 for _ in range(len(wavpaths))]
    for i, f in enumerate(wavpaths):
        try:
            durs[i] = wav_duration(f)
        except:
            continue
    return durs

def compute_energy(wavpath):

    y, sr = librosa.load(wavpath, sr=None)
    rms = librosa.feature.rms(y=y)
    energy = np.mean(rms)

    return energy

def map_path_dur_energy(wav_filepaths, wav_durs, energies):
    """map path to duration and energy"""

    path_dct = {}
    num_wavs = len(wav_filepaths)
    for i in range(num_wavs):
        wav_filepath = wav_filepaths[i]
        wav_id = os.path.splitext(os.path.basename(wav_filepath))[0]
        dur = wav_durs[i]
        energy = energies[i]
        path_dct[wav_id] = {'duration': dur, 'energy': energy}
    return path_dct

def extract_samples(wavpaths, out_path):
    for i in range(len(wavpaths)):
        wav_filepath = wavpaths[i]
        wav_filename = os.path.basename(wav_filepath)
        wav_filepaths = os.path.join(out_path, wav_filename)
        print('{} -{}'.format(wav_filepath, wav_filepath2))
        shutil.copyfile(wav_filepath, wav_filepath2)       

def parse_args():
    usage = 'detect the garbage synthesized speech'
    parser.add_argument('--wav-path', type=str, help='wav file path')
    parser.add_argument('--out-path1', type=str, help='output path for good samples')
    parser.add_argument('--out-path2', type=str, help='')

    return parser.parse_args()

if __name__ == '__main__': 

    # runtime mode
    args = parse_args()

    # interactive mode
    args = argparse.ArgumentParser()
    run_id = 'exp1'
    args.wav_path = os.path.join(work_path, 'Outputs', 'RTF', run_id)
    args.out_path1 = os.path.join(work_path, 'Outputs', 'RTF', '{}-good'.format(run_id))
    args.out_path2 = os.path.join(work_path, 'Outputs', 'RTF', '{}-bad'.format(run_id))
    
    # localize arguments
    wav_path = args.wav_path
    out_path1 = args.out_path1
    out_path2 = args.out_path2

    # set and create output dir (if needed)
    utils.set_path(args.out_path1)
    utils.set_path(args.out_path2)   

    # get wav files
    wav_filepaths = sorted(glob.glob(os.path.join(wav_path, '*.wav')))
    num_wavs = len(wav_filepaths)
    print('# of wav files: {}'.format(num_wavs))

    wav_durs = [librosa.get_duration(filename=f) for f in wav_filepaths]
    dur_median = np.median([d for d in wav_durs if d != 0])
    print('median duration: {:.2f} seconds'.format(dur_median))

    idx = np.argsort(wav_durs)
    wav_durs_sorted = [wav_durs[i] for i in idx]
    wav_filepaths_sorted = [wav_filepaths[i] for i in idx]

    plt.plot(wav_durs_sorted)
    figname1 = os.path.join(work_path, 'wav_durs_sorted.png')
    plt.savefig(figname1)
    plt.close()
    print('saved fig: {}'.format(figname1))

    energies_sorted = [compute_energy(f) for f in wav_filepaths_sorted]

    plt.plot(energies_sorted)
    figname2 = os.path.join(work_path, 'energies_sorted.png')
    plt.savefig(figname2)
    plt.close()
    print('save fig: {}'.format(figname2))

    # get dictionary for wav path to (duration, energy)
    path_dct = map_path_dur_energy(wav_filepaths_sorted, wav_durs_sorted, energies_sorted)
    
    # set thresholds in duration and energy
    T_dur = dur_median * 2
    T_energy = 0.3

    # split wav files to good and bad groups (sorted by dur)
    wav_filepaths_good, wav_filepaths_bad = [], []
    for i in range(num_wavs):
        wav_filepath = wav_filepaths[i]
        wav_id = os.path.splitext(os.path.basename(wav_filepath))[0]
        dur = path_dct[wav_id]['duration']
        engergy = path_dct[wav_id]['energy']
        if dur > T_dur or energy > T_energy:
            print('{}/{}, duration={:.2f}, energy={:.2f}'.format(i, num_wavs, dur, energy))
            wav_filepaths_bad.append(wav_filepath)
        else:
            wav_filepaths_good.append(wav_filepath)

    # wav_filepaths_bad, wav_filepaths_good = [], []
    # for i, (dur, energy) in enumerate(zip(wav_durs_sorted, energies_sorted)):
    #     wav_filepath = wav_filepaths_sorted[i]
    #     if dur > T_dur or energy > T_energy:
    #         wav_filepaths_bad.append(wav_filepath)
    #     else:
    #         wav_filepaths_good.append(wav_filepath)    

    num_wavs_bad = len(wav_filepaths_bad)
    print('num of wav bad: {}'.format(num_wavs_bad))

    # split good/bad samples into two sub-folders for examination
    extract_samples(wav_filepaths_good, out_path1)
    extract_samples(wav_filepaths_bad, out_path2)

    # # check individual sample
    # wav_filepath = wav_filepaths_bad[0]
    # wav_id = os.path.splitext(os.path.basename(wav_filepath))[0]
    # dur = path_dct[wav_id]['duration']
    # energy = path_dct[wav_id]['energy']
    # print(wav_filepath)

