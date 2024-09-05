# collection of additional audio utilities that are not compatible with
# the primary environment (e.g., conda environment:style)
#
# Zhenhao Ge, 2024-07-28

import os
import librosa
import soundfile as sf
import numpy as np
import shutil
from audiostretchy.stretch import stretch_audio

def adjust_speed(input_wavfile, output_wavfile, speed, verbose=False):

    # get ratio from speed
    # new audio duration / original audio duration (<1 means faster)
    ratio = 1 / speed

    output_wavfile_temp = output_wavfile.replace('.wav', '.tmp.wav')
    stretch_audio(input_wavfile, output_wavfile_temp, ratio)

    # get the input and output (temp) wavs
    input_wav, sr = librosa.load(input_wavfile, sr=None)
    output_wav_temp, sr2 = librosa.load(output_wavfile_temp, sr=None)
    assert sr2 == sr, 'input and output wav file have different sampling rate!'
    del sr2

    # get the input and output (temp) #samples
    input_nsamples = len(input_wav)
    output_nsamples_temp = len(output_wav_temp)

    # get the output #samples (should be based on the time-scaling factor)
    output_nsamples = int(np.ceil(input_nsamples/speed))

    # truncate the trailing silence if needed
    output_dur_temp = output_nsamples_temp / sr
    output_dur = output_nsamples / sr
    if output_nsamples_temp > output_nsamples:
        if verbose:
            print('truncating the trailing silence: {:.3f} sec. -> {:.3f} sec.'.format(
                output_dur_temp, output_dur))
        output_wav = output_wav_temp[:output_nsamples]
        sf.write(output_wavfile, output_wav, sr)
        os.remove(output_wavfile_temp)
    else:
        if verbose:
            print('output tmp dur: {:.3f}, output dur: {:.3f}, no truncation needed'.format(
                output_dur_temp, output_dur))
        shutil.move(output_wavfile_temp, output_wavfile)

    return input_wav, output_wav, sr
