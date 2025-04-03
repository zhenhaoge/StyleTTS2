# generate the overlayed audio file from the time-scaled audio segments
#
# Zhenhao Ge, 2024-06-29

import os
from pathlib import Path
import argparse
import glob
import librosa
import soundfile as sf
import numpy as np

# set paths
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current path: {}'.format(os.getcwd()))

from utils import set_path, empty_dir, str2bool

def parse_args():
    usage = 'usage: generate the overlayed audio file from the time-scaled audio segments'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--seg-dir', type=str, help='dir for the scaled audio segments')
    parser.add_argument('--out-dir', type=str, help='output dir')
    parser.add_argument('--bg-audiofile', type=str, help='background audio file')
    parser.add_argument('--dur-lim', type=int, help='duration to be processed in minutes')
    parser.add_argument('--out-file', type=str, help='output audio file')
    parser.add_argument('--with-bg', type=str2bool, nargs='?', const=True,
        default=False, help="true if include the background")
    parser.add_argument('--r', type=float, default=1.0, help='ratio of the reduced volumne of the background audio vs full volume ' + \
        '(1: no reduce, 0: reduce completly)')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # data_dir = os.path.join(work_dir, 'Datasets', 'YouTube')
    # account_id = 'dr-wang'
    # recording_id = '20210915'
    # dur_id = 'full'
    # args = argparse.ArgumentParser()
    # args.seg_dir = os.path.join(work_dir, 'Outputs', 'YouTube', account_id, recording_id, dur_id, 'v6.scaled')
    # args.out_dir = os.path.join(work_dir, 'Outputs', 'YouTube', account_id, recording_id, dur_id, 'v7.opt1')
    # args.bg_audiofile = os.path.join(data_dir, account_id, recording_id, dur_id, f'{recording_id}_L1_accompaniment.wav')
    # args.dur_lim = -1 # use -1 for entire duration
    # args.out_file = os.path.join(args.out_dir, f'{recording_id}_bg+L2.wav')
    # args.with_bg = True
    # args.r = 1.0 # no volume reduction for the background

    # check dir/file existence
    assert os.path.isdir(args.seg_dir), f'segment dir: {args.seg_dir} does not exist!'
    assert os.path.isfile(args.bg_audiofile), f'background audio file: {args.bg_audiofile} does not exist!'

    # set dir
    set_path(args.out_dir, verbose=True)
    empty_dir(args.out_dir)

    # localize arguments
    seg_dir = args.seg_dir
    out_dir = args.out_dir
    out_file = args.out_file
    bg_audiofile = args.bg_audiofile
    dur_lim = args.dur_lim
    with_bg = args.with_bg
    r = args.r

    # get recording id
    recording_id = seg_dir.split(os.sep)[-3]

    # print arguments
    print(f'seg dir: {seg_dir}')
    print(f'out dir: {out_dir}')
    print(f'out file: {out_file}')
    print(f'background audio file: {bg_audiofile}')
    print(f'duration limit: {dur_lim} min')
    print(f'with background audio: {with_bg}')
    print(f'background volume reduction factor: {r}')

    # get the audio segments
    seg_audiofiles = sorted(glob.glob(os.path.join(seg_dir, '*.wav')))
    nsegments = len(seg_audiofiles)
    print(f'# of segments: {nsegments}')

    # get sampling rate
    _, sr0 = librosa.load(seg_audiofiles[0], sr=None)

    # get the duration (in seconds) of the background file
    # dur_total = librosa.get_duration(path=bg_audiofile)
    dur_total = librosa.get_duration(filename=bg_audiofile)
    print(f'background audio duration: {dur_total:.2f} sec. ({dur_total/60:.2f} min.)')

    # create base signal with silence at length of dur_lim min
    if dur_lim == -1:
        L0 = int(np.ceil(dur_total * sr0))
    else:    
        L0 = int(np.ceil(min(dur_lim*60, dur_total) * sr0)) # base signal sample length
    dur_lim_sec = L0 / sr0
    y0 = np.zeros(L0)

    # add in segments into the base signal
    diff_abs0 = int(0.005 * 2 * sr0) + 1 # the max difference due to start time and end time round error
    for i in range(nsegments):
        y, sr = librosa.load(seg_audiofiles[i], sr=None)
        nsamples = len(y)
        assert sr == sr0, f'{i}/{nsegments}: sampling rate inconsistent'
        parts = os.path.splitext(os.path.basename(seg_audiofiles[i]))[0].split('_')
        idx = int(parts[0])
        start_time = round(float(parts[1]), 2)
        end_time = round(float(parts[2]), 2)
        start_idx = int(start_time*sr0)
        end_idx = int(end_time*sr0)
        nsamples2 = end_idx - start_idx
        assert end_idx < L0, f'{i}/{nsegment}: segment end-time exceed the singal boundary'
        diff_abs = np.abs(nsamples-nsamples2)
        assert  diff_abs <= diff_abs0, \
            f'{i}/{nsegments}: segment sample length ({nsamples}) and the allocated sample length ' + \
            f'({nsamples2}) should differ no more than {diff_abs0}, but now {diff_abs}'
        y0[start_idx:start_idx+nsamples] = y

    # write pure voval audio file
    out_audiofile = os.path.join(out_dir, f'{recording_id}_vovals_L2.wav')
    sf.write(out_audiofile, y0, sr0)
    print(f'wrote {out_audiofile}')

    if with_bg:

        # read the background file (up to the duration limit)
        y1, _ = librosa.load(bg_audiofile, sr=sr0, mono=True, offset=0.0, duration=float(dur_lim_sec))
        L1 = len(y1)
        # assert L1 == L0 or L1-L0 == 1, f'check L0 ({L0}) and L1 ({L1})'

        # combine foreground (vocals) and background (music) with a reducing factor
        L = min(L0, L1)
        y2 = y0[:L] + y1[:L] * r

    else:

        y2 = y0 

    dur_out = len(y2) / sr0
    print(f'output audio duration: {dur_out:.2f} sec. ({dur_out/60:.2f} min.)')

    # write out the overlayed signal
    sf.write(out_file, y2, sr0)
    print(f'wrote {out_file}')