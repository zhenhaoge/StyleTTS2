# extract audio segments from audio file (downloaded from YouTube) based on the time stamps
# in the transcription file, also save the meta data file
#
# procedures:
#   - (optional) extract the mono-channel of the audio file
#   - extract audio segments based on the original time stamps obtained from the subtitle file
#   - trim audio segment to shorten them (based on top_db)
#   - save the trimed segments (.wav) along with their meta files (.json)
#   - save the overall meta file (.csv)
#
# Zhenhao Ge, 2024-06-25

import os
from pathlib import Path
import argparse
from pydub import AudioSegment
import glob
import json
import librosa
import soundfile as sf

# set dirs
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

from audio import audioread, audiowrite, extract_wav_channel
from utils import set_path, empty_dir, tuple2csv
from Examples.dub_utils import get_sec, get_value_from_json, extract_segment
from Examples.dub_utils import parse_ass_file, parse_srt_file

def parse_args():
    usage = 'usage: extract audio segments'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--trans-file', type=str, help='transcription file (.ass or .srt)')
    parser.add_argument('--audio-file', type=str, help='audio file to extract segments from')
    parser.add_argument('--out-dir', type=str, help='output path to save segments')
    parser.add_argument('--meta-dir', type=str, help='dir of the meta data')
    parser.add_argument('--dur-lim', type=int, default=0,
        help='duration limit in minutes, 0 means no limit')
    parser.add_argument('--top-db', type=int, help='how much db below max to be considerd as silence')    
    parser.add_argument('--out-ver', type=str, help='output version, e.g. v{x}') 
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # data_dir = os.path.join(home_dir, 'data1', 'datasets', 'YouTube')
    # account_id = 'dr-wang'
    # recording_id = '20210915'
    # dur_id = 'full'
    # args = argparse.ArgumentParser()    
    # args.trans_file = os.path.join(data_dir, account_id, recording_id, dur_id, f'{recording_id}_L1.manual.srt')
    # args.audio_file = os.path.join(data_dir, account_id, recording_id, dur_id, f'{recording_id}_L1_vocals_mono.wav')
    # args.out_dir = os.path.join(work_dir, 'Outputs', 'YouTube', account_id, recording_id, dur_id, 'v1.original')
    # args.meta_dir = os.path.join(work_dir, 'Outputs', 'YouTube', account_id, recording_id, dur_id, 'meta')
    # args.dur_lim = 0
    # args.top_db = 20 # trimming parameter, 20 is more aggresive than 30
    # args.out_ver = 'v1'

    # check file/dir existence
    assert os.path.isfile(args.trans_file), f'transcription file: {args.trans_file} does not exist!'
    assert os.path.isfile(args.audio_file), f'audio file: {args.audio_file} does not exist!'

    # set output dir
    set_path(args.out_dir, verbose=True)
    empty_dir(args.out_dir)

    # set meta dir
    set_path(args.meta_dir, verbose=True)

    # get duration limit in seconds (args.dur_lim to dur_lim_sec)
    duration_total = librosa.get_duration(filename=args.audio_file)
    duration_total = round(duration_total, 2)
    if args.dur_lim == 0:
        dur_lim_sec = duration_total
    elif dur_lim > 0:
        dur_lim_sec = min(float(args.dur_lim*60), duration_total)
        dur_lim_sec = round(dur_lim_sec)
    else:
        raise Exception('duration (minutes) limit should be integer larger than 0 (no limit if 0)!')      
        
    # localize arguments
    trans_file = args.trans_file
    audio_file = args.audio_file
    out_dir = args.out_dir
    meta_dir = args.meta_dir
    top_db = args.top_db
    out_ver = args.out_ver

    # get recording id
    recording_id = os.path.basename(audio_file).split('_')[0]

    # print arguments
    print(f'recording id: {recording_id}')
    print(f'transcription file: {trans_file}')
    print(f'audio file: {audio_file}')
    print(f'output dir: {out_dir}')
    print(f'meta dir: {meta_dir}')
    print(f'duration limit: {dur_lim_sec} seconds ({dur_lim_sec/60:.2f} minutes)')
    print(f'top db: {top_db}')
    print(f'output version: {out_ver}')

    # convert audio file to mono-channel (if needed)
    wav = AudioSegment.from_file(audio_file)
    if wav.channels > 1:
        audio_file2 = audio_file.replace('.wav', '_mono.wav')
        
        if os.path.isfile(audio_file2):
            print(f'use existed mono-channel audio file instead: {audio_file2}')
        else:
            extract_wav_channel(audio_file, audio_file2, channel=0, verbose=True)
            print(f'use newly generated mono-channel audio file: {audio_file2}')    
        audio_file = audio_file2
    else:
        print(f'{audio_file} is already a mono-channel aduio file')

    # parse ass file to get the tuple list of (start_time, end_time, text)
    ext = os.path.splitext(trans_file)[1]
    if ext == '.ass':
        tuple_list = parse_ass_file(trans_file)
    elif ext == '.srt':
        tuple_list = parse_srt_file(trans_file)
    else:
        raise Exception('transcription file should be .ass or .srt file, but now it is {} file!')        
    nsegments = len(tuple_list)
    print(f'# of segments: {nsegments}')

    # trans_file2 = trans_file.replace('_zh', '_en')
    # tuple_list2 = parse_ass_file(trans_file2)

    # extract segments (top_db was reduced from 30 (default) to 20, to trim more agressively)
    extract_segment(audio_file, tuple_list, out_dir, dur_lim_sec, top_db=top_db, verbose=1)

    # get the output wav files
    out_wav_files = sorted(glob.glob(os.path.join(out_dir, '*.wav')))

    # get the output json files
    out_json_files = sorted(glob.glob(os.path.join(out_dir, '*.json')))

    # get the percentage of trimming
    durations = [get_value_from_json(f, 'duration') for f in out_json_files]
    durations_trimmed = [get_value_from_json(f, 'duration-trimmed') for f in out_json_files]
    dur_total = sum(durations)
    dur_trimmed_total = sum(durations_trimmed)
    print(f'trimming%: {(dur_total-dur_trimmed_total)/dur_total*100:.2f}%')

    assert len(out_wav_files) == len(out_json_files), 'output wav and json files mis-match!'
    nsegments_out = len(out_wav_files)

    # get header for the output meta csv
    with open(out_json_files[0]) as f:
        meta = json.load(f)
    header = list(meta.keys())
    header = ['wav-file'] + header

    # construct rows
    rows = [() for _ in range(nsegments_out)]
    for i in range(nsegments_out):

        # load meta for the current segment
        with open(out_json_files[i]) as f:
            meta = json.load(f)

        # get current row
        entry = [out_wav_files[i]]
        for k in header[1:]:
            entry += [meta[k]]
        rows[i] = tuple(entry)

        # rows[i] = (out_wav_files[i],
        #            meta['fid'],
        #            meta['start-time'],
        #            meta['end-time'],
        #            meta['duration'],
        #            meta['start-time-trimmed'],
        #            meta['end-time-trimmed'],
        #            meta['duration-trimmed'],
        #            meta['top_db'],
        #            meta['idx'],
        #            meta['text'])

    # write meta to csv
    out_csv_file = os.path.join(meta_dir, f'{recording_id}_meta_{out_ver}.csv')
    tuple2csv(rows, out_csv_file, delimiter='|', header=header, verbose=True)
