# scale the translated audio segments to have the same total duration of the original segments
# adjust the start time and end time of the scaled segment to fit in the space, here are the strategies used
#  - use the context silence space with minimum gap
#  - scale more than the averge (up to the cap of 1.5x)
#  - shift the next segment for a delayed start time (offset)
#
# this script was developed similarly as ukr-tts/sofw/norm_spk_rate.py, but it more complicated
# norm_spk_rate.py just scale the segments with an overall scaling factor, but here it do other tricks to avoid
# overlap and still maintaining the syncness with the original segments
#
# Zhenhao Ge, 2024-06-28

import os
from pathlib import Path
import argparse
import glob
import librosa
import soundfile as sf
import numpy as np
import json
import csv

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

from audio import adjust_speed

def set_path(path, verbose=False):
    if os.path.isdir(path):
        if verbose:
            print('use existed path: {}'.format(path))
    else:
        os.makedirs(path)
        if verbose:
            print('created path: {}'.format(path))

def empty_dir(folder):
  for filename in os.listdir(folder):
      file_path = os.path.join(folder, filename)
      try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)
          elif os.path.isdir(file_path):
              shutil.rmtree(file_path)
      except Exception as e:
          print('Failed to delete %s. Reason: %s' % (file_path, e))

def tuple2csv(tuple_list, csvname, delimiter=',', header=[], verbose=True):
    with open(csvname, 'w', newline='') as f:
        csv_out = csv.writer(f, delimiter=delimiter)
        if header:
            csv_out.writerow(header)
        n = len(tuple_list)
        for i in range(n):
            csv_out.writerow(list(tuple_list[i]))
    if verbose:
        print('{} saved!'.format(csvname))          

def get_dur_from_file(wavfiles):

    num_wavfiles = len(wavfiles)
    durs = [0 for _ in range(num_wavfiles)]
    for i, f in enumerate(wavfiles):
        durs[i]= librosa.get_duration(path=f)
    return durs

def get_ts_from_filename(wavfiles):
    """get (fid, start_time, end_time) from filename"""

    num_segments = len(wavfiles)    
    tuple_list = [() for _ in range(num_segments)]
    for i in range(num_segments):
        wav_file = wavfiles[i]
        wav_filename = os.path.basename(wav_file)
        parts = os.path.splitext(wav_filename)[0].split('_')
        fid, start_time, end_time = parts[:3]
        fid = int(fid)
        start_time = float(start_time)
        end_time = float(end_time)
        tuple_list[i] = (fid, start_time, end_time)
    return tuple_list

def find_bound(ts_lst, idx, dur_total, gap=0.1):
    """find the lower and upper bounds of the segment with idx
       gap is the min duration between segments"""
    
    nsegments = len(ts_lst)

    if idx == 0: # first segment
        lower_bound = round(min(ts_lst[idx][1], gap), 2)
        upper_bound = round(ts_lst[1][1]-gap, 2)
    elif idx == nsegments - 1: # final segment
        lower_bound = round(ts_lst[idx-1][2]+gap, 2)
        upper_bound = round(dur_total, 2)
    else: # middle segments
        lower_bound = round(ts_lst[idx-1][2]+gap, 2)
        upper_bound = round(ts_lst[idx+1][1]-gap, 2)

    return lower_bound, upper_bound

def get_scaled_ts(lower_bound, upper_bound, duration_scaled):
    """get the timestamps of the scaled segment by putting it in the middle of (lower_bound, upper_bound)"""
    mid = lower_bound + (upper_bound-lower_bound)/2
    start_time_scaled = round(mid-duration_scaled/2, 2)
    end_time_scaled = round(mid+duration_scaled/2, 2)
    return start_time_scaled, end_time_scaled  

def parse_args():
    usage = 'usage: time-scale audio segments'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--in-dir', type=str, help='input dir of tts speech in its original speed')
    parser.add_argument('--ref-dir', type=str, help='reference dir of the real speech ' + \
        '(get actual duration to match in case there is no meta json file)')
    parser.add_argument('--out-dir', type=str, help='output dir of the output time-scaled tts speech')
    parser.add_argument('--meta-dir', type=str, help='dir for the meta data')
    parser.add_argument('--audio-file', type=str, help='audio file to extract segments from')
    parser.add_argument('--speed-lim', type=float, default=1.5, \
        help='speed-up limit from original tts segments to scaled tts segments')
    parser.add_argument('--dur_lim', type=int, help='duration to be processed in minutes')
    parser.add_argument('--out-ver', type=str, help='output version, e.g. v{x}')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # data_path = os.path.join(home_path, 'data1', 'datasets', 'YouTube')
    # account_id = 'laoming'
    # recording_id = '20220212'
    # args = argparse.ArgumentParser()
    # args.in_dir = os.path.join(work_path, 'Outputs', 'YouTube', account_id, recording_id, 'v4.translated')
    # args.ref_dir = os.path.join(work_path, 'Outputs', 'YouTube', account_id, recording_id, 'v3.corrected')
    # args.out_dir = os.path.join(work_path, 'Outputs', 'YouTube', account_id, recording_id, 'v6.scaled')
    # args.meta_dir = os.path.join(work_path, 'Outputs', 'YouTube', account_id, recording_id, 'meta')
    # args.audio_file = os.path.join(data_path, account_id, recording_id, f'{recording_id}_vocals_mono.wav')
    # args.speed_lim = 1.5 # speed up to 1.5x from the syn segment to the scaled segment
    # args.dur_lim = 5 # process up to 5 min
    # args.out_ver = 'v6'

    # check file/dir existence
    assert os.path.isdir(args.in_dir), f'input dir {args.in_dir} does not eixst!'
    assert os.path.isdir(args.ref_dir), f'reference dir: {args.ref_dir} does not exist!'
    assert os.path.isdir(args.meta_dir), f'meta dir: {args.meta_dir} does not exist!'
    assert os.path.isfile(args.audio_file), f'audio file: {args.audio_file} does not exist!'

    # create output dir
    set_path(args.out_dir, verbose=True)
    empty_dir(args.out_dir)

    # localize arguments
    in_dir = args.in_dir
    ref_dir = args.ref_dir
    out_dir = args.out_dir
    meta_dir = args.meta_dir
    audio_file = args.audio_file
    speed_lim = args.speed_lim
    dur_lim = args.dur_lim
    out_ver = args.out_ver

    # print arguments
    print(f'input dir: {in_dir}')
    print(f'reference dir: {ref_dir}')
    print(f'output dir: {out_dir}')
    print(f'meta dir: {meta_dir}')
    print(f'audio file: {audio_file}')
    print(f'speed-up limit: {speed_lim}x')
    print(f'duration limit: {dur_lim} minutes')
    print(f'output version: {out_ver}')

    # get the input audio files
    input_audiofiles = sorted(glob.glob(os.path.join(in_dir, '*.wav')))
    nsegments = len(input_audiofiles)
    print(f'# of segments: {nsegments}')

    # get the reference audio files
    reference_audiofiles = sorted(glob.glob(os.path.join(ref_dir, '*.wav')))
    assert len(reference_audiofiles) == nsegments, '# of segments in the input and reference dirs mis-match!'

    # get timestamp tuple list from the input audio filenames
    ts_lst = get_ts_from_filename(input_audiofiles)

    # get durations for the referenced and synthesized wavs from files directly
    durs_syn = get_dur_from_file(input_audiofiles)
    durs_ref = get_dur_from_file(reference_audiofiles)

    # get the total duration for the referenced and synthesized wavs
    dur_syn_total = sum(durs_syn)
    dur_ref_total = sum(durs_ref)

    # get the overall speed factor for speaking rate normalization
    speed = dur_syn_total / dur_ref_total
    print('average speed factor: {:.3f}'.format(speed))

    # get the duration of the entire audio file
    dur_total = librosa.get_duration(path=audio_file)

    # get the duration to be processed (seconds)
    dur_lim = min(dur_total, float(dur_lim*60))

    # find the adjusted timestgamps for the scaled segments    
    ts_ref = [() for _ in range(nsegments)]
    ts_allowed = [() for _ in range(nsegments)]
    ts_scaled = [() for _ in range(nsegments)]
    previous_upper_bound = 0
    duration_offset = 0
    duration_offsets = [0 for _ in range(nsegments)]
    gap = 0.1
    for i in range(nsegments):

        # get the timestamps about the reference chinese audio segment
        reference_audiofile = reference_audiofiles[i]
        parts = os.path.splitext(os.path.basename(reference_audiofile))[0].split('_')
        start_time_ref = round(float(parts[1]), 2)
        end_time_ref = round(float(parts[2]), 2)
        duration_ref = round(end_time_ref-start_time_ref, 2)
        ts_ref[i] = (start_time_ref, end_time_ref, duration_ref)

        # get the maximum-allowed timestamps based on the adjacent segments
        lower_bound, upper_bound = find_bound(ts_lst, i, dur_lim, gap=gap)
        # adjust the lowerbound to make sure it is larger than the previous upper bound
        lower_bound = max(previous_upper_bound, lower_bound) # prevous upper bound already include the gap
        duration_allowed = round(upper_bound-lower_bound, 2)
        assert duration_allowed > 0, \
            f'{i}/{nsegments}: check lower and upper bound ({lower_bound}, {upper_bound})'
        ts_allowed[i] = (lower_bound, upper_bound, duration_allowed)

        # get the pre-adjusted duration of scaled segment (dur_scaled) based on dur_syn and overall speed factor
        duration_syn = durs_syn[i]
        duration_scaled = round(duration_syn/speed, 2)

        if duration_scaled <= duration_ref-duration_offset:
            start_time_scaled = round(start_time_ref+duration_offset, 2)
            end_time_scaled = round(start_time_scaled+duration_scaled, 2)
            ts_scaled[i] = (start_time_scaled, end_time_scaled, duration_scaled)
            duration_offset = 0
        elif duration_scaled > duration_ref-duration_offset and duration_scaled <= duration_allowed:
            if duration_scaled < upper_bound - start_time_ref:
                start_time_scaled = start_time_ref
                end_time_scaled = start_time_ref + duration_scaled
            else:
                start_time_scaled, end_time_scaled = get_scaled_ts(lower_bound, upper_bound, duration_scaled)
            # assert end_time_scaled - start_time_scaled == duration_scaled, \
            #     f'{i}/{nsegments}: scaled timestamp disqualified (case 1)'
            ts_scaled[i] = (start_time_scaled, end_time_scaled, duration_scaled)
            duration_offset = 0
        elif duration_scaled > duration_allowed:
            speed2 = np.ceil(duration_syn/duration_allowed*100)/100
            # under speed cap
            if speed2 <= speed_lim:
                duration_scaled = round(duration_syn/speed2, 2)
                start_time_scaled, end_time_scaled = get_scaled_ts(lower_bound, upper_bound, duration_scaled)
                # assert end_time_scaled - start_time_scaled == duration_scaled, \
                #     f'{i}/{nsegments}: scaled timestamp disqualified (case 2)'
                ts_scaled[i] = (start_time_scaled, end_time_scaled, duration_scaled)
                duration_offset = 0
            # over speed cap, need to have duration offset    
            else:
                duration_scaled = round(duration_syn/speed_lim, 2)
                start_time_scaled = round(lower_bound, 2)
                end_time_scaled = round(start_time_scaled+duration_scaled, 2)
                ts_scaled[i] = (start_time_scaled, end_time_scaled, duration_scaled)
                duration_offset = round(end_time_scaled-upper_bound, 2)
                # raise Exception(f'{i}/{nsegments}: need to speed up to {speed2} to fit')
        else:
            raise Exception(f'{i}/{nsegments}: unexpected condition, please check!')
        duration_offsets[i] = duration_offset
        previous_upper_bound = round(end_time_scaled+gap, 2)

    # check duration offsets (cases that segment has to be shifted to a later start time due to no-space)
    duration_offsets_pos = [v for v in duration_offsets if v > 0]
    print(f'{len(duration_offsets_pos)}/{nsegments} segments has offset with avg. value ' + \
        f'{np.mean(duration_offsets_pos):.2f} secs.')

    # sanity check to ensure no overlaps in the adjacent segments
    for i in range(1, nsegments):
        start_time = ts_scaled[i][0]
        end_time_previous = ts_scaled[i-1][1]
        assert start_time > end_time_previous, f'{i}/{nsegments}: {(end_time_previous-start_time):.2f} overlap'

    # generate the scaled segments with statistics
    speeds = [0 for _ in range(nsegments)]
    durs_scaled = [0 for _ in range(nsegments)]
    durs_diff_abs = [0 for _ in range(nsegments)]
    durs_diff_percent = [0 for _ in range(nsegments)]
    xfactors = [0 for _ in range(nsegments)]

    for i in range(nsegments):

        duration_syn = durs_syn[i]
        duration_scaled = ts_scaled[i][2]
        speeds[i] = duration_syn / duration_scaled # >1: dur_scaled < dur_syn, speed up syn segment
        # print(f'{i}/{nsegments}: syn dur {duration_syn:.2f} / scaled dur {duration_scaled:.2f} ' + \
        #     f'(speed: {speeds[i]:.2f})')

        # collect statistics
        start_time_scaled = ts_scaled[i][0]
        end_time_scaled = ts_scaled[i][1]
        duration_scaled = ts_scaled[i][2]
        durs_scaled[i] = duration_scaled
        duration_ref = ts_ref[i][2]
        durs_diff_abs[i] = round(duration_scaled-duration_ref, 2)
        durs_diff_percent[i] = durs_diff_abs[i] / duration_ref
        xfactors[i] = duration_ref / duration_scaled # >1: dur_scaled < dur_ref, segment become shorter (ori -> scaled)

        # print message
        msg1 = f'{i}/{nsegments}: ({start_time_scaled:.2f}, {end_time_scaled:.2f}), ' + \
            f'speed: {speeds[i]:.2f} (syn -> scaled), '
        msg2 = f'duration: {ts_ref[i][2]:.2f} (ori) -> {duration_scaled:.2f} (scaled), ' + \
            f'abs: {durs_diff_abs[i]:.2f}, percent: {durs_diff_percent[i]*100:.2f}%, '
        msg3 = f'xfactor: {xfactors[i]:.2f} (ori -> scaled)'
        print(msg1 + msg2 + msg3)

        # write out scaled segment
        fid = f'{i:04d}_{start_time_scaled:.2f}_{end_time_scaled:.2f}_{speeds[i]:.2f}_{xfactors[i]:.2f}x'
        output_audiofile = os.path.join(out_dir, f'{fid}.wav')
        input_wav, output_wav, sr = adjust_speed(input_audiofiles[i], output_audiofile, speeds[i])

        # sanity check
        assert speeds[i] - (len(input_wav) / len(output_wav)) < 0.01, f'{i}/{nsegments} speed mis-match'

        # read meta from the input audio file
        input_jsonfile = input_audiofiles[i].replace('.wav', '.json')
        with open(input_jsonfile, encoding='utf-8') as f:
            meta0 = json.load(f) 

        # save meta
        output_jsonfile = os.path.join(out_dir, f'{fid}.json')
        meta = {'fid': fid,
                'idx': i,
                'start-time-zh': round(ts_ref[i][0], 2),
                'end-time-zh': round(ts_ref[i][1], 2),
                'duration-zh': round(ts_ref[i][2], 2),
                'start-time-en': round(ts_scaled[i][0], 2),
                'end-time-en': round(ts_scaled[i][1], 2),
                'duration-en': round(ts_scaled[i][2], 2),
                'speed': round(speeds[i], 2),
                'xfactor': round(xfactors[i], 2),
                'text-zh': meta0['text-zh'],
                'text-en': meta0['text-en']}
        with open(output_jsonfile, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    # collect all output jsonfiles
    output_jsonfiles = sorted(glob.glob(os.path.join(out_dir, '*.json')))
    assert len(output_jsonfiles) == nsegments, '# of the output jsonfiles mis-match!'

    # get the header for the output meta csv
    with open(output_jsonfile) as f:
        meta = json.load(f)
    header = list(meta.keys())
    header = ['wav-file'] + header

    # construct rows
    rows = [() for _ in range(nsegments)]
    for i in range(nsegments):
        with open(output_jsonfiles[i]) as f:
            meta = json.load(f)
        output_audiofile = output_jsonfiles[i].replace('.json', '.wav')
        rows[i] = (output_audiofile, *meta.values())

    # write meta to csv
    out_csv_file = os.path.join(meta_dir, f'{recording_id}_meta_{out_ver}.csv')
    tuple2csv(rows, out_csv_file, delimiter='|', header=header, verbose=True)
