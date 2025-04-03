# concatenate speech segments extracted using timestamps
#
# Experiment the quality and speed using word-level accumulation method to generate speech in a streaming setting
# 3rd step: concatenate speech segments
#
# Zhenhao Ge, 2024-10-22

import os
from pathlib import Path
import argparse
import glob
import json
import pyloudnorm as pyln
import librosa
import numpy as np
from pydub import AudioSegment
import soundfile as sf

home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

from utils import tuple2csv, convertible_to_integer
from audio import audioread, audiowrite, normalize_audio

sr = 24000 # TTS audio sampling rate
# meter = pyln.Meter(sr)

def filter_file(filelist):

    filelist2 = [f for f in filelist if '_t' not in f]
    filelist2 = [f for f in filelist2 if '_concat_' not in f]
    filelist2 = [f for f in filelist2 if '_reference' not in f]

    return filelist2

def get_float_precision(number):
    # Convert the float to string
    float_str = str(number)
    # Find the decimal point
    if '.' in float_str:
        # Split on the decimal point and count the digits after it
        return len(float_str.split('.')[1])
    else:
        # No decimal point means precision is 0
        return 0 

def round_snap(raw_time, snap_dur=0.0125):
    """round toward multiples of snap_dur"""
    precision = get_float_precision(snap_dur)
    rounded_time = round(round(raw_time/snap_dur) * snap_dur, precision)
    return rounded_time

def extract_word_ts(tgfile, idx_word_start=0, idx_word_end=-1, round_method='regular', round_precision=2, snap_dur=0.125):
    """extract word and timestamp tuples from the TextGrid file in json format"""
    # read the TextGrid file with alignment
    with open(tgfile, 'r') as json_file:
        data_dict = json.load(json_file)
        # get the # of words    
        nwords = len(data_dict['words'])
        # update the end word idx
        if idx_word_end == -1:
            idx_word_end = nwords

    # wavfile = tgfile.replace('.TextGrid', '.wav')
    # duration = librosa.get_duration(filename=wavfile)

    # extract the word timestamps
    word_ts = []
    for i in range(idx_word_start, idx_word_end):
        word = data_dict['words'][i]
        alignedWord = word['alignedWord']
        start_time_raw = float(word['start'])
        end_time_raw = float(word['end'])
        if round_method == 'regular':
            start_time = round(start_time_raw, round_precision)
            end_time = round(end_time_raw, round_precision)
        elif round_method == 'snap':
            start_time = round_snap(start_time_raw, snap_dur)
            end_time = round_snap(end_time_raw, snap_dur)
        word_ts.append([alignedWord, start_time, end_time])

    # adjust the start time of the first word based on the gap to the previous word
    if idx_word_start > 0:
        word_previous = data_dict['words'][idx_word_start-1]
        end_time_previous_raw = float(word_previous['end'])
        end_time_previous = round(end_time_previous_raw, round_precision)
        start_time_first = word_ts[0][1]
        start_time_first_updated_raw = (start_time_first + end_time_previous) / 2
        if round_method == 'regular':
            start_time_first_updated = round(start_time_first_updated_raw, round_precision)
        elif round_method == 'snap':
            start_time_first_updated = round_snap(start_time_first_updated_raw, snap_dur)
            word_ts[0][1] = start_time_first_updated
    
    # adjust the end time of the last word based on the gap of the next word
    if idx_word_end < nwords:
        word_next = data_dict['words'][idx_word_end]
        start_time_next = round(float(word_next['start']), round_precision)
        end_time_last = word_ts[-1][2]
        end_time_last_updated_raw = (start_time_next + end_time_last) / 2
        if round_method == 'regular':
            end_time_last_updated = round(end_time_last_updated_raw, round_precision)
        elif round_method == 'snap':
            end_time_last_updated = round_snap(end_time_last_updated_raw, snap_dur)
        word_ts[-1][2] = end_time_last_updated

    return word_ts

def get_tgfile_tuples(texts, tgfiles, nwords_future):

    ntexts = len(texts)
    ntgfiles = len(tgfiles)
    assert ntexts == ntgfiles, 'len(texts) and len(tgfiles) mis-match!'

    # get the start idx of texts (the first text with nwords > nwords_future)
    idx_start = [i for i, text in enumerate(texts) if len(text.split())>nwords_future][0]

    words_upto_current = [[] for _ in range(ntexts-idx_start)]
    for i in range(idx_start, ntexts):
        words_upto_current[i-idx_start] = texts[i].split()[:-nwords_future]

    # # sanity check: print out words upto the current
    # for i, words in enumerate(words_upto_current):
    #     print(f'{i}/{ntexts-idx_start}: ' + ' '.join(words))

    words_current = ['' for _ in range(ntexts-idx_start)]
    words_current[0] = words_upto_current[0]

    # fid_idxs_tuples = ['' for _ in range(ntexts-idx_start)]
    # fid = os.path.splitext(os.path.basename(out_wavfiles[idx_start]))[0]
    # idx_word_start = 0
    # idx_word_end = len(words_upto_current[0])
    # fid_idxs_tuples[0] = (fid, idx_word_start, idx_word_end)

    for i in range(idx_start+1, ntexts):

        idx_word_start = len(words_upto_current[i-idx_start-1])
        idx_word_end = len(words_upto_current[i-idx_start])
        # print(f'i:{i}, idx: {idx_word_start}, {idx_word_end}')
        words_current[i-idx_start] = words_upto_current[i-idx_start][idx_word_start:idx_word_end]

        # fid = os.path.splitext(os.path.basename(out_wavfiles[i]))[0]
        # fid_idxs_tuples[i-idx_start] = (fid, idx_word_start, idx_word_end)

    # # sanity check: print out the final current words
    # for i, words in enumerate(words_current):
    #     print(f'{i}/{ntexts-idx_start}: ' + ' '.join(words))

    # get tgfile tuples (tgfile, words, idx_word_start, idx_word_end)
    ntexts2 = ntexts - idx_start
    assert len(words_current) == ntexts2, 'check words_current!'
    tgfile_tuples = [() for _ in range(ntexts2)]
    for i in range(ntexts2):
        if i == 0:
            idx_word_start = 0
            idx_word_end = len(words_upto_current[i])
        else:
            idx_word_start = len(words_upto_current[i-1])
            idx_word_end = len(words_upto_current[i])
        tgfile_tuples[i] = [tgfiles[i+idx_start], words_current[i], idx_word_start, idx_word_end]

    return tgfile_tuples

def update_crossfade(dur0, dur1, crossfade):
    dur = min(dur0, dur1)
    cf = min(int(dur/2), crossfade)
    return cf    

def crossfade_segment(audio0, audio1, crossfade, type='regular'):

    # update crossfade duration based on the durations of aduio0 and audio1
    dur0 = int(audio0.duration_seconds * 1000)
    dur1 = int(audio1.duration_seconds * 1000)
    cf = update_crossfade(dur0, dur1, crossfade)
    if type == 'regular':
        combined = audio0.append(audio1, crossfade=cf)
    elif type == 'samedur':
        crossfade_part = audio0[-cf:].append(audio1[:cf], crossfade=cf)
        audio0_nonoverlapped = audio0[:-cf]
        audio1_nonoverlapped = audio1[cf:]
        # # combine 3 parts (has data format issue)
        # combined = audio0_nonoverlapped + crossfade_part + audio1_nonoverlapped
        # combine 3 parts (data format issue fixed)
        crossfade_part2 = AudioSegment(crossfade_part._data,
            frame_rate=audio0.frame_rate, sample_width=audio0.sample_width, channels=audio0.channels)
        if isinstance(audio0_nonoverlapped._data, np.ndarray):
            audio0_nonoverlapped2 = AudioSegment(audio0_nonoverlapped._data.tobytes(),
                frame_rate=audio0.frame_rate, sample_width=audio0.sample_width, channels=audio0.channels)
        else:
           audio0_nonoverlapped2 = AudioSegment(audio0_nonoverlapped._data,
                frame_rate=audio0.frame_rate, sample_width=audio0.sample_width, channels=audio0.channels)
        audio1_nonoverlapped2 = AudioSegment(audio1_nonoverlapped._data.tobytes(),
            frame_rate=audio1.frame_rate, sample_width=audio1.sample_width, channels=audio1.channels)
        combined = audio0_nonoverlapped2 + crossfade_part2 + audio1_nonoverlapped2
        # # combine 3 parts (alternative method, does not work)
        # crossfade_part_samples = crossfade_part.get_array_of_samples()
        # crossfade_part_array = np.array(crossfade_part_samples)
        # crossfade_part2 = crossfade_part
        # crossfade_part2._data = crossfade_part_array
        # combined = audio0_nonoverlapped + crossfade_part2 + audio1_nonoverlapped
        # dur_combined = int(combined.duration_seconds * 1000)
        # print(f'duration after crossfade ({type}): {dur_combined} ms')
    return combined  

def parse_args():

    usage = 'usage: concatenate speech segments extracted by timestamps'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--output-path', type=str, help='root output path')
    parser.add_argument('--exp-id', type=str, help='exp id')
    parser.add_argument('--ref-id', type=str, help='ref id')
    parser.add_argument('--nwords-future', type=int, help='#words after extraction')
    parser.add_argument('--crossfade-dur', type=float, help='crossfade duration in ms')
    parser.add_argument('--tolerance-dur', type=float, help='tolerance duration in ms')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # work_path = os.getcwd() # e.g., '/home/users/zge/code/repo/style-tts2'
    # args.output_path = os.path.join(work_path, 'Outputs', 'Scratch', 'LibriTTS')
    # args.exp_id = 2
    # args.ref_id = 'YOU1000000038_S0000079'
    # args.nwords_future = 2
    # args.crossfade_dur = 5
    # args.tolerance_dur = 2.5

    # set the output path
    if convertible_to_integer(args.exp_id):
        args.exp_id = int(args.exp_id)
        output_folder = f'exp-{args.exp_id:02d}'
    else:
        output_folder = f'gs-{args.exp_id}'
    output_path = os.path.join(args.output_path, output_folder)   
    if not os.path.isdir(output_path):
        print(f'creating new output dir: {output_path}')
        os.makedirs(output_path)
    else:
        print(f'using existing output dir: {output_path}')

    # get the sorted wav files
    out_wavfiles = sorted(glob.glob(os.path.join(output_path, f'{args.ref_id}*.wav')))
    out_wavfiles = filter_file(out_wavfiles)
    idxs = [int(os.path.basename(f).split('-')[1]) for f in out_wavfiles]
    out_wavfiles = [f for _, f in sorted(zip(idxs, out_wavfiles))]

    # get the text files and TextGrid files
    out_txtfiles = [f.replace('.wav', '.txt') for f in out_wavfiles]
    out_tgfiles = [f.replace('.wav', '.TextGrid') for f in out_wavfiles]

    # get texts
    texts = [open(f, 'r').readlines()[0].strip() for f in out_txtfiles]

    # get the number of texts
    ntexts = len(texts)
    print(f'# of texts: {ntexts}')

    # get tuple list for (.TextGrid file, current words, idx_word_start, idx_word_end)
    tgfile_tuples = get_tgfile_tuples(texts, out_tgfiles, args.nwords_future)

    # include future words for the last tgfile_tuples
    tgfile_last = tgfile_tuples[-1][0]
    idx_word_start = tgfile_tuples[-1][2]
    words_ts_last = extract_word_ts(tgfile_last, idx_word_start=idx_word_start, idx_word_end=-1, round_method='regular', round_precision=2)
    words_last = [sublst[0] for sublst in words_ts_last]
    tgfile_tuples[-1][1] = words_last
    tgfile_tuples[-1][3] = idx_word_start + len(words_last)

    idx_start = [i for i, text in enumerate(texts) if len(text.split()) > args.nwords_future][0]
    ntexts2 = ntexts - idx_start
    texts2 = texts[idx_start:]

    # extract the words from tgfiles based on tgfile_tuples
    seg_list = [{} for _ in range(ntexts2)]
    for i in range(ntexts2):

        # parse the TextGrid tuples
        tgfile, words, idx_word_start, idx_word_end = tgfile_tuples[i]

        # get a list of (word, start_time, end_time) tuples for the selected words
        words_ts_sel = extract_word_ts(tgfile, idx_word_start, idx_word_end, round_method='regular', round_precision=2)

        start_time = max(0, words_ts_sel[0][1] - args.tolerance_dur/1000)
        end_time = words_ts_sel[-1][2] + args.tolerance_dur/1000
        duration = end_time - start_time

        wavfile = tgfile.replace('.TextGrid', '.wav')
        assert os.path.isfile(wavfile), f'wav file: {wavfile} does not exist!'

        # read/write using audioread/audiowrite (method 1)
        data, params = audioread(wavfile, start_time, duration)
        seg_list[i] = {'data': data, 'params': params}
        seg_wavfile = wavfile.replace('.wav', f'_{idx_word_start}-{idx_word_end}_t{args.tolerance_dur}.wav')
        audiowrite(seg_wavfile, data, params)
        print(f'{i}/{ntexts2}: wrote {seg_wavfile}')

    # read durations and rtfs from meta json files
    durations_proc = [0 for _ in range(ntexts)]
    durations_out = [0 for _ in range(ntexts)]
    rtfs = [0 for _ in range(ntexts)]
    for i in range(ntexts):

        out_jsonfile = out_wavfiles[i].replace('.wav', '.json')
        assert os.path.isfile(out_jsonfile), f'meta json file: {out_jsonfile} does not exist!'

        with open(out_jsonfile, 'r') as json_file:
            meta = json.load(json_file)

        durations_proc[i] = meta['dur-proc']
        durations_out[i] = meta['dur-out']
        rtfs[i] = meta['rtf']    

    # collect meta data for the texts and audio samples (before getting concatenated sample)
    rows = [() for _ in range(ntexts2)]
    for i in range(ntexts2):
        text = texts2[i]
        nwords = len(text.split())
        tgfile, words_current, idx_word_start, idx_word_end = tgfile_tuples[i]
        wavfile = tgfile.replace('.TextGrid', '.wav')
        dur_all = librosa.get_duration(filename=wavfile, sr=sr)
        seg_wavfile = wavfile.replace('.wav', f'_{idx_word_start}-{idx_word_end}_t{args.tolerance_dur}.wav')
        dur_words = librosa.get_duration(filename=seg_wavfile, sr=sr)
        text_current = ' '.join(words_current)
        fid = os.path.splitext(os.path.basename(tgfile))[0]
        seg_id = f'{fid}_{idx_word_start}-{idx_word_end}_t{args.tolerance_dur}'
        dur_proc = durations_proc[i+idx_start]
        rtf = rtfs[i+idx_start]
        rows[i] = (i, text, nwords, text_current, fid, seg_id, f'{dur_proc:.3f}', f'{dur_all:.3f}', f'{rtf:.3f}', f'{dur_words:.3f}')

    # write meta data to csv    
    csvfile = os.path.join(args.output_path, f'{output_folder}.csv')
    header = ['id', 'text', 'nwords', 'words', 'file-id', 'seg-id', 'dur-proc', 'dur-all', 'rtf', 'dur-words']
    tuple2csv(rows, csvfile, delimiter="|", header=header)

    # concatenate audio segments without crossfading and tolerance
    wavfile_concat = os.path.join(output_path, f'{args.ref_id}_concat_cf0_t0.wav')
    data_list = [seg['data'] for seg in seg_list]
    nframes_list = [seg['params'][3] for seg in seg_list]
    data = np.concatenate(data_list)
    nframes = sum(nframes_list)
    params[3] = nframes
    audiowrite(wavfile_concat, data, params)
    print(f'wrote {wavfile_concat}')

    # print the overall RTF
    duration_concat = nframes / sr
    duration_proc = sum(durations_proc)
    rtf_overall =  duration_proc / duration_concat 
    print(f'overall RTF for {os.path.basename(wavfile_concat)}: {rtf_overall:.3f} ({duration_proc:.3f}/{duration_concat})')

    # save the overall RTF in json file
    jsonfile = os.path.join(args.output_path, f'{output_folder}.json')
    meta = {'dur-concat': duration_concat, 'duration-proc': duration_proc, 'rtf-overall': rtf_overall}
    with open(jsonfile, 'w') as fp:
        json.dump(meta, fp, indent=2)
    print(f'wrote overall RTF to json file: {jsonfile}')    

    # sample rate used in normalization
    sample_rate = 16000

    # save the normalized version
    audio = normalize_audio(wavfile_concat, dB=-20.0, sample_rate=sample_rate)
    wavfile_concat_norm = wavfile_concat.replace('.wav', '_norm.wav')
    sf.write(wavfile_concat_norm, audio, sample_rate)
    print(f'wrote the normalized wav file: {wavfile_concat_norm}')

    # set crossfade type
    cf_type = 'regular' # 'regular' or 'samedur'

    # concatenate audio segments with crossfading and tolerance
    channels, sample_width, frame_rate = params[:3]
    combined = AudioSegment(data=data_list[0].tobytes(), sample_width=sample_width, frame_rate=frame_rate, channels=channels)
    for i, data in enumerate(data_list[1:]):
        seg = AudioSegment(data=data.tobytes(), sample_width=sample_width, frame_rate=frame_rate, channels=channels)
        combined = crossfade_segment(combined, seg, args.crossfade_dur, type=cf_type)
    wavfile_concat_cf = os.path.join(output_path, f'{args.ref_id}_concat_cf{args.crossfade_dur}_t{args.tolerance_dur}.wav')
    combined.export(wavfile_concat_cf, format="wav")

    # save the normalized version
    audio = normalize_audio(wavfile_concat_cf, dB=-20.0, sample_rate=sample_rate)
    wavfile_concat_cf_norm = wavfile_concat_cf.replace('.wav', '_norm.wav')
    sf.write(wavfile_concat_cf_norm, audio, sample_rate)
    print(f'wrote the normalized wav file: {wavfile_concat_cf_norm}')
