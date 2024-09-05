# collection of utilies for dubbing pipeline
#
# Zhenhao Ge, 2024-07-29

import os
import librosa
import soundfile as sf
import json
import csv

# print('check: current dir: {}'.format(os.getcwd()))
# from utils import get_value_from_json

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

def get_value_from_json(jsonfile, key):
    with open(jsonfile) as f:
        dct = json.load(f)
        return dct[key]

def get_dur_from_file(wavfiles):

    num_wavfiles = len(wavfiles)
    durs = [0 for _ in range(num_wavfiles)]
    for i, f in enumerate(wavfiles):
        durs[i]= librosa.get_duration(path=f)
    return durs

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

def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def parse_ass_file(ass_file):
    """parse ass file to return tuple list with (start_time, end_time, text) per tuple"""

    lines = open(ass_file, 'r').readlines()
    lines = [line for line in lines if len(line) > 10 and line[:10] == 'Dialogue: ']
    nsegments = len(lines)

    tuple_list = [() for _ in range(nsegments)]
    for i, line in enumerate(lines):
        parts = line.strip().split(',')
        start = get_sec(parts[1])
        end = get_sec(parts[2])
        idx = parts[-1].find('}')
        text = parts[-1][idx+1:]
        tuple_list[i] = (start, end, text)

    return tuple_list

def parse_srt_file(srt_file):
    """parse ass file to return tuple list with (start_time, end_time, text) per tuple"""

    lines = open(srt_file, 'r').readlines()
    idxs_ts = [i for i, line in enumerate(lines) if '-->' in line]
    nsegments = len(idxs_ts)
    
    tuple_list = [() for _ in range(nsegments)]
    for i, idx in enumerate(idxs_ts):
        line = lines[idx]
        parts = line.rstrip().split(' --> ')
        start = hms2sec(parts[0])
        end = hms2sec(parts[1])
        text = lines[idx+1].rstrip()
        tuple_list[i] = (start, end, text)

    return tuple_list    

def sec2hms(seconds, precision=2):
    """convert seconds to hh:mm:ss,ms, where ms is 3 digits, e.g., 01:03:2,180"""
    hh = f'{int(seconds/3600):02d}' # 2 digits (00 ~ 99)
    mm = f'{int((seconds%3600)/60):02d}' # 2 digits (0 ~ 59)
    secs = seconds%3600%60 
    ss = f'{int(secs):d}' # either 1 or 2 digits (0 ~ 59)
    ms =  f'{int(round(secs-float(ss), precision)*1000):03d}' # 3 digits (000 ~ 999)
    hms = f'{hh}:{mm}:{ss},{ms}'
    return hms

def hms2sec(hms, precision=2):
    """convert hh:mm:ss,ms to seconds, where ms is 3 digits, e.g., 01:03:2,180""" 
    hh, mm, ss = hms.split(':')
    ss, ms = ss.split(',')
    seconds = int(hh) * 3600 + int(mm) * 60 + int(ss) + float(ms)/1000
    seconds = round(seconds, precision)
    return seconds

def extract_segment(audio_file, tuple_list, out_dir, dur_lim_sec, top_db=30, verbose=False):
    """extract segments with options to trim the leading and trailing silence"""

    nsegments = len(tuple_list)
    for i in range(nsegments):
        start_time, end_time, text = tuple_list[i]

        if end_time > dur_lim_sec:
            print(f'processed {i}/{nsegments} segments (duration limit: {int(dur_lim_sec/60)} min)')
            break

        duration = end_time - start_time
        # data, params = audioread(audio_file, start_time, duration)
        y, sr = librosa.load(audio_file, sr=None, offset=start_time, duration=duration)
        len_y = len(y)

        # trim audio and updated the time stamps
        y_trimmed, idx = librosa.effects.trim(y, top_db=top_db)
        len_y_trimmed = len(y_trimmed)
        dur_sil_leading = idx[0] / sr
        start_time_trimmed = start_time + dur_sil_leading
        dur_sil_tail = (len_y-idx[1]) / sr
        end_time_trimmed = end_time - dur_sil_tail
        duration_trimmed = end_time_trimmed - start_time_trimmed

        # fid = f'{i:04d}_{start_time:.2f}_{end_time:.2f}'
        fid = f'{i:04d}_{start_time_trimmed:.2f}_{end_time_trimmed:.2f}'
        out_file = os.path.join(out_dir, f'{fid}.wav')
        # audiowrite(out_file, data, params)
        sf.write(out_file, y, sr)
        if verbose >= 2:
            print(f'{i}/{num_segments_grouped}: wrote {out_file}')

        # write meta info to json file
        json_file = os.path.join(out_dir, f'{fid}.json')
        meta = {'fid': fid,
                'start-time': round(start_time, 2),
                'end-time': round(end_time, 2),
                'duration': round(end_time-start_time, 2),
                'start-time-trimmed': round(start_time_trimmed, 2),
                'end-time-trimmed': round(end_time_trimmed, 2),
                'duration-trimmed': round(end_time_trimmed-start_time_trimmed, 2),
                'top_db': top_db,
                'idx': i,
                'text': text}
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        if verbose >= 2:
            print(f'{i}/{num_segments_grouped}: wrote {json_file}')            

    if verbose >= 1:
        print(f'split {audio_file} and wrote {nsegments} segments to {out_dir}')     

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

def get_ts(ts_file, keep_ori_fid=True):
    """get (fid, start_time, end_time, start_fid, end_fid) from timestamp csv file"""

    lines = open(ts_file, 'r').readlines()
    header = lines[0].strip().split('|')
    assert header == ['idx', 'start-time', 'end-time', 'text'], \
        f'columns in {ts_file} are not correct!'
    lines = lines[1:]

    nsegments = len(lines)
    tuple_list = [() for _ in range(nsegments)]
    for i in range(nsegments):
        parts0 = lines[i].strip().split('|')
        nparts0 = len(parts0)

        # get start and end fids in the old version
        if i < nsegments-1: # not the last one
            parts1 = lines[i+1].strip().split('|')
            start_fid = int(parts0[0].split(',')[0])
            next_start_fid = int(parts1[0].split(',')[0])
            end_fid = next_start_fid - 1
        else:
            start_fid = int(parts0[0].split(',')[0])
            end_fid = int(parts0[0].split(',')[-1])

        # idxs = parts[0].split(',')
        # idxs = [int(idx) for idx in idxs]
        # nidxs = len(idxs)
        # start_fid = idxs[0]
        # end_fid = idxs[-1]

        # get the start, end time in the new version
        start_time = round(float(parts0[1]), 2)
        end_time = round(float(parts0[2]), 2)

        # get tuple list with or without text
        if nparts0 > 3:
            text = parts0[3]
            if keep_ori_fid:
                tuple_list[i] = (i, start_time, end_time, start_fid, end_fid, text)
            else:
                tuple_list[i] = (i, start_time, end_time, text) 
        else:
            if keep_ori_fid:
                tuple_list[i] = (i, start_time, end_time, start_fid, end_fid)
            else:
                tuple_list[i] = (i, start_time, end_time) 

    return tuple_list        

def group_ts(tuple_list, gap=0.1, verbose=False):
    """group time intervals with the gap <= gap_limit, e.g., if the start time of the current interval
       is smaller or equal to (the end time of the last interval + 0.1), then group them"""

    num_segments = len(tuple_list)

    # pre-processing
    cnt = 0
    start_time = tuple_list[0][1]
    start_fid = 0
    end_time_last = tuple_list[0][2]
    end_fid = 0

    # group segments
    i = 1
    tuple_list_new = []
    while i < num_segments:
        # if current segment start time is larger than the last segment end time
        # (i.e., not connected, so cannot be grouped)
        if tuple_list[i][1] > end_time_last + gap:

            end_time = end_time_last
            tuple_list_new.append((cnt, start_time, end_time, start_fid, end_fid))
            if verbose: print(tuple_list_new[-1])
            cnt += 1

            # set new start time and new start fid
            start_time = tuple_list[i][1]
            start_fid = i
        
        # keep update end time and end fid (set current end time as the last end time)    
        end_time_last = tuple_list[i][2]
        end_fid = i

        i += 1

    # post-processing (include the last grouped segment) 
    if start_time < end_time_last:
        end_time = end_time_last
        tuple_list_new.append((cnt, start_time, end_time, start_fid, end_fid)) 
        if verbose: print(tuple_list_new[-1])

    # sanity check new segments (fid_new, start_time, end_time, start_fid, end_fid)
    # (start_time is the start time in start_fid, end_time is the end time in end_fid)
    num_segments_grouped = len(tuple_list_new)
    for i in range(num_segments_grouped):
        fid, start_time, end_time, start_fid, end_fid = tuple_list_new[i]
        assert start_time == tuple_list[start_fid][1], \
            f'{i}/{num_segments_grouped}, new start time ({i}:{start_time}) != ' + \
            f'old start time ({start_fid}:{tuple_list[start_fid][1]})'
        assert end_time == tuple_list[end_fid][2], \
            f'{i}/{num_segments_grouped}, new end time ({i}:{end_time}) != ' + \
            f'old end time ({end_time}:{tuple_list[end_fid][2]})'

    return tuple_list_new

def extract_grouped_segment(audio_file, tuple_list_grouped, in_json_files, out_dir, refresh=False, verbose=0):
    """extract grouped audio segments, where len(tuple_list_grouped) <= len(in_json_files)
       in_json_files provides the original start/end time and texts to be concatenated"""

    num_segments_grouped = len(tuple_list_grouped)
    for i in range(num_segments_grouped):

        parts = tuple_list_grouped[i]
        nparts = len(parts)
        if nparts == 5: # first grouping
            fid, start_time, end_time, start_fid, end_fid = tuple_list_grouped[i]
        elif nparts == 6: # second grouping (update after manual correction)
            fid, start_time, end_time, start_fid, end_fid, text = tuple_list_grouped[i]
        else:
            raise Exception('grouped tuple list contains either 5 or 6 elements only!')

        duration = end_time - start_time
        y, sr = librosa.load(audio_file, sr=None, offset=start_time, duration=duration)

        fid = f'{i:04d}_{start_time:.2f}_{end_time:.2f}'
        out_file = os.path.join(out_dir, f'{fid}.wav')
        sf.write(out_file, y, sr)
        if verbose >= 2:
            print(f'{i}/{num_segments_grouped}: wrote {out_file}')

        # get the original start and end time
        if nparts == 5: # first grouping
            start_time_ori = get_value_from_json(in_json_files[start_fid], 'start-time-trimmed')
            end_time_ori = get_value_from_json(in_json_files[end_fid], 'end-time-trimmed')
        elif nparts == 6: # second grouping (update after manual correction)
            start_time_ori = get_value_from_json(in_json_files[start_fid], 'start-time-ori')
            end_time_ori = get_value_from_json(in_json_files[end_fid], 'end-time-ori')

        # get the grouped text (only when nparts == 5)
        if nparts == 5 or (nparts==6 and refresh):
            nsegs = end_fid - start_fid + 1
            texts = ['' for _ in range(nsegs)]
            for j, fid0 in enumerate(range(start_fid, end_fid+1)):
                texts[j] = get_value_from_json(in_json_files[fid0], 'text')
            text0 = ''.join(texts)
            text = text0

        if nparts == 6 and refresh:
            assert text == text0, f'check text: (new) {text} vs. (old) {text0}'
    
        # write meta info to json file
        json_file = os.path.join(out_dir, f'{fid}.json')
        meta = {'fid': fid,
                'start-time': round(start_time, 2),
                'end-time': round(end_time, 2),
                'duration': round(end_time-start_time, 2),
                'start-time-ori': round(start_time_ori, 2),
                'end-time-ori': round(end_time_ori, 2),
                'duration-ori': round(end_time_ori-start_time_ori, 2),
                'idx': i,
                'idx-ori-start': start_fid,
                'idx-ori-end': end_fid,
                'text': text}
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        if verbose >= 2:
            print(f'{i}/{num_segments_grouped}: wrote {json_file}')    

    if verbose >= 1:
        print(f'split {audio_file} and wrote {num_segments_grouped} segments to {out_dir} after grouping')

def count_letters(text):

    text_nospace = text.replace(' ', '')
    return len(text_nospace)

def split_text(text):
    """split text into two parts, which is even in length"""

    texts = text.split('.')
    texts = [t.strip() for t in texts if t != '']

    ntexts = len(texts)
    idx_split = int(ntexts/2)
    text1 = '. '.join(texts[:idx_split])
    text2 = '. '.join(texts[idx_split:])

    # if ntexts is odd, consider the alternative split
    if ntexts % 2 == 1:
        text3 = '. '.join(texts[:idx_split+1])
        text4 = '. '.join(texts[idx_split+1:])

        abs1 = abs(len(text1) - len(text2))
        abs2 = abs(len(text3) - len(text4))
        if abs2 < abs1:
            text1 = text3
            text2 = text4

    return  text1, text2      