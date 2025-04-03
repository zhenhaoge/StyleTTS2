# select audio segments based on speaker embedding similarity and concatenate the selected segments
# to form a reliable speaker embedding
# 
# the concatenated audio segment is used in 03_gen_segment.py, so do this before runnning that script
#
# note: current method only applies to recording with one speaker
#
# Zhenhao Ge, 2024-06-28

import os
from pathlib import Path
import argparse
import glob
import torch
from speechbrain.pretrained import EncoderClassifier
import torch.nn.functional as F
import librosa
import soundfile as sf
import torchaudio
import numpy as np
from numpy.linalg import norm

# set dirs
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

# set the samlpe rate of the data used to train the speaker embedding model
sample_rate = 16000

spk_model = {
    "speechbrain/spkrec-xvect-voxceleb": 512, 
    "speechbrain/spkrec-ecapa-voxceleb": 192,
}

def signal2embed(signal, classifier, size_embed):
    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        embeddings = F.normalize(embeddings, dim=2)
        embeddings = embeddings.squeeze().cpu().numpy()
    assert embeddings.shape[0] == size_embed, embeddings.shape[0]
    return embeddings

def cos_sim(A,B):
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    return cosine    

def parse_args():
    usage = 'usage: select audio segments to get a reliable speaker embedding'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--in-dir', type=str, help='input dir containing audio segments')
    parser.add_argument('--concat-audiofile', type=str, help='concatenated audio files with selected segments')
    parser.add_argument('--speaker-embed-model', type=str, help='speaker embedding model')
    parser.add_argument('--dur-search', type=float, help='duration (seconds) to search from')
    parser.add_argument('--dur-select', type=float, help='duration (seconds) to select after search')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # data_dir = os.path.join(home_dir, 'data1', 'datasets', 'YouTube')
    # account_id = 'laoming'
    # recording_id = '20220212'
    # dur_id = 'full'
    # args.in_dir = os.path.join(work_dir, 'Outputs', 'YouTube', account_id, recording_id, dur_id, 'v1.original')
    # args.concat_audiofile = os.path.join(data_dir, account_id, recording_id, dur_id, f'{recording_id}_L1_spk.wav')
    # # args.speaker_embed_model = "speechbrain/spkrec-ecapa-voxceleb"
    # args.speaker_embed_model = "speechbrain/spkrec-xvect-voxceleb"
    # args.dur_search = 600.0
    # args.dur_select = 30.0

    # check file/dir existence
    assert os.path.isdir(args.in_dir), f'input dir: {args.in_dir} does not exist!'

    # localize arguments
    in_dir = args.in_dir
    concat_audiofile = args.concat_audiofile
    speaker_embed_model = args.speaker_embed_model
    dur_search = args.dur_search
    dur_select = args.dur_select

    # print arguments
    print(f'input dir: {in_dir}')
    print(f'concatenated audio file: {concat_audiofile}')
    print(f'speaker embedding model: {speaker_embed_model}')
    print(f'duration to search: {dur_search} seconds')
    print(f'duration to select: {dur_select} seconds')

    # get audio segments
    audiofiles = glob.glob(os.path.join(in_dir, '*.wav'))
    nsegments = len(audiofiles)
    print(f'# of segments in input dir {in_dir}: {nsegments}')

    # get audiofiles to search (accumulate the first few segments up to total duration of dur_search)
    durations = []
    dur_total = 0
    for i in range(nsegments):
        dur = librosa.get_duration(filename=audiofiles[i])
        durations.append(round(dur, 2))
        dur_total += dur
        if dur_total > dur_search:
            break
    nsegments_search = len(durations)
    audiofiles_search = audiofiles[:nsegments_search]
    print(f'the first {nsegments_search} segments with total duration of {dur_total/60:.2f} mins were selected to search from')

    # load speaker embedding (xvector) model 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source=speaker_embed_model, run_opts={"device": device}, savedir=os.path.join('/tmp', speaker_embed_model))
    size_embed = spk_model[speaker_embed_model]
    print(f'speaker model: {speaker_embed_model} with dimension {size_embed} was selected')

    # get speaker embeddings
    embedding_lst = ['' for _ in range(nsegments_search)]
    for i in range(nsegments_search):
        print(f'processing segment {i}/{nsegments_search} ...')
        audiofile = audiofiles_search[i]
        y, sr = librosa.load(audiofile, sr=sample_rate)
        signal = torch.from_numpy(y).unsqueeze(0)
        embedding_lst[i] = signal2embed(signal, classifier, size_embed)

    # compute speaker similarity score between each two segments
    scores = np.zeros((nsegments_search, nsegments_search))
    for i in range(nsegments_search):
        for j in range(nsegments_search):
            scores[i][j] = cos_sim(embedding_lst[i], embedding_lst[j])

    # get the avg. speaker score per segment
    spk_scores = np.mean(scores, axis=0)
    print(f'mean speaker score: {np.mean(spk_scores):.3f}')

    # get the idxs with scores in descending order
    idxs = np.argsort(spk_scores)[::-1]

    # get the sorted spk scores and its corresponding segment duration
    spk_scores_descend = [spk_scores[idx] for idx in idxs]
    durations_descend = [durations[idx] for idx in idxs]

    # select the first few segments with total duration up to dur_select
    cnt = 0
    dur_total = 0
    while dur_total < dur_select:
        dur_total += durations_descend[cnt]
        cnt += 1
    nsegments_select = cnt
    # idxs_select = sorted(idxs[:nsegments_select]) # reorder based on the number of index
    idxs_select = idxs[:nsegments_select]
    print(f'selected {nsegments_select} segments with avg. speaker score >= {spk_scores[idxs_select[-1]]:.3f}')

    # get the selected audiofiles (which are the most clean ones)
    audiofiles_select = ['' for _ in range(nsegments_select)]
    for i in range(nsegments_select):
        audiofiles_select[i] = audiofiles_search[idxs_select[i]]

    # concatenate these selected audio segments
    y = np.array([])
    for i, audiofile in enumerate(audiofiles_select):
        y0, sr = librosa.load(audiofile, sr=None)
        y = np.append(y, y0)

    # write the concatenated segment
    sf.write(concat_audiofile, y, sr)
    print(f'wrote the concatenated audio file: {concat_audiofile}')
  