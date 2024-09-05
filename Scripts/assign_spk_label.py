import os
from pathlib import Path
import argparse
import pickle
import librosa
import torch
from speechbrain.pretrained import EncoderClassifier
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import shutil

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

from utils import get_fid

def signal2embed(signal, classifier, size_embed):
    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        embeddings = F.normalize(embeddings, dim=2)
        embeddings = embeddings.squeeze().cpu().numpy()
    assert embeddings.shape[0] == size_embed, embeddings.shape[0]
    return embeddings

def convert_label(label_dct_val):
    """convert label dict value to label, e.g. {'layer-1': 3, 'layer-2': 8, 'layer-3': 0} to '380'"""
    l1 = label_dct_val['layer-1']
    l2 = label_dct_val['layer-2']
    l3 = label_dct_val['layer-3']
    label = str(l1) + str(l2) + str(l3)
    return label

def predict_label(x, kms):
    km1, km2, km3 = kms
    x = np.expand_dims(x, axis=0)
    label1 = km1.predict(x)[0]
    label2 = km2[label1].predict(x)[0]
    label3 = km3[label1][label2].predict(x)[0]
    label_dct = {'layer-1':label1, 'layer-2':label2, 'layer-3':label3}
    return label_dct      

def parse_args():
    usage = 'usage: assign speaker label'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--data-dir', type=str, help='data dir')
    parser.add_argument('--label-file', type=str, help='label file')
    parser.add_argument('--manifest-file', type=str, help='manifest file, where the speaker label shall be updated')
    parser.add_argument('--speaker-embed-model', type=str, help='speaker embedding model')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # interactive mode
    args = argparse.ArgumentParser()
    cat = 'train'
    sample_size = 20000
    args.data_dir = os.path.join(work_dir, 'Datasets', 'GigaSpeech-Zhenhao')
    args.cluster_file = os.path.join(args.data_dir, f'clusters_{sample_size}.pkl')
    args.label_file = os.path.join(args.data_dir, f'labels_train_{sample_size}.pkl')
    args.manifest_file = os.path.join(work_dir, 'Data', 'GigaSpeech', f'{cat}_list_10p_nodur.txt')
    args.speaker_embed_model = "speechbrain/spkrec-ecapa-voxceleb"

    # check file existence
    assert os.path.isfile(args.label_file), f'label file {args.label_file} does not exist!'

    # localize arguments
    data_dir = args.data_dir
    cluster_file = args.cluster_file
    label_file = args.label_file
    manifest_file = args.manifest_file
    speaker_embed_model = args.speaker_embed_model

    # print arguments
    print(f'data dir: {data_dir}')
    print(f'cluster file: {cluster_file}')
    print(f'label file: {label_file}')
    print(f'manifest file: {manifest_file}')
    print(f'speaker embedding model: {speaker_embed_model}')

    # load speaker embedding (xvector) model 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source=speaker_embed_model, run_opts={"device": device}, savedir=os.path.join('/tmp', speaker_embed_model))
    size_embed = spk_model[speaker_embed_model]
    print(f'speaker model: {speaker_embed_model} with dimension {size_embed} was selected')

    # load the clusters
    with open(cluster_file, 'rb') as f:
        kms, label_dct = pickle.load(f)

    # read the speaker label files (contains speaker label for all training data of GigaSpeech)
    with open(label_file, 'rb') as f:
        label_dct = pickle.load(f)

    # get fids with label from the label dict    
    fids_with_label = list(label_dct.keys())

    # read the manifest file
    fids = get_fid(manifest_file)
    nfids = len(fids)
    print(f'# of fids: {nfids}')

    # get fid2label dict
    # about 40 min for train which are already enrolled
    # 44 sec for val which require speaker embedding extraction
    fid2label = {}
    batch_size = 10000
    for i, fid in tqdm(enumerate(fids)):
        if i % batch_size == 0:
            print(f'getting labels for fid index: {i} ~ {min(i+batch_size, nfids)} (total {nfids})...')
        if fid in fids_with_label:
            label_dct_val = label_dct[fid]
        else:
            audiofile = os.path.join(data_dir, fid)
            y, sr = librosa.load(audiofile, sr=sample_rate)
            signal = torch.from_numpy(y).unsqueeze(0)
            x = signal2embed(signal, classifier, size_embed)
            label_dct_val = predict_label(x, kms)
        label = convert_label(label_dct_val)
        fid2label[fid] = label

    # update the manifest file
    manifest_file_backup = manifest_file.replace('.txt', '.backup.txt') # backup the original manifest file
    shutil.copyfile(manifest_file, manifest_file_backup)

    # replace the original dummy speaker ids
    lines = open(manifest_file, 'r').readlines()
    lines2 = ['' for _ in range(nfids)]
    for i in range(nfids):
        parts = lines[i].rstrip().split('|')
        fid = parts[0]
        parts[-1] = fid2label[fid]
        lines2[i] = '|'.join(parts)

    # write new manifest file with real speaker ids
    open(manifest_file, 'w').writelines('\n'.join(lines2) + '\n')
    print(f'wrote the updated manifest file: {manifest_file}')
