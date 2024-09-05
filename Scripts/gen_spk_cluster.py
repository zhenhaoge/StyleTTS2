# generate 3-layers of speaker clusters
# later the audio segment will be assigned speaker id based on the speaker clustering results
#
# Zhenhao Ge, 2024-07-31

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
import random
from sklearn.cluster import MiniBatchKMeans, KMeans
import pickle

# set dirs
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

# set the samlpe rate of the data used to train the speaker embedding model
sample_rate = 16000

# set number of clusters in each layer of clustering
# (10 layer-1 clusters, 10X10 layer-2 clusters, 10X10X10 layer-3 clusters)
num_clusters = [10, 10, 10]

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

def compute_score_from_two_files(audiofile1, audiofile2, sr):
    y1, _ = librosa.load(audiofile1, sr=sample_rate)
    y2, _ = librosa.load(audiofile2, sr=sample_rate)
    signal1 = torch.from_numpy(y1).unsqueeze(0)
    signal2 = torch.from_numpy(y2).unsqueeze(0)
    embedding1 = signal2embed(signal1, classifier, size_embed)
    embedding2 = signal2embed(signal2, classifier, size_embed)
    score = cos_sim(embedding1, embedding2)
    return score

def run_kmeans(X, K=10, num_runs=10, method='regular', verbose=False):
    best_intertia = float('inf')     
    for i in range(num_runs):
        if method == 'regular':
            # clustering with regular KMeans (slower, but touch the fit limit, yeild very close intertia every run)                        
            kmeans = KMeans(n_clusters=K,
                            random_state=i,
                            init='k-means++').fit(X)
        elif method == 'mini-batch':                    
            # clustering with MiniBatch KMeans (much faster, slightly less fit)
            kmeans = MiniBatchKMeans(n_clusters=K,
                                    random_state=i,
                                    batch_size=1024,
                                    init='k-means++').fit(X)
        else:
            raise Exception('method should be either regular or mini-batch!')                                        
        if verbose:
            print(f'run {method} {i}/{num_runs}: intertia: {kmeans.inertia_}, #iters: {kmeans.n_iter_}')
        if kmeans.inertia_ < best_intertia:
            best_intertia = kmeans.inertia_
            km = kmeans
    return km

def get_sample_idx(km):
    """get sample indeces per cluster"""

    K = km.cluster_centers_.shape[0]
    km_idxs = [[] for _ in range(K)]
    for i in range(K):
        km_idxs[i] = [idx for idx, l in enumerate(km.labels_) if l==i]
    return km_idxs

def get_score_to_centroids(X, km):
    """get score from sample to centroids"""

    # get sample size and # of clusters
    sample_size = X.shape[0]
    K = km.cluster_centers_.shape[0]

    # initialize scores (size: sample_size X K)
    scores = np.zeros((sample_size, K))

    # initalize positive and negative score (size: sample_size)
    # score_pos: the score between sample and its beloning centroid
    # score_neg: the avg. score between sample and other centroids
    score_pos = np.zeros(sample_size)
    score_neg = np.zeros(sample_size)
    for i in range(sample_size):
        for j in range(K):
            scores[i][j] = cos_sim(X[i], km.cluster_centers_[j])
        score_pos[i] = scores[i][km.labels_[i]]
        score_neg[i] = np.mean([s for k, s in enumerate(scores[i]) if k != km.labels_[i]])
    return scores, (score_pos, score_neg)

def assign_label(fids, label_dct, km, layer):
    """assign label at specified layer for the cluster"""

    nfids = len(fids)
    nlabels = len(km.labels_)
    assert nfids == nlabels, '#fids should be equal to #labels!'

    for i, fid in enumerate(fids):
        if fid not in label_dct:
            label_dct[fid] = {}
        label_dct[fid][f'layer-{layer}'] = km.labels_[i]

    return label_dct

def split_data(X, fids, km):
    """split data and fids into K groups based on cluster with K centroids"""
    K = km.cluster_centers_.shape[0]
    X_lst = [list() for _ in range(K)]
    fids_lst = [list() for _ in range(K)]
    km_idxs = get_sample_idx(km)
    for i in range(K):
        for idx in km_idxs[i]:
            X_lst[i].append(X[idx])
            fids_lst[i].append(fids[idx])
        X_lst[i] = np.array(X_lst[i])
    return X_lst, fids_lst

def verify_label(x, fid, kms, label_dct):
    km1, km2, km3 = kms
    x = np.expand_dims(x, axis=0)
    label1 = km1.predict(x)[0]
    label2 = km2[label1].predict(x)[0]
    label3 = km3[label1][label2].predict(x)[0]
    assert label1 == label_dct[fid]['layer-1'], \
        'layer-1 cluster should be {}, but {}!'.format(label_dct[fid]['layer-1'], label1)
    assert label2 == label_dct[fid]['layer-2'], \
        'layer-2 cluster should be {}, but {}!'.format(label_dct[fid]['layer-2'], label2)
    assert label3 == label_dct[fid]['layer-3'], \
        'layer-3 cluster should be {}, but {}!'.format(label_dct[fid]['layer-3'], label3)
    return True

def predict_label(x, fid, kms, label_dct):
    km1, km2, km3 = kms
    x = np.expand_dims(x, axis=0)
    label1 = km1.predict(x)[0]
    label2 = km2[label1].predict(x)[0]
    label3 = km3[label1][label2].predict(x)[0]
    label_dct[fid] = {'layer-1':label1, 'layer-2':label2, 'layer-3':label3}
    return label_dct               

def parse_args():
    usage = 'usage: generate speaker clusters'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--data-dir', type=str, help='data dir')
    parser.add_argument('--manifest-file', type=str, help='manifest file')
    parser.add_argument('--speaker-embed-model', type=str, help='speaker embedding model')
    parser.add_argument('--cluster-file', type=str, help='cluster output result file')
    parser.add_argument('--sample-size', type=int, help='number of audio files selected')
    parser.add_argument('--seed', type=int, help='seed to randomly select audiofiles for clustering')
    parser.add_argument('--num-runs', type=int, help='#runs for the same clustering')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # interactive mode
    args = argparse.ArgumentParser()
    args.data_dir = os.path.join(work_dir, 'Datasets', 'GigaSpeech-Zhenhao')
    args.manifest_file = os.path.join(work_dir, 'Data', 'GigaSpeech', 'train_list.txt')
    # args.speaker_embed_model = "speechbrain/spkrec-xvect-voxceleb"
    args.speaker_embed_model = "speechbrain/spkrec-ecapa-voxceleb"
    args.sample_size = 20000
    args.seed = 1234
    args.num_runs = 5
    args.cluster_file = os.path.join(args.data_dir, f'clusters_{args.sample_size}.pkl')

    # localize arguments
    data_dir = args.data_dir
    manifest_file = args.manifest_file
    speaker_embed_model = args.speaker_embed_model
    cluster_file = args.cluster_file
    sample_size = args.sample_size
    seed = args.seed
    num_runs = args.num_runs

    # check file existence
    assert os.path.isfile(manifest_file), f'manifest file: {manifest_file} does not exist!'

    print(f'data dir: {data_dir}')
    print(f'manifest file: {manifest_file}')
    print(f'speaker embedding model: {speaker_embed_model}')
    print(f'cluster output result file: {cluster_file}')
    print(f'sample size: {sample_size}')
    print(f'seed: {seed}')
    print(f'num of runs: {num_runs}')

    # load speaker embedding (xvector) model 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source=speaker_embed_model, run_opts={"device": device}, savedir=os.path.join('/tmp', speaker_embed_model))
    size_embed = spk_model[speaker_embed_model]
    print(f'speaker model: {speaker_embed_model} with dimension {size_embed} was selected')

    # get audiofiles
    lines = open(manifest_file, 'r').readlines()
    audiofiles = [os.path.join(data_dir, line.split('|')[0]) for line in lines]
    num_audiofiles = len(audiofiles)
    print(f'# of audio files total: {num_audiofiles}')
    
    # randomize audiofiles with seed
    random.Random(seed).shuffle(audiofiles)
    # print('\n'.join([os.path.basename(f) for f in audiofiles[:3]]))

    # select audiofiles for clustering
    audiofiles_sel = audiofiles[:sample_size]
    print(f'{sample_size} audio files are selected')

    # get the selected fids based on selected audiofiles (relative file paths) 
    fids_sel = [audiofile.replace(data_dir, '').strip(os.sep) for audiofile in audiofiles_sel]

    # extract speaker embedding from the selected audio files
    embedding_lst = ['' for _ in range(sample_size)]
    batch_size = 1000
    for i in range(sample_size):
        if i % batch_size == 0:
            print(f'processing audio file [{i}, {min(i+batch_size, sample_size)}), {sample_size} total ...')
        audiofile = audiofiles_sel[i]
        y, sr = librosa.load(audiofile, sr=sample_rate)
        signal = torch.from_numpy(y).unsqueeze(0)
        embedding_lst[i] = signal2embed(signal, classifier, size_embed)

    #%% layer-1 clustering
    print('layer-1 clustering ...')

    # convert embedding_list (list of np array) to 2D np array
    X1 = np.array([list(emb) for emb in embedding_lst])  

    # fids used in layer-1 clustering
    fids1 = fids_sel

    # run layer-1 clustering
    km1 = run_kmeans(X1, K=num_clusters[0], num_runs=num_runs, method='mini-batch', verbose=True)
    # km1 = run_kmeans(X1, K=num_clusters[0], num_runs=10, method='regular', verbose=True)

    # sanity check 1: get data indeces per cluster (layer-1 clustering, 10 clusters)
    km1_idxs = get_sample_idx(km1)
    for i in range(num_clusters[0]):
        print(f'# of samples in layer-1 cluster {i}: {len(km1_idxs[i])}')

    # sanity check 2: get scores to centroids (layer-1 clsuter)
    scores, (score_pos, score_neg) = get_score_to_centroids(X1, km1)
    print(f'mean pos score: {np.mean(score_pos):.4f}, mean neg score: {np.mean(score_neg):.4f}')

    # assign labels at layer-1
    label_dct = {}
    label_dct = assign_label(fids_sel, label_dct, km1, layer=1)

    #%% layer-2 clustering
    print('layer-2 clustering ...')

    # split data based on km1 (X1 -> X2)
    X2, fids2 = split_data(X1, fids1, km1)

    # run layer-2 clustering
    km2 = [list() for _ in range(num_clusters[0])]
    for i in range(num_clusters[0]):
        print(f'running layer-2 clustering in group {i} ...')
        km2[i] = run_kmeans(X2[i], K=num_clusters[1], num_runs=num_runs, method='mini-batch', verbose=True)
        # km2[i] = run_kmeans(X2[i], K=num_clusters[1], num_runs=10, method='regular', verbose=True)

    # sanity check 1: get data indeces per cluster (layer-2 clustering, 10 X 10 groups, 10 clusters per group)
    km2_idxs = [list() for _ in range(num_clusters[0])]
    for i in range(num_clusters[0]):
        km2_idxs[i] = get_sample_idx(km2[i])
        for j in range(num_clusters[1]):
            print(f'# of samples in layer-2 cluster ({i}, {j}): {len(km2_idxs[i][j])}')

    # sanity check 2: get scores to centroids (layer-2 clsutering)
    for i in range(num_clusters[0]):
        scores, (score_pos, score_neg) = get_score_to_centroids(X2[i], km2[i])
        print(f'(group {i}), mean pos score: {np.mean(score_pos):.4f}, mean neg score: {np.mean(score_neg):.4f}')

    # assign labels at layer-2
    for i in range(num_clusters[0]):
        label_dct = assign_label(fids2[i], label_dct, km2[i], layer=2)

    #%% layer-3 clustering
    print('layer-3 clustering ...')

    # split data based on km2 (X2 -> X3)
    X3 = [list() for _ in range(num_clusters[0])]
    fids3 = [list() for _ in range(num_clusters[0])]
    for i in range(num_clusters[0]):
        X3[i], fids3[i] = split_data(X2[i], fids2[i], km2[i])

    # run layer-3 clustering
    km3 = [[list() for _ in range(num_clusters[1])] for _ in range(num_clusters[0])]
    for i in range(num_clusters[0]):
        for j in range(num_clusters[1]):
            print(f'running layer-3 clustering in group ({i}, {j}) ...')
            K = min(num_clusters[2], len(X3[i][j])) # this avoid clustering with #samples < #clusters
            km3[i][j] = run_kmeans(X3[i][j], K=K, num_runs=num_runs, method='regular', verbose=True)  

    # sanity check 1: get data indeces per cluster (layer-3 clusters, 10 X 10 X 10 groups, 10 clusters per group)
    km3_idxs = [[list() for _ in range(num_clusters[1])] for _ in range(num_clusters[0])]
    for i in range(num_clusters[0]):
        for j in range(num_clusters[1]):
            km3_idxs[i][j] = get_sample_idx(km3[i][j])
            K = min(num_clusters[2], len(X3[i][j]))
            for k in range(K):
                print(f'# of samples in layer-3 cluster ({i}, {j}, {k}): {len(km3_idxs[i][j][k])}')

    # sanity check 2: get scores to centroids (layer-3 clsutering)
    for i in range(num_clusters[0]):
        for j in range(num_clusters[1]):
            scores, (score_pos, score_neg) = get_score_to_centroids(X3[i][j], km3[i][j])
            print(f'group ({i}, {j}), mean pos score: {np.mean(score_pos):.4f}, mean neg score: {np.mean(score_neg):.4f}')

    # assign labels at layer-3
    for i in range(num_clusters[0]):
        for j in range(num_clusters[1]):
            label_dct = assign_label(fids3[i][j], label_dct, km3[i][j], layer=3)         
    
    #%% cluster saving and verification

    # save clusters and enrolled speaker labels to pkl file
    kms = (km1, km2, km3)
    with open(cluster_file, 'wb') as f:
        pickle.dump([kms, label_dct], f)

    # # load clusters and enrolled speaker labels from saved pkl file
    # with open(cluster_file, 'rb') as f:
    #     kms, label_dct = pickle.load(f)

    # verfiy if every sample is predicted in the same cluster as it is labeled
    status = [0 for _ in range(sample_size)]
    for i in range(sample_size):
        x, fid = X1[i], fids1[i]
        status[i] = verify_label(x, fid, kms, label_dct)
    assert all(status), 'not all samples are the same!'

    # check inner and inter score with one example
    # inner score (within the cluster (1,2,3))
    # inter score (between the cluster (1,2,3) and the cluster (3,2,1))

    # get fids in the cluster (1,2,3)
    i0, j0, k0 = 1, 2, 3
    labels = km3[i0][j0].labels_
    fids_c1 = [fid for k, fid in enumerate(fids3[i0][j0]) if labels[k] == k0]
    nfids_c1 = len(fids_c1)
    print(f'#fids in cluster ({i0},{j0},{k0}): {nfids_c1}')

    # get fids in the cluster (3,2,1)
    i0, j0, k0 = 3, 2, 1
    labels = km3[i0][j0].labels_
    fids_c2 = [fid for k, fid in enumerate(fids3[i0][j0]) if labels[k] == k0]
    nfids_c2 = len(fids_c2)
    print(f'#fids in cluster ({i0},{j0},{k0}): {nfids_c2}')

    # compute the inner score within the cluster (1,2,3)
    scores_inner = np.zeros((nfids_c1, nfids_c1))
    for i, fid in enumerate(fids_c1):
        audiofile1 = os.path.join(data_dir, fid)
        print(audiofile1)
        for j, fid in enumerate(fids_c1):
            audiofile2 = os.path.join(data_dir, fid)
            scores_inner[i][j] = compute_score_from_two_files(audiofile1, audiofile2, sr=sample_rate)
    # show mean inner score inside cluster (1,2,3)
    # ex1: 0.9573 (10000 samples, 512 emb-dim, km1: mini-batch, km2,km3: regular)
    # ex2: 0.7648 (20000 samples, 192 emb-dim, km1,km2: mini-batch, km3: regular) 
    print(f'mean inner score: {np.mean(scores_inner):.4f}')

    # compute the inter score between the cluster (1,2,3) and the cluster (3,2,1)
    scores_inter = np.zeros((nfids_c1, nfids_c2))
    for i, fid in enumerate(fids_c1):
        audiofile1 = os.path.join(data_dir, fid)
        for j, fid in enumerate(fids_c2):
            audiofile2 = os.path.join(data_dir, fid)
            scores_inter[i][j] = compute_score_from_two_files(audiofile1, audiofile2, sr=sample_rate)
    # show mean inter score between cluster (1,2,3) and cluster (3,2,1)
    # ex1: 0.9157 (10000 samples, 512 emb-dim, km1: mini-batch, km2,km3: regular)
    # ex2: 0.0423 (20000 samples, 192 emb-dim, km1,km2: mini-batch, km3: regular)
    print(f'mean inter score: {np.mean(scores_inter):.4f}')

    #%% assign labels to all data

    # initialize label dict for all audio files (including unselected audio files)
    label_dct_all = label_dct.copy()

    # predict labels for the unselected audio files
    # (TODO: consider paralllelize it to speed up)
    audiofiles_unsel = audiofiles[sample_size:]
    num_unsel = len(audiofiles_unsel)
    for i, audiofile in tqdm(enumerate(audiofiles_unsel)):
        if i % batch_size == 0:
            print(f'predicting labels for unselected audio files [{i}, {min(i+batch_size, num_unsel)}), {num_unsel} total ...')
        fid = audiofile.replace(data_dir, '').strip(os.sep)
        y, sr = librosa.load(audiofile, sr=sample_rate)
        signal = torch.from_numpy(y).unsqueeze(0)
        x = signal2embed(signal, classifier, size_embed)
        label_dct_all = predict_label(x, fid, kms, label_dct_all)

    # save the label file
    cat = os.path.basename(manifest_file).split('_')[0]
    label_file = os.path.join(data_dir, f'labels_{cat}_{sample_size}.pkl')
    with open(label_file, 'wb') as f:
        pickle.dump(label_dct_all, f)
    print(f'saved labels for category:{cat} in {label_file}')

    # # sanity check: reload the saved label file
    # with open(label_file, 'rb') as f:
    #     label_dct_all2 = pickle.load(f)
    # assert label_dct_all == label_dct_all2, 'label file mis-match!'   

