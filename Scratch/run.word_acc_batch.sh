#!/bin/bash
#
# Zhenhao Ge, 2024-10-29

ROOT_DIR=/home/users/zge/code/repo/style-tts2

CURRENT_DIR=$PWD
[[ $CURRENT_DIR != $ROOT_DIR ]] && cd $ROOT_DIR \
  && echo "change current dir to: $ROOT_DIR"

# set general arguments
data_path=$ROOT_DIR/Scratch/gigaspeech_samples
device="cuda:0"
seed=0
nwords_future=2
crossfade_dur=10

# set model specific arguments (libritts)
config_path=$ROOT_DIR/Configs/config_libritts.yml
model_path=$ROOT_DIR/Models/LibriTTS/epochs_2nd_00020.pth
output_path=$ROOT_DIR/Outputs/Scratch/LibriTTS

echo "generating concatenated speech segments using LibriTTS model ..."
bash $ROOT_DIR/Scratch/word_acc_batch.sh \
    $config_path \
    $model_path \
    $output_path \
    $data_path \
    $device \
    $seed \
    $nwords_future \
    $crossfade_dur

# set model specific arguments (gigaspeech-10p-singlespk)
config_path=$ROOT_DIR/Configs/config_gigaspeech_10p_singlespk_second.yml
model_path=$ROOT_DIR/Models/GigaSpeech/10p_singlespk/epoch_2nd_00041.pth
output_path=$ROOT_DIR/Outputs/Scratch/GigaSpeech_10p_singlespk

echo "generating concatenated speech segments using GigaSpeech-10p-singlespk model ..."
bash $ROOT_DIR/Scratch/word_acc_batch.sh \
    $config_path \
    $model_path \
    $output_path \
    $data_path \
    $device \
    $seed \
    $nwords_future \
    $crossfade_dur