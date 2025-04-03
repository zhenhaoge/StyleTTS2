#!/bin/bash
#
# generate speech segments using the word-wise accumulation method on top of StyleTTS2 on multiple samples (batch version)
#
# Zhenhao Ge, 2024-10-29

ROOT_DIR=/home/users/zge/code/repo/style-tts2

CURRENT_DIR=$PWD
[[ $CURRENT_DIR != $ROOT_DIR ]] && cd $ROOT_DIR \
  && echo "change current dir to: $ROOT_DIR"

# set model and config paths

# libritts config and model paths
config_path=${1:-$ROOT_DIR/Configs/config_libritts.yml}
model_path=${2:-$ROOT_DIR/Models/LibriTTS/epochs_2nd_00020.pth}

# # gigaspeech (10%) single-speaker config and model paths
# config_path=$ROOT_DIR/Configs/config_gigaspeech_10p_singlespk_second.yml
# model_path=$ROOT_DIR/Models/GigaSpeech/10p_singlespk/epoch_2nd_00041.pth

# set output path
output_path=${3:-$ROOT_DIR/Outputs/Scratch/LibriTTS}
# output_path=${3:-$ROOT_DIR/Outputs/Scratch/GigaSpeech_10p_singlespk}

# data_path=${4:-$ROOT_DIR/Datasets/GigaSpeech-Zhenhao}
data_path=${4:-$ROOT_DIR/Scratch/gigaspeech_samples}

# set other arguments
device=${5:-cuda:0}
seed=${6:-0}
nwords_future=${7:-2}
crossfade_dur=${8:-10}
tolerance_dur=$(awk "BEGIN {print $crossfade_dur / 2}")

# Check if the directory exists
if [[ ! -d "$data_path" ]]; then
    echo "Error: Directory '$data_path' does not exist."
    exit 1
fi

# print out arguments
echo "config path: ${config_path}"
echo "model_path: ${model_path}"
echo "output_path: ${output_path}"
echo "data_path: ${data_path}"
echo "device: ${device}"
echo "seed: ${seed}"
echo "nwords_future: ${nwords_future}"
echo "crossfade_dur: ${crossfade_dur}"
echo "tolerance_dur: ${tolerance_dur}"

# get reference wav paths
ref_wav_paths=($data_path/*.wav)

# loop over to generate the word-accumulated speech segments
for ref_wav_path in "${ref_wav_paths[@]}"; do

  # # for testing
  # ref_wav_path=${ref_wav_paths[0]}

  # get the reference wav and text paths
  ref_wav_rel_path=$(basename $ref_wav_path)
  # echo $ref_wav_rel_path
  ref_txt_path=${ref_wav_path//.wav/.txt}
  ref_txt_rel_path=$(basename $ref_txt_path)
  echo "reference text for ${ref_txt_rel_path}:"
  cat $ref_txt_path

  # get the experiment id
  sid=${ref_wav_rel_path%.*}
  exp_id=${sid}_${seed}
  
  # step 1: generate TTS speech
  echo "step 1: generating TTS speech segments for exp id: ${exp_id} ..."
  python $ROOT_DIR/Scratch/gen_speech.py \
    --config-path $config_path \
    --model-path $model_path \
    --output-path $output_path \
    --data-path $data_path \
    --ref-wav-rel-path $ref_wav_rel_path \
    --seed $seed \
    --exp-id $exp_id \
    --device $device

  # step 2: align TTS speech
  echo "step 2: alignning TTS speech segment for exp id: ${exp_id} ..."
  python $ROOT_DIR/Scratch/align_speech.py \
    --output-path $output_path \
    --exp-id $exp_id \
    --ref-id $sid

  # step 3: concatenate extracted word speech segments
  echo "step 3: concatentating extracted word speech segments for exp id: ${exp_id} ..."
  python $ROOT_DIR/Scratch/concat_speech.py \
    --output-path $output_path \
    --exp-id $exp_id \
    --ref-id $sid \
    --nwords-future $nwords_future \
    --crossfade-dur $crossfade_dur \
    --tolerance-dur $tolerance_dur

done