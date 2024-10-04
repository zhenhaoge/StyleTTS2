#!/bin/bash
#
# Zhenhao Ge, 2024-05-13

ROOT_DIR=/home/users/zge/code/repo/style-tts2

CURRENT_DIR=$PWD
[[ $CURRENT_DIR != $ROOT_DIR ]] && cd $ROOT_DIR \
  && echo "change current dir to: $ROOT_DIR"

dataset=LJSpeech
config_path=$ROOT_DIR/Models/$dataset/config.yml
model_path=$ROOT_DIR/Models/$dataset/epoch_2nd_00100.pth
output_path=$ROOT_DIR/Outputs/Demo/$dataset
device='cuda:1'

mkdir -p $output_path

echo "config path: ${config_path}"
echo "model path: ${model_path}"
echo "output path: ${output_path}"
echo "device: ${device}"

# run LJSpeech demo
python $ROOT_DIR/Demo/Inference_LJSpeech.py \
    --config-path $config_path \
    --model-path $model_path \
    --output-path $output_path \
    --device $device
