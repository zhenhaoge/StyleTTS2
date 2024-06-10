#!/bin/bash
#
# Zhenhao Ge, 2024-05-16

ROOT_DIR=/home/users/zge/code/repo/style-tts2

CURRENT_DIR=$PWD
[[ $CURRENT_DIR != $ROOT_DIR ]] && cd $ROOT_DIR \
  && echo "change current dir to: $ROOT_DIR"

# general setup
model_path=$ROOT_DIR/Models/LibriTTS
model_name=epochs_2nd_00020.pth
output_path=$ROOT_DIR/Outputs/RTF
manifest_file=$output_path/manifest.txt

# set device
# if device=cpu (export OMP_NUM_THREADS=1 to use single CPU)
device="cpu" # options: cpu, cuda, or cuda:x

# tuning parameters
diffusion_steps=5
embedding_scale=1
alpha=0.3
beta=0.7

# parameters for the rtf measurement
run_id=exp3
num_reps=10
num_warmup=3 

python $ROOT_DIR/infer.rtf.py \
    --model-path $model_path \
    --model-name $model_name \
    --output-path $output_path \
    --manifest-file $manifest_file \
    --device $device \
    --diffusion-steps $diffusion_steps \
    --embedding-scale $embedding_scale \
    --alpha $alpha \
    --beta $beta \
    --run-id $run_id \
    --num-reps $num_reps \
    --num-warmup $num_warmup

