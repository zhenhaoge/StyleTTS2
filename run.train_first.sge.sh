#!/bin/bash
#
# example:
# /home/users/hkwon/bin/run_gpu_job.sh run.train_first.sge.sh titan16 Logs/test.log 2 "stylett2-first"
# 
# Zhenhao Ge, 2024-05-30

# # the following screen command does not help (just monitor via log output)
# screen -r 2690627.1 # @ titan16

export https_proxy=http://10.16.0.132:8000
export http_proxy=http://10.16.0.132:8000

source /home/users/zge/.zshrc
conda activate style

cd /home/users/zge/code/repo/style-tts2

# # train LJSpeech first-stage (titan13, pass)
# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --main_process_port 29501 train_first.py \
#     --config_path ./Configs/config_ljspeech_first.yml
# # accelerate launch train_first.py --config_path ./Configs/config_ljspeech_first.yml

# train GigaSpeech first-stage (titan16)
CUDA_VISIBLE_DEVICE=0,1,2,3 accelerate launch --main_process_port 29501 train_first.py \
    --config_path ./Configs/config_gigaspeech_10p_first.yml

# train GigaSpeech first-stage (titan12)
CUDA_VISIBLE_DEVICE=0,1,2,3 accelerate launch --main_process_port 29501 train_first.py \
    --config_path ./Configs/config_gigaspeech_10p_first.yml    

# train LJSpeech first-stage (titan16, pass)
# CUDA_VISIBLE_DEVICES=2,3,4,5 accelerate launch --main_process_port 29501 train_first.py \
#     --config_path ./Configs/config_ljspeech_first.yml