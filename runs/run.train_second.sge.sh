#!/bin/bash
#
# example:
# /home/users/hkwon/bin/run_gpu_job.sh run.train_second.sge.sh titan13 Logs/style-tts2_train_second.log 2 "stylett2-second"
# 
# Zhenhao Ge, 2024-05-30

export https_proxy=http://10.16.0.132:8000
export http_proxy=http://10.16.0.132:8000

source /home/users/zge/.zshrc
conda activate style

cd /home/users/zge/code/repo/style-tts2

CUDA_VISIBLE_DEVICES=6,7 python train_second.py --config_path ./Configs/config_ljspeech_second.yml