#!/bin/bash
#
# Zhenhao Ge, 2024-06-07

WORK_DIR=$HOME/code/repo/style-tts2

dataset=LJSpeech

# log_folder=20240530.stage-1.titan13
log_folder=20240603.stage-2.titan13

log_path=${WORK_DIR}/Models/${dataset}/tensorboard/${log_folder}
output_path=${WORK_DIR}/Outputs/Training/${dataset}/${log_folder}

[ -d $log_path ] && echo "log dir: $log_path"

python ${WORK_DIR}/Scripts/extract_tb_data.py \
    --log-path $log_path \
    --output-path $output_path