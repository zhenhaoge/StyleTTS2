#!/bin/bash
#
# generate speech segments using the word-wise accumulation method on top of StyleTTS2 on single sample
#
# Zhenhao Ge, 2024-10-23

ROOT_DIR=/home/users/zge/code/repo/style-tts2

CURRENT_DIR=$PWD
[[ $CURRENT_DIR != $ROOT_DIR ]] && cd $ROOT_DIR \
  && echo "change current dir to: $ROOT_DIR"

# set paths
config_path=$ROOT_DIR/Configs/config_libritts.yml
model_path=$ROOT_DIR/Models/LibriTTS/epochs_2nd_00020.pth
output_path=$ROOT_DIR/Outputs/Scratch/LibriTTS
data_path=$ROOT_DIR/Datasets/GigaSpeech-Zhenhao

# set GPU device
device=cuda:0

# # set reference and experiment id (exp2)
# cat=youtube
# pid=P0000
# aid=YOU1000000038
# sid=${aid}_S0000079
# ref_wav_rel_path="segment/${cat}/${pid}/${aid}/${sid}.wav"
# exp_id=2

# # set reference and experiment id (exp3-5)
# cat=podcast
# pid=P0000
# aid=POD1000000018
# sid=${aid}_S0000158
# ref_wav_rel_path="segment/${cat}/${pid}/${aid}/${sid}.wav"
# seed=1
# exp_id=4

# # set reference and experiment id (exp6-8)
# cat=youtube
# pid=P0000
# aid=YOU1000000044
# sid=${aid}_S0001442
# ref_wav_rel_path="segment/${cat}/${pid}/${aid}/${sid}.wav"
# seed=1
# exp_id=7

# # set reference and experiment id (exp9-11)
# cat=podcast
# pid=P0000
# aid=POD1000000010
# sid=${aid}_S0000023
# ref_wav_rel_path="segment/${cat}/${pid}/${aid}/${sid}.wav"
# seed=1
# exp_id=11

# check the reference sentence
ref_txt_rel_path=${ref_wav_rel_path//.wav/.txt}
ref_txt_path=$data_path/$ref_txt_rel_path
[ ! -f $ref_txt_path ] && echo "${ref_txt_path} does not exist, please check path!" && exit 0
echo "reference text for ${sid}:"
cat $ref_txt_path

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
nwords_future=2
crossfade_dur=5
# tolerance_dur=$(echo "$crossfade_dur / 2" | bc -l)
tolerance_dur=$(awk "BEGIN {print $crossfade_dur / 2}")
echo "concatenate speech with nwords_future:${nwords_future}, crossfade_dur:${crossfade_dur} and tolerance_dur:${tolerance_dur} ..."

python $ROOT_DIR/Scratch/concat_speech.py \
    --output-path $output_path \
    --exp-id $exp_id \
    --ref-id $sid \
    --nwords-future $nwords_future \
    --crossfade-dur $crossfade_dur \
    --tolerance-dur $tolerance_dur

# step 3a: with crossfade duration 10ms instead of 5ms
crossfade_dur=10
tolerance_dur=$(awk "BEGIN {print $crossfade_dur / 2}")
echo "concatenate speech with nwords_future:${nwords_future}, crossfade_dur:${crossfade_dur} and tolerance_dur:${tolerance_dur} ..."

python $ROOT_DIR/Scratch/concat_speech.py \
    --output-path $output_path \
    --exp-id $exp_id \
    --ref-id $sid \
    --nwords-future $nwords_future \
    --crossfade-dur $crossfade_dur \
    --tolerance-dur $tolerance_dur
