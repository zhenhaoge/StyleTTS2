#!/bin/bash
#
# run spleeter for source separation
#
# Zhenhao Ge, 2024-08-27

SPLEETER_DIR=${1:-/home/users/zge/code/repo/spleeter}
AUDIO_FILE_ORI=$2
OUT_DIR=$3

recording_id=$(basename $AUDIO_FILE_ORI)
recording_id=${recording_id%.*}

# get the duration of the audio file (format: hh:mm:ss.ss and sec)
audio_dur_ori=$(soxi -d $AUDIO_FILE_ORI)
hh=$(echo $audio_dur_ori | cut -d ":" -f 1)
mm=$(echo $audio_dur_ori | cut -d ":" -f 2)
ss=$(echo $audio_dur_ori | cut -d ":" -f 3)
audio_dur_ori_sec=$(bc -l <<< "$hh * 3600 + $mm * 60 + $ss")
echo "audio duration: $audio_dur_ori ($audio_dur_ori_sec seconds)"

# get the duration ceiling value used in source seperation
tmp=${audio_dur_ori_sec/.*}
dur_lim_sec=$((tmp+1))
echo "set duration limit for spleeter to ${dur_lim_sec} seconds"

# separate souce using spleeter
export MODEL_PATH=$SPLEETER_DIR/pretrained_models
spleeter separate -d $dur_lim_sec -p spleeter:2stems -o $OUT_DIR $AUDIO_FILE_ORI

# move the separated audio files from their default dir to the specified output dir
AUDIO_FILE_ACC=$OUT_DIR/${recording_id}_accompaniment.wav
AUDIO_FILE_VOC=$OUT_DIR/${recording_id}_vocals.wav
mv "$OUT_DIR/${recording_id}/accompaniment.wav" $AUDIO_FILE_ACC
mv "$OUT_DIR/${recording_id}/vocals.wav" $AUDIO_FILE_VOC
echo "separated audio into accompaniment ($AUDIO_FILE_ACC) and vocals ($AUDIO_FILE_VOC)"

# remove the default output dir
rm -rf $OUT_DIR/${recording_id}