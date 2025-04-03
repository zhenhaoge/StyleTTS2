#!/bin/bash
#
# run the YouTube dubbing pipeline
#
# Zhenhao Ge, 2024-07-28

export https_proxy=http://10.16.0.132:8000
export http_proxy=http://10.16.0.132:8000

# use conda envionment 'style'
source ~/.zshrc
conda activate style

ROOT_DIR=$HOME/code/repo/style-tts2
[ $PWD != $ROOT_DIR ] && cd $ROOT_DIR
echo "current dir: $PWD"
WORK_DIR=$ROOT_DIR/Examples/YouTube/s1

# # example 1
# account_id=laoming
# recording_id=20220212
# dur_id=full

# example 2
account_id=dr-wang
recording_id=20210915
dur_id=full

# set input dir and video file
IN_DIR=$ROOT_DIR/Datasets/YouTube/${account_id}/${recording_id}/${dur_id}
[ -d $IN_DIR ] && echo "input dir: $IN_DIR" \
    || (echo "input dir $IN_DIR does not exist!" && exit 1)
OUT_DIR=$ROOT_DIR/Outputs/YouTube/${account_id}/${recording_id}/${dur_id}  
VIDEO_FILE_ORI=$IN_DIR/${recording_id}_L1.mp4
[ -f $VIDEO_FILE_ORI ] && echo "original video file: $VIDEO_FILE_ORI" \
    || (echo "original video file $VIDEO_FILE_ORI does not exist!" && exit 1)

# get the duration of the video file (format: hh:mm:ss.ss and sec)
video_dur_ori=$(ffmpeg -i $VIDEO_FILE_ORI 2>&1 | grep Duration | awk '{print $2}' | cut -d ',' -f 1)
hh=$(echo $video_dur_ori | cut -d ":" -f 1)
mm=$(echo $video_dur_ori | cut -d ":" -f 2)
ss=$(echo $video_dur_ori | cut -d ":" -f 3)
video_dur_ori_sec=$(bc -l <<< "$hh * 3600 + $mm * 60 + $ss")
tmp=${video_dur_ori_sec/.*}
dur_lim_sec=$((tmp+1)) # get the duration ceiling value used in source seperation
echo "video duration: $video_dur_ori ($video_dur_ori_sec seconds)"

# extract audio in mp3
AUDIO_FILE_ORI="$IN_DIR/$(basename $VIDEO_FILE_ORI .mp4).mp3"
# nproc=$(grep -c processor /proc/cpuinfo) # get #threads in ubuntu
ffmpeg -i $VIDEO_FILE_ORI -vn $AUDIO_FILE_ORI # extract audio
# ffmpeg -i $AUDIO_FILE_ORI # used to check audio file info
echo "extracted audio file: $AUDIO_FILE_ORI"

# extract silence video
VIDEO_FILE_SIL="$IN_DIR/${recording_id}_L1_silent.mp4"
ffmpeg -i $VIDEO_FILE_ORI -c:v copy -an $VIDEO_FILE_SIL
echo "extracted silent video file: $VIDEO_FILE_SIL"

# split audio into accompaniment (background) and vocal (audio source separation)
SPLEETER_DIR=$HOME/code/repo/spleeter
cd $SPLEETER_DIR # switch to the spleeter dir to follow its default model path (pretrained_models)
spleeter separate -d $dur_lim_sec -p spleeter:2stems -o $IN_DIR $AUDIO_FILE_ORI
basename=$(basename $AUDIO_FILE_ORI .mp3)
AUDIO_FILE_ACC=$IN_DIR/${basename}_accompaniment.wav
AUDIO_FILE_VOC=$IN_DIR/${basename}_vocals.wav
mv "$IN_DIR/${basename}/accompaniment.wav" $AUDIO_FILE_ACC
mv "$IN_DIR/${basename}/vocals.wav" $AUDIO_FILE_VOC
rm -rf $IN_DIR/${basename}
cd $ROOT_DIR
echo "separated audio into accompaniment ($AUDIO_FILE_ACC) and vocals ($AUDIO_FILE_VOC)"

# sanity check: combine the sepearated files to see if it become the original file
# (AUDIO_FILE_MIX should sound similar to AUDIO_FILE_ORI)
AUDIO_FILE_MIX=$IN_DIR/${recording_id}_L1_accompaniment+vocals.wav
sox -m $IN_DIR/${recording_id}_L1_accompaniment.wav $IN_DIR/${recording_id}_L1_vocals.wav $AUDIO_FILE_MIX

# extract mono channel from the vocal audio file
channel_idx=1 # 1: left-channel, 2: right-channel
AUDIO_FILE_VOC_MONO=$IN_DIR/${recording_id}_L1_vocals_mono.wav
sox $AUDIO_FILE_VOC $AUDIO_FILE_VOC_MONO remix $channel_idx

# # convert mono channel back to stereo channel (two identical channels)
# AUDIO_FILE_VOC_STER=$IN_DIR/${recording_id}_L1_vocals_stereo.wav
# python $WORK_DIR/helpers/mono2stereo.py \
#     --infile $AUDIO_FILE_VOC_MONO \
#     --outfile $AUDIO_FILE_VOC_STER

# step 1: extract trimmed audio segments based on subtitle file with timestamps
trans_file=$IN_DIR/${recording_id}_L1.manual.srt
ext=${trans_file##*.}
if [ $ext == 'srt' ] || [ $ext == 'ass' ] ; then
    echo "${trans_file} is in $ext format"
else
    echo "transcription file should be in either .srt or .ass format!" && exit 1
fi
audio_file=$IN_DIR/${recording_id}_L1_vocals_mono.wav
out_dir=$OUT_DIR/v1.original
meta_dir=$OUT_DIR/meta
dur_lim=0 # 0 means no limit
top_db=20
out_ver=v1
python $WORK_DIR/01_extract_segment.py \
    --trans-file $trans_file \
    --audio-file $audio_file \
    --out-dir $out_dir \
    --meta-dir $meta_dir \
    --dur-lim $dur_lim \
    --top-db $top_db \
    --out-ver $out_ver

# step 2: group adjacent audio segments for better translation later
audio_file=$IN_DIR/${recording_id}_L1_vocals_mono.wav
in_dir=$OUT_DIR/v1.original
out_dir=$OUT_DIR/v2.grouped
meta_dir=$OUT_DIR/meta
out_ts_file=${meta_dir}/${recording_id}_L1_ts-text_v2.csv
out_ver=v2
python $WORK_DIR/02_group_segment.py \
    --audio-file $audio_file \
    --in-dir $in_dir \
    --out-dir $out_dir \
    --meta-dir $meta_dir \
    --out-ts-file $out_ts_file \
    --out-ver $out_ver

# step 3: correct the L1 text (${recording_id}_L1_ts-text_v2.csv -> ${recording_id}_L1_ts-text_v2.corrected.csv)
# (manual correction now, but will add modules to semi-auto it)

# step 4: update the grouped segments after text correction
# (run 02_group_segment.py again to regroup, including segment concatenation)
audio_file=$IN_DIR/${recording_id}_L1_vocals_mono.wav
in_dir=$OUT_DIR/v2.grouped
out_dir=$OUT_DIR/v3.corrected
meta_dir=$OUT_DIR/meta
in_ts_file=${meta_dir}/${recording_id}_L1_ts-text_v2.corrected.csv
out_ts_file=${meta_dir}/${recording_id}_L1_ts-text_v3.csv
out_ver=v3
[ ! -f $in_ts_file ] && echo "text-corrected timestamp file $in_ts_file does not exist, prepare it first!" && exit 1
python $WORK_DIR/02_group_segment.py \
    --audio-file $audio_file \
    --in-dir $in_dir \
    --out-dir $out_dir \
    --meta-dir $meta_dir \
    --in-ts-file $in_ts_file \
    --out-ts-file $out_ts_file \
    --out-ver $out_ver

# step 5.1: get the translated texts (${recording_id}_L1_text_v3.txt -> ${recording_id}_L2_text_v3.manual.txt)
# (manual translation now, will automate it later)

# step 5.2: convert L2 text .txt file to L2 ts-text .csv file (used in synthesized segment generation)
meta_dir=$OUT_DIR/meta
L1_ts_file=$OUT_DIR/meta/${recording_id}_L1_ts-text_v3.csv # input 1: timestamps + texts(L1)
L2_text_file=$OUT_DIR/meta/${recording_id}_L2_text_v3.manual.txt # input 2: texts(L2)
L2_ts_file=$OUT_DIR/meta/${recording_id}_L2_ts-text_v3.manual.csv # output: timestamps + texts(L2)
python $WORK_DIR/helpers/prep_ts_text.py \
    --zh-ts-file ${L1_ts_file} \
    --en-text-file ${L2_text_file} \
    --en-ts-file ${L2_ts_file}

# step 6.1: prepare the master speaker file for speaker style adaptation
in_dir=$OUT_DIR/v1.original
concat_audiofile=$IN_DIR/${recording_id}_L1_spk.wav
speaker_embed_model="speechbrain/spkrec-xvect-voxceleb"
dur_search=600.0
dur_select=30.0
python $WORK_DIR/helpers/get_spk_emb.py \
    --in-dir ${in_dir} \
    --concat-audiofile ${concat_audiofile} \
    --speaker-embed-model ${speaker_embed_model} \
    --dur-search ${dur_search} \
    --dur-select ${dur_select}

# step 6.2: generate the synthesized L2 audio segments
config_file=$ROOT_DIR/Models/LibriTTS/config.yml
model_file=$ROOT_DIR/Models/LibriTTS/epochs_2nd_00020.pth
meta_file=$OUT_DIR/meta/${recording_id}_L2_ts-text_v3.manual.csv # generated in step 5.2
in_dir=$OUT_DIR/v3.corrected
out_dir=$OUT_DIR/v4.translated
use_master_file=true
master_audio_file=$IN_DIR/${recording_id}_L1_spk.wav # generated in step 6.1
device="cuda:0"
python $WORK_DIR/03_gen_segment.py \
    --config-file ${config_file} \
    --model-file ${model_file} \
    --meta-file ${meta_file} \
    --in-dir  ${in_dir} \
    --out-dir ${out_dir} \
    --use-master-style ${use_master_file} \
    --master-audio-file ${master_audio_file} \
    --device ${device}

# step 7: match the L2 segments with the L1 segments via time-scaling and shifting

# switch envionment to accomodate package 'audiostretchy'
# source ~/.zshrc
conda activate espnet

in_dir=$OUT_DIR/v4.translated
out_dir=$OUT_DIR/v6.scaled
ref_dir=$OUT_DIR/v3.corrected
meta_dir=$OUT_DIR/meta
audio_file=$IN_DIR/${recording_id}_L1_vocals_mono.wav
speed_lim_factor="2.0"
dur_lim=100 # 100 min to cover most cases
out_ver="v6"
python $WORK_DIR/05_scale_segment.py \
    --in-dir ${in_dir} \
    --out-dir ${out_dir} \
    --ref-dir ${ref_dir} \
    --meta-dir ${meta_dir} \
    --audio-file ${audio_file} \
    --speed-lim-factor ${speed_lim_factor} \
    --dur-lim ${dur_lim} \
    --out-ver ${out_ver}

# switch back envionment
conda activate style

# step 8: overlay to mix concatenated vocal with background audio
seg_dir=$OUT_DIR/v6.scaled
out_dir=$OUT_DIR/v7.opt1
dur_lim="-1"
bg_audiofile=$IN_DIR/${recording_id}_L1_accompaniment.wav
r="1.0"
# out_file=${out_dir}/${recording_id}_bg+L2.wav
# with_bg=true
out_file=${out_dir}/${recording_id}_L2_mono.wav
with_bg=false
python $WORK_DIR/06_overlay.py \
    --seg-dir ${seg_dir} \
    --out-dir ${out_dir} \
    --bg-audiofile ${bg_audiofile} \
    --dur-lim ${dur_lim} \
    --out-file ${out_file} \
    --with-bg ${with_bg} \
    --r $r

# convert mixed audio from mono channel to stereo channel
# mono_audio_file=${out_dir}/${recording_id}_bg+L2.wav
# ster_audio_file=${out_dir}/${recording_id}_bg+L2_ster.wav
mono_audio_file=${out_dir}/${recording_id}_L2_mono.wav
ster_audio_file=${out_dir}/${recording_id}_L2.wav
python $WORK_DIR/helpers/mono2stereo.py \
    --infile ${mono_audio_file} \
    --outfile ${ster_audio_file}

# convert the result audio file from wav to mp3
wav_audio_file=${ster_audio_file}
mp3_audio_file=${out_dir}/${recording_id}_L2.mp3
ffmpeg -i ${wav_audio_file} -vn -b:a 128k ${mp3_audio_file}

# combine audio and video (no sound) files
AUDIO_FILE_OVL=${ster_audio_file}
# VIDEO_FILE_CMD=$OUT_DIR/v7.opt1/${recording_id}_bg+L2.mp4
VIDEO_FILE_CMD=$OUT_DIR/v7.opt1/${recording_id}_L2.mp4
ffmpeg -i $VIDEO_FILE_SIL -i $AUDIO_FILE_OVL -c:v copy -c:a aac $VIDEO_FILE_CMD
echo "$(basename $VIDEO_FILE_SIL) + $(basename $AUDIO_FILE_OVL) -> $(basename $VIDEO_FILE_CMD)"

# create the subtitle file
meta_file=$OUT_DIR/meta/${recording_id}_meta_v6.csv
srt_file=$OUT_DIR/v7.opt1/${recording_id}_L2.srt
python $WORK_DIR/helpers/prep_srt.py \
  --meta-file $meta_file \
  --srt-file $srt_file

# add soft subtile to the video file
VIDEO_FILE_STT=$OUT_DIR/v7.opt1/${recording_id}_bg+L2_subtitled.mp4
ffmpeg -i $VIDEO_FILE_CMD -i ${srt_file} -c copy -c:s mov_text -metadata:s:s:0 language=eng $VIDEO_FILE_STT
echo "$(basename $VIDEO_FILE_CMD) + $(basename ${srt_file}) -> $(basename $VIDEO_FILE_STT)"