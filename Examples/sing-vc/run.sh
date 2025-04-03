#!/bin/bash
#
# run spleeter on a video
#
# Zhenhao Ge, 2024-11-11

export https_proxy=http://10.16.0.132:8000
export http_proxy=http://10.16.0.132:8000

# use conda envionment 'style'
source ~/.zshrc
conda activate style

# set root dir as the current dir
ROOT_DIR=$HOME/code/repo/style-tts2
[ $PWD != $ROOT_DIR ] && cd $ROOT_DIR
echo "current dir: $PWD"

WORK_DIR=$ROOT_DIR/Examples/sing-vc
DATA_DIR=$ROOT_DIR/Datasets/YouTube
SPLEETER_DIR=$HOME/code/repo/spleeter
FREEVC_DIR=$HOME/code/repo/free-vc

account_id="pillow"
recording_id='adele.someone-like-you'
reference_id='voice_yufang_fssx2'

data_dir=$DATA_DIR/${account_id}
mkdir -p ${data_dir}
output_dir=$ROOT_DIR/Outputs/YouTube/${account_id}
mkdir -p ${output_dir}

yt_link="https://www.youtube.com/watch?v=z7GCiVTlv04"

# download the 1080p mp4 video file
# yt-dlp -f best ${yt_link}
audio_file=${data_dir}/${recording_id}.mp3
ref_audio_file=${data_dir}/${reference_id}.wav

# download the audio file in the format of mp3
if [ ! -f $audio_file ]; then
    yt-dlp -x --audio-format mp3 ${yt_link} -o ${audio_file}
fi    

# audio separation for the song audio
audio_file_acc=${data_dir}/${recording_id}_accompaniment.wav
audio_file_voc=${data_dir}/${recording_id}_vocals.wav
if [ ! -f $audio_file_acc ] || [ ! -f $audio_file_voc ]; then
    bash $WORK_DIR/run.spleeter.sh $SPLEETER_DIR ${audio_file} ${data_dir}
else
    echo "audio separation already done for ${recording_id}, skip."
fi

# convert wav audio files to mp3 audio files
audio_file_acc_mp3=${audio_file_acc%.wav}.mp3
audio_file_voc_mp3=${audio_file_voc%.wav}.mp3
sr=$(sox --i -r ${audio_file_acc})
nchannels=$(sox --i -c ${audio_file_acc})
ffmpeg -i ${audio_file_acc} -vn -ar $sr -ac $nchannels -b:a 192k ${audio_file_acc_mp3}
ffmpeg -i ${audio_file_voc} -vn -ar $sr -ac $nchannels -b:a 192k ${audio_file_voc_mp3}

# audio saparation for the reference voice audio
ref_audio_file_acc=${ref_audio_file/.wav/_accopaniment.wav}
ref_audio_file_voc=${ref_audio_file/.wav/_vocals.wav}
if [ ! -f $ref_audio_file_acc ] || [ ! -f $ref_audio_file_voc ]; then
    bash $WORK_DIR/run.spleeter.sh $SPLEETER_DIR ${ref_audio_file} ${data_dir}
else
    echo "audio separation already done for ${reference_id}"
fi

# prepare the text file with single pair for voice conversion
src_file=$audio_file_voc
tgt_file=$ref_audio_file_voc
txt_file=$FREEVC_DIR/txtfiles/${recording_id}_${reference_id}.txt
python $WORK_DIR/prep_txtfile_single.py \
    --src-file ${src_file} \
    --tgt-file ${tgt_file} \
    --txt-file ${txt_file}

# specify free-vc model file
ptfile=$FREEVC_DIR/checkpoints/24kHz/freevc-24.pth
[ ! -f $ptfile ] && echo "model $ptfile does not exist!" && exit 1
echo "model file: ${ptfile}"

# specify free-vc config file
hpfile=$FREEVC_DIR/configs/freevc-24.json
[ ! -f $hpfile ] && echo "config $hpfile does not exist!" && exit 1
echo "config file: ${hpfile}"

cd $FREEVC_DIR
device=0
CUDA_VISIBLE_DEVICES=$device python $FREEVC_DIR/convert_24.py \
    --hpfile $hpfile \
    --ptfile $ptfile \
    --txtpath $txt_file \
    --outdir $output_dir
cd $ROOT_DIR    

