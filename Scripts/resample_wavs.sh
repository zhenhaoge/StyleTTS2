#!/bin/bash
#
# resample audio samples
# no longer needed, the audio files will be resampled to 24000 on-the-fly
#
# Zhenhao Ge, 2024-05-29

DATA_DIR=/home/users/zge/data1/datasets/LJSpeech
INPUT_DIR=${DATA_DIR}/wavs-22050
OUTPUT_DIR=${DATA_DIR}/wavs-24000

[ ! -d $OUTPUT_DIR ] && mkdirs $OUTPUT_DIR

input_wavs=($(find $INPUT_DIR -name "*.wav"))

# get input sampling rate
sr=$(sox --i -r ${input_wavs[0]})
echo "sample rate of the input audio files: $sr"

# set output sampling rate
sr2=24000
echo "sample rate of the output audio files: $sr2"

# resample audio from $sr to $sr2
for f in ${input_wavs[@]}; do
    filename=$(basename $f)
    f2=${OUTPUT_DIR}/${filename}
    sox -t wav $f -r $sr2 -t wav $f2
    echo "$f --> $f2"
done