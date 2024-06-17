#!/bin/bash
#
# run infer.rtf.py with different settings
#  - set manifest_file
#  - select GPU/CPU settings
#    - exp1: cuda, single_CPU=false
#    - exp2: cuda, single_CPU=true
#    - exp3: cpu, single_CPU=false
#    - exp4: cpu, single_CPU=true
#
# Zhenhao Ge, 2024-05-16

ROOT_DIR=/home/users/zge/code/repo/style-tts2

CURRENT_DIR=$PWD
[[ $CURRENT_DIR != $ROOT_DIR ]] && cd $ROOT_DIR \
  && echo "change current dir to: $ROOT_DIR"

# set model
model_path=$ROOT_DIR/Models/LibriTTS
model_name=epochs_2nd_00020.pth

# set manifest file (basic, short, mid, long)
manifest_file=$ROOT_DIR/Outputs/RTF/manifests/manifest_mid.txt

# set run_id (and device and single_CPU) for different exps
run_id=exp4

# set device and single_CPU based on run_id
# device options: cpu, cuda, or cuda:x
if [ $run_id = 'exp1' ]; then
    # GPU and CPU, multi-CPUs
    device="cuda"
    single_CPU=false
elif [ $run_id = 'exp2' ]; then
    # GPU and CPU, single-CPU
    device="cuda"
    single_CPU=true
elif [ $run_id = 'exp3' ]; then
    # CPU only, multi-CPUs
    device="cpu"
    single_CPU=false
elif [ $run_id = 'exp4' ]; then
    # CPU only, single-CPU
    device="cpu"
    single_CPU=true
else
    echo "run_id must be from exp1 to exp4" && exit 1
fi
echo "run id: $run_id, device: $device, single_CPU: $single_CPU"

# option to restricting to a single CPU (export OMP_NUM_THREADS=1 to use single CPU)
if [ "$single_CPU" = true ]; then
    echo "exporting OMP_NUM_THREADS=1"
    export OMP_NUM_THREADS=1
else
    echo "allowing multiple CPUs"
fi

# set output path with run id
output_path=$ROOT_DIR/Outputs/RTF/$run_id

# tuning parameters
diffusion_steps=10
embedding_scale=1
alpha=0.3
beta=0.7

# parameters for the rtf measurement
num_reps=10
num_warmup=3

# print out the arguments
echo "model: $model_path/$model_name"
echo "output path: $output_path"
echo "manifest file: $manifest_file"
echo "device: $device"
echo "single CPU: $single_CPU"
echo "diffusion steps: $diffusion_steps"
echo "embedding scale: $embedding_scale"
echo "alpha: $alpha"
echo "beta: $beta"

# run inference to get RTF
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

# get the average duration in the output dir
dur_total=$(soxi -DT $output_path/*.wav)
num_files=$(ls $output_path/*.wav | wc -l)
dur_mean=$(echo "scale=2; $dur_total / $num_files" | bc)
echo "mean duration of wavs in $output_path: $dur_mean seconds"