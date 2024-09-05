#!/bin/bash
#
# prepare the OOD texts with the text column
#
# Zhenhao Ge, 2024-08-09

work_dir=$HOME/code/repo/style-tts2
data_dir="/home/data/LibriTTS/wav24k"

vmem="40G"
mem="40G"
cpu="04:59:00"
goodmachines="*@titan[3-4]*,*@titan[6-9]*,*@titan1*"
# goodmachines="*@rats[1-2]*,*@rats[5-6]*,*@titan[1-4].*,*@titan[6-9]*,*@titan10.*,*@c*" # goodmachines="*@rats*,*@titan*,*@c*"
SUBMIT_SGE="qsub-command -l h_vmem=$vmem,mem_free=$mem,h_rt=$cpu -q $goodmachines"

cache_dir=$work_dir/cache
list_dir=$work_dir/Data/lists && mkdir -p ${list_dir}

in_txt_file=${work_dir}/Data/OOD_texts.ori.txt

# split the input text file into multiple sub text files (1000 lines per file)
split -l 1000 -d ${in_txt_file} $list_dir/OOD_texts.

# get the list file
find ${list_dir}/OOD_texts.* > ${work_dir}/Data/OOD_texts.lst

# prepare the bash script with multiple commands (one command per line)
commands_script=${work_dir}/Scripts/run.prep_ood_texts.sh
[ -f ${commands_script} ] && rm ${commands_script}
touch ${commands_script}
for in_txt_file in $(cat ${work_dir}/Data/OOD_texts.lst); do
    out_txt_file=${in_txt_file/OOD_texts/OOD_texts2}
    echo "source /home/users/zge/.zshrc; conda activate style; python ${work_dir}/Scripts/prep_ood_texts.py --in-txt-file ${in_txt_file} --out-txt-file ${out_txt_file} --data-dir ${data_dir}" >> ${commands_script} 
done

# # single run
# python ${work_dir}/Scripts/prep_ood_texts.py \
#     --in-txt-file ${work_dir}/Data/lists/OOD_texts.00 \
#     --out-txt-file ${work_dir}/Data/lists/OOD_texts2.00 \
#     --data-dir ${data_dir}

$SUBMIT_SGE -batch_size 1 -N prep-ood-texts -o ${work_dir}/Logs/prep_ood_texts -f ${commands_script}
