#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
#
#bash scripts/run_distribute_train.sh scripts/rank_table_8pcs.json /home/hw/cyh/SAR_first_batch_8class/data/20zhang_train.mindrecord /home/hw/cyh/HRNetW48_seg/SAR_first_batch_8class/train/20230328 /home/hw/cyh/HRNetW48_seg/0-sar-code/hrnetv2_w48_imagenet_pretrained.ckpt  True

#bash scripts/run_distribute_train.sh scripts/rank_table_4pcs.json /home/hw/cyh/SAR_first_batch_8class/data/20zhang_train.mindrecord /home/hw/cyh/HRNetW48_seg/SAR_first_batch_8class/train/20230328-4 /home/hw/cyh/HRNetW48_seg/0-sar-code/hrnetv2_w48_imagenet_pretrained.ckpt True
#bash scripts/run_distribute_train.sh scripts/rank_table_4pcs.json /home/hw/cyh/SAR_first_batch_8class/data/21zhang_train.mindrecord /home/hw/cyh/HRNetW48_seg/SAR_first_batch_9class/train/20230328-4 /home/hw/cyh/HRNetW48_seg/0-sar-code/hrnetv2_w48_imagenet_pretrained.ckpt

# bash scripts/run_distribute_train-1-3.sh scripts/rank_table_4pcs-1-3.json /home/hw/cyh/SAR_first_batch_DATA/data/20zhang_train_re03.mindrecord /home/hw/cyh/HRNetW48_seg/SAR_first_batch_8class/train/20230403

# if [ $# != 4 ] && [ $# != 6 ]
# then
#     echo "Using: bash scripts/run_distribute_train.sh /home/hw/cyh/HRNetW48_seg/0-sar-code/scripts/rank_table_4pcs.json /home/hw/cyh/SAR_first_batch_8class/data/20zhang_train.mindrecord [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH]"
#     echo "or" 
#     echo "Using: bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]"
#     exit 1
# fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

PATH1=$(get_real_path $1)    # rank_table_file
PATH2=$(get_real_path $2)    # dataset_path
PATH3=$(get_real_path $3)    # train_output_path 
# PATH4=$(get_real_path $4)    # pretrained or resume ckpt_path

if [ ! -f $PATH1 ]
then
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file."
    exit 1
fi

if [ ! -f $PATH2 ]
then
    echo "error: DATASET_PATH=$PATH2 is not a directory."
    exit 1
fi

if [ ! -d $PATH3 ]
then
    echo "error: TRAIN_OUTPUT_PATH=$PATH3 is not a directory."
fi

# if [ ! -f $PATH4 ]
# then
#     echo "error: CHECKPOINT_PATH=$PATH4 is not a file."
#     exit 1
# fi

ulimit -u unlimited
export DEVICE_NUM=4
export RANK_SIZE=4
export RANK_TABLE_FILE=$PATH1

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

# echo "$DEVICE_NUM"
for((i=0; i<${DEVICE_NUM}; i++))
do  
    echo "$i"
    export DEVICE_ID=${i}
    # export DEVICE_ID=$((4 + i))
    export RANK_ID=$((rank_start + i))
    rm -rf $PATH3/train_parallel$i    
    mkdir $PATH3/train_parallel$i
    cp /home/hw/cyh/HRNetW48_seg/0-sar-code/train.py $PATH3/train_parallel$i
    cp -r /home/hw/cyh/HRNetW48_seg/0-sar-code/src $PATH3/train_parallel$i
    cd $PATH3/train_parallel$i || exit
    echo "Start training for rank $RANK_ID, device $DEVICE_ID."
    env > env.log
    if [ $# == 3 ]
    then
        # python3 train.py --data_url $PATH2 --train_url $PATH3 --checkpoint_url $PATH4 --modelarts False --run_distribute True --eval $5 &> log &
        # python3 train.py --data_url $PATH2 --train_url $PATH3 --checkpoint_url $PATH4 --modelarts False --run_distribute True --eval True &> log &
        python3 train.py --data_url $PATH2 --train_url $PATH3 --modelarts False --run_distribute True --eval True &> log &
    elif [ $# == 6 ]
    then
        python3 train.py --data_url $PATH2 --train_url $PATH3 --checkpoint_url $PATH4 --modelarts False --run_distribute True --begin_epoch $5 --eval $6 &> log &
    fi
    cd ..
done

