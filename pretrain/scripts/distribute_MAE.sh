#!/bin/bash
# ulimit -u unlimited
# bash /home/hw/shijie/MAE_BYOL/code/pretrain/scripts/distribute_MAE.sh /home/hw/shijie/MAE_BYOL/pretrain_out
get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

PATH1=$(get_real_path $1)    # train_output_path

export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=/home/hw/shijie/MAE_BYOL/code/pretrain/scripts/jobstart_hccl.json

export SERVER_ID=0
rank_start=0
# echo "$DEVICE_NUM"

for((i=0; i<${DEVICE_NUM}; i++))
do  
    echo "$i"
    export DEVICE_ID=${i}
    # export DEVICE_ID=$((4 + i))
    export RANK_ID=$((rank_start + i))

    rm -rf $PATH1/train_parallel$i
    mkdir $PATH1/train_parallel$i
    cp -r /home/hw/shijie/MAE_BYOL/code/pretrain $PATH1/train_parallel$i
    cd $PATH1/train_parallel$i || exit
    echo "Start training for rank $RANK_ID, device $DEVICE_ID."
    env > env.log
    python3 ./pretrain/train_offline_MAE.py >${RANK_ID}.log 2>&1 &
done
