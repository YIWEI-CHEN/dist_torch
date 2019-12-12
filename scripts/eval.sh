#!/usr/bin/env bash

GPU=0
BATCH_SIZE=256
SEED=42
EXP_PATH="exp/cifar10/single_gpu/batch_size${BATCH_SIZE}_gpu${GPU}"

python single_gpu.py --data /home/yiwei/cifar10 --batch_size ${BATCH_SIZE} --gpu ${GPU} \
    --save ${EXP_PATH} --seed ${SEED} --cutout
