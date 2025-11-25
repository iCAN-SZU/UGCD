#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port 12348 --nproc_per_node=8 train_mp.py \
    --dataset_name 'imagenet_1k' \
    --batch_size 512 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --fp16 \
    --eval_funcs 'v2' \
    --id_temp 0.1 \
    --ood_temp 0.2 \
    --exp_name imagenet_1k_cacgcd
    --print_freq 100
