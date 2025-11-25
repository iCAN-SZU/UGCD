#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name 'scars' \
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
    --exp_name scars_cacgcd
