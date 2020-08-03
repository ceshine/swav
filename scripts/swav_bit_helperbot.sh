#!/usr/bin/fish
set -x DATASET_PATH "/data/pretrain_dataset/"
set -x EXPERIMENT_PATH "cache/swav/swav_bit50_helperbot_e5"
mkdir -p $EXPERIMENT_PATH

python -u main_swav_bit_helperbot.py \
--data_path $DATASET_PATH \
--nmb_crops 2 4 \
--size_crops 256 96 \
--min_scale_crops 0.35 0.10 \
--max_scale_crops 1. 0.35 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_prototypes 1000 \
--queue_length 2560 \
--epochs 5 \
--batch_size 24 \
--base_lr 0.00008 \
--freeze_prototypes_niters 500 \
--wd 0. \
--warmup_epochs .5 \
--arch BiT-M-R50x1 \
--pretrained_path cache/pretrained/ \
--use_fp16 true \
--dump_path $EXPERIMENT_PATH \
--checkpoint_freq 5000 \
--grad_accu 4 \
--hidden_mlp 1028 \
--seed 42
