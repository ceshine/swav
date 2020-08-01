#!/usr/bin/fish
set -x DATASET_PATH "/data/pretrain_dataset/"
set -x EXPERIMENT_PATH "cache/swav_logs/swav_bit50_pretrain"
mkdir -p $EXPERIMENT_PATH

python -u main_swav_bit.py \
--data_path $DATASET_PATH \
--nmb_crops 2 6 \
--size_crops 224 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_prototypes 500 \
--queue_length 0 \
--epochs 5 \
--batch_size 16 \
--base_lr 0.0001 \
--final_lr 0.000001 \
--freeze_prototypes_niters 313 \
--wd 0.000001 \
--warmup_epochs 1 \
--start_warmup 0.000001 \
--arch BiT-M-R50x1 \
--use_fp16 true \
--dump_path $EXPERIMENT_PATH
