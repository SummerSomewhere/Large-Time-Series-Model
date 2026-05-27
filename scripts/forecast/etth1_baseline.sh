#!/bin/bash

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path='random'  # 不使用预训练，从头训练
data=ETTh1

# 基线配置：不启用 Refinement
enable_refinement=0
refine_patches=""
refine_iterations=0
refine_alpha=0

for subset_rand_ratio in 1
do
  torchrun --nnodes=1 --nproc_per_node=4 run.py \
    --task_name forecast \
    --is_training 1 \
    --seed 1 \
    --ckpt_path $ckpt_path \
    --root_path ./datasets/ \
    --data_path $data.csv \
    --data $data \
    --model_id etth1_baseline \
    --model $model_name \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --output_len $output_len \
    --e_layers 8 \
    --factor 3 \
    --des 'Baseline_No_Refinement' \
    --d_model 1024 \
    --d_ff 2048 \
    --batch_size 2048 \
    --learning_rate 3e-5 \
    --train_epochs 10 \
    --num_workers 4 \
    --patch_len $patch_len \
    --train_test 1 \
    --subset_rand_ratio $subset_rand_ratio \
    --itr 1 \
    --gpu 0 \
    --use_ims \
    --use_multi_gpu \
    --enable_refinement $enable_refinement \
    --refine_patches $refine_patches \
    --refine_iterations $refine_iterations \
    --refine_alpha $refine_alpha
done
