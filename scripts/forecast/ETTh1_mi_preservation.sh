#!/bin/sh
# ETTh1 finetune with Timer MI Preservation Loss (cosine align layer-0 vs last-layer hidden at patches 3,6).
# Training-only: inference does not pass return_mi_feats (no extra encoder pass).
#
# Override via env:
#   LAMBDA_MI=0.002 MI_WARMUP=3 bash scripts/forecast/ETTh1_mi_preservation.sh
#   CUDA_VISIBLE_DEVICES=0,1 CKPT=checkpoints/Timer_forecast_1.0.ckpt bash scripts/forecast/ETTh1_mi_preservation.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4}"

LAMBDA_MI="${LAMBDA_MI:-0.001}"
MI_WARMUP="${MI_WARMUP:-2}"
MI_PATCH="${MI_PATCH:-3,6}"

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path="${CKPT:-checkpoints/Timer_forecast_1.0.ckpt}"
data=ETTh1

for subset_rand_ratio in 1
do
torchrun --nnodes=1 --nproc_per_node=2 run.py \
  --task_name forecast \
  --is_finetuning 1 \
  --is_training 1 \
  --seed 1 \
  --ckpt_path "$ckpt_path" \
  --root_path ./datasets/ETT-small/ \
  --data_path "$data.csv" \
  --data "$data" \
  --model_id "etth1_mi_pres_sr_${subset_rand_ratio}" \
  --model "$model_name" \
  --features M \
  --seq_len "$seq_len" \
  --label_len "$label_len" \
  --pred_len "$pred_len" \
  --output_len "$output_len" \
  --e_layers 8 \
  --factor 3 \
  --des 'Exp_mi_pres' \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 1024 \
  --learning_rate 3e-5 \
  --num_workers 4 \
  --patch_len "$patch_len" \
  --train_test 0 \
  --subset_rand_ratio "$subset_rand_ratio" \
  --itr 1 \
  --gpu 0 \
  --use_ims \
  --use_multi_gpu \
  --mi_preservation 1 \
  --lambda_mi "$LAMBDA_MI" \
  --mi_warmup_epochs "$MI_WARMUP" \
  --mi_patch_indices "$MI_PATCH"
done
