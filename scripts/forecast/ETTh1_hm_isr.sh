#!/bin/sh
# ETTh1 finetune with HM-ISR (hard-masked in-situ refiner): only selected patches get alpha*MLP residual.
# Default mask: patch indices 3 and 6 (override with HM_ISR_PATCH_INDICES="3,5" etc.).
# Uses --hm_isr 1 (in-stack entry after layer 0 + exit before last layer). Legacy --sig_gate is not used.
#
# Usage:
#   bash scripts/forecast/ETTh1_hm_isr.sh
#   HM_ISR_PATCH_INDICES="3,6" HM_ISR_LAMBDA=12 bash scripts/forecast/ETTh1_hm_isr.sh
#   CUDA_VISIBLE_DEVICES=0,1 CKPT=checkpoints/Timer_forecast_1.0.ckpt bash scripts/forecast/ETTh1_hm_isr.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,5}"

HM_ISR_PATCH_INDICES="${HM_ISR_PATCH_INDICES:-3,6}"
HM_ISR_LAMBDA="${HM_ISR_LAMBDA:-10.0}"

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
  --model_id "etth1_hm_isr_sr_${subset_rand_ratio}" \
  --model "$model_name" \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --e_layers 8 \
  --factor 3 \
  --des 'Exp_hm_isr' \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 1024 \
  --learning_rate 3e-5 \
  --num_workers 4 \
  --patch_len $patch_len \
  --train_test 0 \
  --subset_rand_ratio $subset_rand_ratio \
  --itr 1 \
  --gpu 0 \
  --use_ims \
  --use_multi_gpu \
  --hm_isr 1 \
  --hm_isr_lambda_scale "$HM_ISR_LAMBDA" \
  --hm_isr_patch_indices "$HM_ISR_PATCH_INDICES"
done
