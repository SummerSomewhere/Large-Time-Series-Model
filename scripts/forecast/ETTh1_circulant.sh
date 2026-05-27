#!/bin/sh

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt

data=ETTh1
# ── GPU 配置 ─────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=5,6
# torchrun 直接指定使用 GPU 5,6,7
nproc_per_node=2

# train
torchrun --nnodes=1 --nproc_per_node=$nproc_per_node run.py \
  --task_name forecast \
  --is_training 1 \
  --is_finetuning 1 \
  --seed 1 \
  --ckpt_path $ckpt_path \
  --root_path ./datasets/ \
  --data_path ${data}.csv \
  --data $data \
  --model_id etth1_ca_1 \
  --model $model_name \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --e_layers 8 \
  --factor 3 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 1024 \
  --learning_rate 3e-5 \
  --num_workers 3 \
  --patch_len $patch_len \
  --train_test 0 \
  --subset_rand_ratio 1 \
  --itr 1 \
  --use_ims \
  --use_multi_gpu \
  --devices 5,6,7 \
