#!/bin/sh
#
# Dual-Path (Trend + Seasonal) Forecast on ETTh1 — Parallel Mode
# Trend and seasonal each have their own encoder (more expressive, more parameters).
#
# Architecture:
#   Trend Path  : separate encoder with circulant attention (dual_path_trend_layers)
#   Seasonal Path: full encoder with Circulant Attention
#   Recomposition: alpha * Trend + (1 - alpha) * Seasonal  (alpha is learnable)
#

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt

data=ETTh1
# ── GPU 配置 ─────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=5,6,7
nproc_per_node=3

# ── Dual-Path Configuration ──────────────────────────────────
# parallel: separate encoder for trend path (more expressive)
dual_path_mode=parallel

# trend_mode: linear (per-patch Linear projection) or conv (depthwise 1D Conv + gating)
trend_mode=conv

# Trend encoder layers (can be fewer than e_layers since trend needs less capacity)
dual_path_trend_layers=2

# alpha_init: 0.5 means equal initial weight for trend and seasonal
dual_path_alpha=0.5

# ── Training ─────────────────────────────────────────────────
torchrun --nnodes=1 --nproc_per_node=$nproc_per_node run.py \
  --task_name forecast \
  --is_training 1 \
  --is_finetuning 1 \
  --seed 1 \
  --ckpt_path $ckpt_path \
  --root_path ./datasets/ \
  --data_path ${data}.csv \
  --data $data \
  --model_id etth1_dualpath_parallel_1 \
  --model $model_name \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --e_layers 8 \
  --factor 3 \
  --des 'DualPathParallel_Exp' \
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
  --use_circulant_attention \
  --circulant_head_dim 16 \
  --causal_attention 1 \
  --ca_last_n_layers 2 \
  --use_learnable_reg_lambda \
  --circulant_reg_lambda 0.01 \
  --use_dual_path \
  --dual_path_mode $dual_path_mode \
  --trend_mode $trend_mode \
  --dual_path_trend_layers $dual_path_trend_layers \
  --dual_path_alpha_init $dual_path_alpha
