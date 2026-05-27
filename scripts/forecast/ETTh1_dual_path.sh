#!/bin/sh
#
# Input-Level Dual-Path (Trend + Seasonal) Forecast on ETTh1
# Based on ETTh1_circulant.sh with --use_dual_path enabled.
#
# Architecture:
#   X ──► SeriesDecomp (FFT-based STL) ──► [Trend, Seasonal]
#                         ├─► TrendProjector(MLP) ──► T_trend
#                         └─► FullAttention Encoder ──► T_seasonal
#                                           │
#   α·T_trend + (1-α)·T_seasonal ◄────────┘
#

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=720
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt

data=ETTh1
# ── GPU 配置 ─────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=5,6
nproc_per_node=${nproc_per_node:-2}

# ── Dual-Path Configuration ──────────────────────────────────
# decomp_mode: stl (FFT-based seasonal extraction) or ma (simple moving average)
decomp_mode=stl

# period: 0 = adaptive (N // 4, recommended)
# 24 = manual daily period, 168 = weekly period
dual_path_period=0

# trend_mode: mlp (2-layer MLP, richer) or linear (single Linear projection)
trend_mode=mlp

# alpha_init: 0.5 means equal initial weight for trend and seasonal
dual_path_alpha=0.5

# ── Training ─────────────────────────────────────────────────
torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --rdzv_id=dual_path_$$ --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 run.py \
  --task_name forecast \
  --is_training 1 \
  --is_finetuning 1 \
  --seed 1 \
  --ckpt_path $ckpt_path \
  --root_path ./datasets/ \
  --data_path ${data}.csv \
  --data $data \
  --model_id etth1_dualpath_1 \
  --model $model_name \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --e_layers 2 \
  --factor 3 \
  --des 'DualPath_Exp' \
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
  --use_learnable_reg_lambda \
  --circulant_reg_lambda 0.1 \
  --use_dual_path \
   --use_token_weight \
  --decomp_mode $decomp_mode \
  --dual_path_period $dual_path_period \
  --trend_mode $trend_mode \
  --dual_path_alpha_init $dual_path_alpha
