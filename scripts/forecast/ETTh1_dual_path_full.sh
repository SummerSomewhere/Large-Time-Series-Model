#!/bin/sh
#
# Full Dual-Path (trend + seasonal separated) on ETTh1
# All other configs same as ETTh1_dual_path.sh
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
output_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt

data=ETTh1
# ── GPU 配置 ─────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=5,6
nproc_per_node=${nproc_per_node:-2}

# ── Dual-Path Configuration ──────────────────────────────────
decomp_mode=stl
dual_path_period=0      # adaptive (N // 4)
trend_mode=mlp
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
  --e_layers 8 \
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
  --decomp_mode $decomp_mode \
  --dual_path_period $dual_path_period \
  --trend_mode $trend_mode \
  --dual_path_alpha_init $dual_path_alpha
