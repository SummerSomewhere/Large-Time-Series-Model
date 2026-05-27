#!/bin/sh
#
# ETTh1 finetune: Timer with T5-style relative position bias on attention logits (learned
# [num_buckets, n_heads] table; exact + log buckets for i-j). Patch stream is value-only linear
# projection — no additive sinusoidal PE, no depthwise causal conv, no RoPE/TS-RoPE on Q/K.
#
# Physical GPUs: set CUDA_VISIBLE_DEVICES as needed; processes see cuda:0, cuda:1 (two ranks).

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

# Optional overrides:
#   PATCH_TS_ROPE=0 — must stay 0 when using relative bias (script default is 0).
#   PATCH_ROPE=0 — must stay 0 when using relative bias.
#   PATCH_NO_POSITION=0 — redundant when PATCH_RELATIVE_POSITION_BIAS=1 (additive PE already off).
#   IMPLICIT_POSITION_CONV=0 — no implicit conv (default 0); keep 0 for pure value + rel-bias logits.
#   PATCH_RELATIVE_POSITION_BIAS=1 — T5-style bias after QK^T (default 1).
#   PATCH_RELATIVE_BUCKETS=32 — even; try 64 for finer bins.
#   PATCH_RELATIVE_MAX_DISTANCE=128 — tail of log bucketing (>= num_patches).
PATCH_TS_ROPE="${PATCH_TS_ROPE:-0}"
PATCH_TS_ROPE_K="${PATCH_TS_ROPE_K:-5}"
PATCH_ROPE="${PATCH_ROPE:-0}"
PATCH_ROPE_BASE="${PATCH_ROPE_BASE:-10000}"
PATCH_NO_POSITION="${PATCH_NO_POSITION:-0}"
IMPLICIT_POSITION_CONV="${IMPLICIT_POSITION_CONV:-0}"
IMPLICIT_POSITION_KERNEL="${IMPLICIT_POSITION_KERNEL:-3}"
PATCH_RELATIVE_POSITION_BIAS="${PATCH_RELATIVE_POSITION_BIAS:-1}"
PATCH_RELATIVE_BUCKETS="${PATCH_RELATIVE_BUCKETS:-32}"
PATCH_RELATIVE_MAX_DISTANCE="${PATCH_RELATIVE_MAX_DISTANCE:-128}"

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
data=ETTh1

for subset_rand_ratio in 1
do
torchrun --nnodes=1 --nproc_per_node=2 run.py \
  --task_name forecast \
  --is_finetuning 1 \
  --is_training 1 \
  --seed 1 \
  --test_shuffle_patches 1 \
  --ckpt_path "$ckpt_path" \
  --root_path ./datasets \
  --data_path $data.csv \
  --data $data \
  --model_id etth1_rel_bias_$subset_rand_ratio \
  --model $model_name \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --e_layers 8 \
  --factor 3 \
  --des 'RelBias' \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 1024 \
  --learning_rate 3e-5 \
  --num_workers 4 \
  --patch_len $patch_len \
  --geometric_hpe 0 \
  --patch_ts_rope "$PATCH_TS_ROPE" \
  --patch_ts_rope_k "$PATCH_TS_ROPE_K" \
  --patch_rope "$PATCH_ROPE" \
  --patch_rope_base "$PATCH_ROPE_BASE" \
  --patch_no_position "$PATCH_NO_POSITION" \
  --implicit_position_conv "$IMPLICIT_POSITION_CONV" \
  --implicit_position_kernel "$IMPLICIT_POSITION_KERNEL" \
  --patch_relative_position_bias "$PATCH_RELATIVE_POSITION_BIAS" \
  --patch_relative_position_buckets "$PATCH_RELATIVE_BUCKETS" \
  --patch_relative_position_max_distance "$PATCH_RELATIVE_MAX_DISTANCE" \
  --train_test 0 \
  --subset_rand_ratio $subset_rand_ratio \
  --itr 1 \
  --gpu 0 \
  --use_ims \
  --use_multi_gpu
done
