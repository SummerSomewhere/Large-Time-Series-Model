#!/bin/sh
# ETTh1 长序列预测 + Timer 最后一层 Resonance Head 微调（提点配置）。
# 用法（在仓库根目录）: bash scripts/forecast/ETTh1_resonance.sh
#
# 策略摘要：
# - last_attention_only：整网冻结，仅最后一层 Attention（Q/K/V/out + 共振 inner）可训。
# - ω 初值：由 --freq h 推断周期 24h → ω=1/24（与物理时间 T 的 dt_hours 一致）；可改 --resonance_period_hours。
# - Loss：(1-α)*MSE + α*MAE(rFFT)，默认 α=0.5。
# - Adam：ω、λ 组 1e-5，其余可训参数组 1e-4（--learning_rate）。
#
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4}"

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
resonance_head_mask="${RESONANCE_HEAD_MASK:-}"

data=ETTh2

NPROC="${NPROC:-2}"
if [ "$NPROC" -lt 1 ]; then NPROC=1; fi

FINETUNE_TRAINABLE="${FINETUNE_TRAINABLE:-last_attention_only}"
LOSS_FFT_ALPHA="${LOSS_FFT_ALPHA:-0.5}"

_extra_res_mask=""
if [ -n "$resonance_head_mask" ]; then
  _extra_res_mask="--resonance_head_mask $resonance_head_mask"
fi

for subset_rand_ratio in 1
do
  torchrun --nnodes=1 --nproc_per_node="$NPROC" run.py \
    --task_name forecast \
    --is_finetuning 1 \
    --is_training 1 \
    --seed 1 \
    --ckpt_path "$ckpt_path" \
    --root_path ./datasets/ETT-small/ \
    --data_path "${data}.csv" \
    --data "$data" \
    --freq h \
    --model_id "etth1_res_sr_${subset_rand_ratio}" \
    --model "$model_name" \
    --features M \
    --seq_len "$seq_len" \
    --label_len "$label_len" \
    --pred_len "$pred_len" \
    --output_len "$output_len" \
    --e_layers 8 \
    --factor 3 \
    --des 'Exp' \
    --d_model 1024 \
    --d_ff 2048 \
    --batch_size 1024 \
    --learning_rate 1e-4 \
    --finetune_res_omega_lr 1e-5 \
    --finetune_res_lambda_lr 1e-5 \
    --num_workers 4 \
    --patch_len "$patch_len" \
    --train_test 0 \
    --subset_rand_ratio "$subset_rand_ratio" \
    --itr 1 \
    --gpu 0 \
    --use_ims \
    --use_multi_gpu \
    --diurnal_attn_bias 0 \
    --resonance_last_layer 1 \
    --resonance_dt_hours 1.0 \
    --resonance_lambda_init 0.1 \
    --resonance_phi_init 0.0 \
    --finetune_trainable "$FINETUNE_TRAINABLE" \
    --loss_fft_alpha "$LOSS_FFT_ALPHA" \
    $_extra_res_mask
done
