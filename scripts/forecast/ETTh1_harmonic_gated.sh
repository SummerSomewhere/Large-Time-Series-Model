#!/bin/sh
# ETTh1 + Timer「谐波门控共振头」微调：单 specialist head，S = S_orig ⊙ σ(RS+b)；
# RS 仅用 physical_timestamps 的多谐波 cos(2πkωΔT+φ_k)（cos(A-B) 分解）；λ_k 初值极小 + FFT 热启动 ω。
# 混合损失见 run.py loss_fft_alpha。与 ETTh1_resonance 数据/IMS 配置类似。
#
# 仓库根目录: bash scripts/forecast/ETTh1_harmonic_gated.sh
#
set -e

# =============================================================================
# 在下面直接改
# =============================================================================

export CUDA_VISIBLE_DEVICES=3,4
NPROC=2

model_name=Timer
data=ETTh1
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt

seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96

# harmonic_gated_resonance=1 时不必再开 resonance_last_layer（物理时间仍会为 harmonic 计算）
# 仅训谐波子模块 + 预测头 proj，其余 ~67M backbone 冻结
FINETUNE_TRAINABLE=harmonic_proj

LEARNING_RATE=1e-4
FINETUNE_RES_OMEGA_LR=1e-5
FINETUNE_RES_LAMBDA_LR=1e-5
LOSS_FFT_ALPHA=0.5

# Specialist head 索引 0..n_heads-1；谐波个数 K（ω,2ω,…,Kω）；λ_k 初值（强度预热，宜极小）
HARMONIC_SPECIALIST_HEAD=0
HARMONIC_N_HARMONICS=3
HARMONIC_LAMBDA_INIT=1e-4
# 首个 train batch 用 rFFT 主峰热启动 ω；0=关闭，仅用周期先验
HARMONIC_FFT_WARMSTART=1

FINETUNE_EPOCHS=
RESONANCE_HEAD_MASK=

# =============================================================================

if [ "$NPROC" -lt 1 ]; then NPROC=1; fi

_extra_epochs=
if [ -n "$FINETUNE_EPOCHS" ]; then
  _extra_epochs="--finetune_epochs $FINETUNE_EPOCHS"
fi

_extra_res_mask=
if [ -n "$RESONANCE_HEAD_MASK" ]; then
  _extra_res_mask="--resonance_head_mask $RESONANCE_HEAD_MASK"
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
    --model_id "etth1_harm_gated_sr_${subset_rand_ratio}" \
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
    --learning_rate "$LEARNING_RATE" \
    --finetune_res_omega_lr "$FINETUNE_RES_OMEGA_LR" \
    --finetune_res_lambda_lr "$FINETUNE_RES_LAMBDA_LR" \
    --num_workers 4 \
    --patch_len "$patch_len" \
    --train_test 0 \
    --subset_rand_ratio "$subset_rand_ratio" \
    --itr 1 \
    --gpu 0 \
    --use_ims \
    --use_multi_gpu \
    --diurnal_attn_bias 0 \
    --resonance_last_layer 0 \
    --harmonic_gated_resonance 1 \
    --harmonic_specialist_head "$HARMONIC_SPECIALIST_HEAD" \
    --harmonic_n_harmonics "$HARMONIC_N_HARMONICS" \
    --harmonic_lambda_init "$HARMONIC_LAMBDA_INIT" \
    --harmonic_fft_warmstart "$HARMONIC_FFT_WARMSTART" \
    --resonance_dt_hours 1.0 \
    --finetune_trainable "$FINETUNE_TRAINABLE" \
    --loss_fft_alpha "$LOSS_FFT_ALPHA" \
    $_extra_epochs \
    $_extra_res_mask
done
