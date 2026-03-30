#!/bin/sh
# ETTh1 + Timer 最后一层 Resonance，与 ETTh1_resonance.sh 同款数据/损失/共振设置，
# 区别：全参微调（finetune_trainable=full），所有权重与 bias 均可更新，用于和「只训最后一层 attention」对比是否提点。
#
# 仓库根目录: bash scripts/forecast/ETTh1_resonance_full.sh
#
set -e

# =============================================================================
# 在下面直接改
# =============================================================================

export CUDA_VISIBLE_DEVICES=3,4
NPROC=2

model_name=Timer
data=ETTh2
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt

# 与 ETTh1_resonance 对齐；须与预训练/对比实验一致
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96

# 全参微调学习率可略保守；若要对齐 ETTh1_resonance 的 1e-4 可改成 1e-4
LEARNING_RATE=3e-5
FINETUNE_RES_OMEGA_LR=1e-5
FINETUNE_RES_LAMBDA_LR=1e-5

# 与 ETTh1_resonance 一致：(1-α)MSE + α·MAE(rFFT)
LOSS_FFT_ALPHA=0.5

# 留空不写 --finetune_epochs（run.py 默认 10）；或填数字加大轮数
FINETUNE_EPOCHS=

# 共振 head 掩码，留空=全 head
RESONANCE_HEAD_MASK=

# 实验 id 前缀（避免与 etth1_res_sr_* 混淆）
MODEL_ID_PREFIX=etth1_res_full

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
    --model_id "${MODEL_ID_PREFIX}_sr_${subset_rand_ratio}" \
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
    --resonance_last_layer 1 \
    --resonance_dt_hours 1.0 \
    --resonance_lambda_init 0.1 \
    --resonance_phi_init 0.0 \
    --finetune_trainable full \
    --loss_fft_alpha "$LOSS_FFT_ALPHA" \
    $_extra_epochs \
    $_extra_res_mask
done
