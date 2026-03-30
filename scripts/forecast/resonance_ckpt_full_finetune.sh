#!/bin/sh
# 用 ETTh1_resonance.sh 产出的 checkpoint，在指定数据集上做全参微调。
# 仓库根目录执行: bash scripts/forecast/resonance_ckpt_full_finetune.sh
#
set -e

# =============================================================================
# 在下面直接改（勿依赖 export，改完保存即生效）
# =============================================================================

# 必填：改成 ETTh1_resonance 跑完后目录里的 checkpoint.pth（须含 resonance 层）


CKPT_PATH=checkpoints/Timer_forecast_1.0.ckpt
# 目标数据集（与 data_factory 里名字一致）
DATA=ETTh1

# 数据目录与文件名（相对仓库根）
ROOT_PATH=./datasets/ETT-small/
DATA_PATH="${DATA}.csv"

# 时间特征 / 共振物理步长（小时）；换 ETTm 等时按需改 FREQ、RES_DT
FREQ=h
RESONANCE_DT_HOURS=1.0

# 可选：显式周期（小时），不设则 run.py 按 FREQ 推断；要覆盖 ω 初值可再设 resonance_omega（见 run.py）
# RESONANCE_PERIOD_HOURS=24

export CUDA_VISIBLE_DEVICES=3,4
NPROC=2
SEED=1
NUM_WORKERS=4
BATCH_SIZE=1024

# 与 ETTh1_resonance 对齐；换 ckpt 时必须与保存该 ckpt 时一致，否则 shape 报错
model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
e_layers=8
factor=3
d_model=1024
d_ff=2048

# 全参微调学习率；ω/λ 仍走独立组（见 run.py）
LEARNING_RATE=3e-5
FINETUNE_RES_OMEGA_LR=1e-5
FINETUNE_RES_LAMBDA_LR=1e-5

# 频域混合损失系数；0 = 纯 MSE
LOSS_FFT_ALPHA=0.0

# 微调轮数；留空表示不写该参数（用 run.py 默认 10）
FINETUNE_EPOCHS=

# 实验名前缀 → model_id = ${MODEL_ID_PREFIX}_${DATA}_sr_1
MODEL_ID_PREFIX=res_full_ft

# 共振 head 掩码，留空=不写参数（全 head）；否则例: RESONANCE_HEAD_MASK="1 1 1 1 1 1 1 1"
RESONANCE_HEAD_MASK=

# =============================================================================
# 以下一般不用改
# =============================================================================

if [ -z "$CKPT_PATH" ]; then
  echo "ERROR: 请在脚本顶部填写 CKPT_PATH" >&2
  exit 1
fi
if [ ! -f "$CKPT_PATH" ]; then
  echo "ERROR: 找不到权重文件: $CKPT_PATH" >&2
  exit 1
fi

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
    --is_finetuning 0 \
    --is_training 0 \
    --seed "$SEED" \
    --ckpt_path "$CKPT_PATH" \
    --root_path "$ROOT_PATH" \
    --data_path "$DATA_PATH" \
    --data "$DATA" \
    --freq "$FREQ" \
    --model_id "${MODEL_ID_PREFIX}_${DATA}_sr_${subset_rand_ratio}" \
    --model "$model_name" \
    --features M \
    --seq_len "$seq_len" \
    --label_len "$label_len" \
    --pred_len "$pred_len" \
    --output_len "$output_len" \
    --e_layers "$e_layers" \
    --factor "$factor" \
    --des 'Exp' \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --finetune_res_omega_lr "$FINETUNE_RES_OMEGA_LR" \
    --finetune_res_lambda_lr "$FINETUNE_RES_LAMBDA_LR" \
    --num_workers "$NUM_WORKERS" \
    --patch_len "$patch_len" \
    --train_test 0 \
    --subset_rand_ratio "$subset_rand_ratio" \
    --itr 1 \
    --gpu 0 \
    --use_ims \
    --use_multi_gpu \
    --diurnal_attn_bias 0 \
    --resonance_last_layer 1 \
    --resonance_dt_hours "$RESONANCE_DT_HOURS" \
    --resonance_lambda_init 0.1 \
    --resonance_phi_init 0.0 \
    --finetune_trainable full \
    --loss_fft_alpha "$LOSS_FFT_ALPHA" \
    $_extra_epochs \
    $_extra_res_mask
done
