#!/bin/bash

# Timer MI 引导选择性跳过推理实验
# 对应 experiments/timer_mi_selective_skip.py

# ── 实验配置 ───────────────────────────────────────────────────────────────
model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
data=ETTh1

# ── MI 结果目录 ──────────────────────────────────────────────────────────────
MI_RESULT_DIR="./outputs/timer_mi_ksg_pca/Timer_MI_20260516_133108"

# ── 输出目录 ──────────────────────────────────────────────────────────────
OUTPUT_DIR="./outputs/timer_mi_selective_skip"

# ── GPU 配置 ──────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=4,5,6,7

# ── 跳过的 Token 比例（0%–50%）────────────────────────────────────────────
SKIP_RATES="0.0,0.10,0.20,0.30,0.40,0.50"

# ── 训练/推理配置 ─────────────────────────────────────────────────────────
batch_size=64
num_workers=4
seed=42

echo "=================================================="
echo "Timer MI 选择性跳过推理实验"
echo "=================================================="
echo "  数据集       : $data"
echo "  序列长度     : $seq_len"
echo "  预测长度     : $pred_len"
echo "  Patch 长度   : $patch_len"
echo "  Skip Rates   : $SKIP_RATES"
echo "  MI 结果目录  : $MI_RESULT_DIR"
echo "  输出目录     : $OUTPUT_DIR"
echo "  GPU          : $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python experiments/timer_mi_selective_skip.py \
    --mi_result_dir "$MI_RESULT_DIR" \
    --root_path ./datasets/ \
    --data_path ${data}.csv \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --patch_len $patch_len \
    --ckpt_path $ckpt_path \
    --batch_size $batch_size \
    --e_layers 8 \
    --d_model 1024 \
    --d_ff 2048 \
    --n_heads 16 \
    --dropout 0.1 \
    --skip_rates "$SKIP_RATES" \
    --output_dir $OUTPUT_DIR \
    --device cuda \
    --seed $seed

echo ""
echo "[Done] 结果保存至: $OUTPUT_DIR/"
