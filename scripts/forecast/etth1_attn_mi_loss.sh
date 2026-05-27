#!/bin/bash
#
# ETTh1 Attention-MI Alignment Loss Fine-tuning Script
#
# Description:
#   在 Timer 微调阶段引入注意力显式对齐损失（Attention-MI Alignment Loss）。
#   将 HSIC 得到的 MI 分布作为"注意力应该呈现的样子"，
#   通过 KL 散度约束模型的注意力权重去拟合该分布。
#
#   本脚本先运行 Baseline，再运行改进版，最后打印两者的 MSE/MAE 对比。
#
# Formula:
#   L_attn_guidance = KL(P_MI || Mean_Attention_Map)
#
# Usage:
#   bash scripts/forecast/etth1_attn_mi_loss.sh
#
# Customization via environment variables:
#   ATTN_MI_FILE   path to MI distribution file (.json or .pt)
#   ATTN_MI_LAYERS comma-separated layer indices (default: "6,7")
#   ATTN_MI_WEIGHT loss weight (default: 0.1)
#   CUDA_VISIBLE_DEVICES  override GPU selection
#
# Example:
#   ATTN_MI_FILE=global_mi_peaks_etth1.json \
#   ATTN_MI_LAYERS="6,7" \
#   ATTN_MI_WEIGHT=0.05 \
#   bash scripts/forecast/etth1_attn_mi_loss.sh
#

set -e

_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$_ROOT" || exit 1

# ─────────────────────────────────────────────────────────────────────────────
# 实验参数配置
# ─────────────────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
N_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
data=ETTh1
root_path=./datasets/
num_workers=4
e_layers=8
factor=3
d_model=1024
d_ff=2048
n_heads=8
features=M
batch_size=2048

# ── Attention-MI Alignment Loss 参数 ─────────────────────────────────────────
ATTN_MI_FILE=${ATTN_MI_FILE:-./results/mi_hsic_etth1_decoder/global_mi_peaks_etth1.json}
ATTN_MI_LAYERS=${ATTN_MI_LAYERS:-0,1,2,3,4,5,6,7}
ATTN_MI_WEIGHT=${ATTN_MI_WEIGHT:-1}
ATTN_MI_TAU=${ATTN_MI_TAU:-0.1}

# 可学习的 weight 参数（初始值由 ATTN_MI_WEIGHT 指定）
USE_LEARNABLE=${USE_LEARNABLE_ATTN_MI_WEIGHT:-0}
# 0 = 固定权重（推荐），1 = 可学习权重

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: 运行 Baseline（无 Attention-MI Loss）
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo ">>> Running: Baseline"
echo "============================================================"
torchrun --nnodes=1 --nproc_per_node=$N_GPU run.py \
    --task_name forecast \
    --is_training 1 \
    --seed 1 \
    --ckpt_path "$ckpt_path" \
    --root_path "$root_path" \
    --data_path ${data}.csv \
    --data "$data" \
    --model_id etth1_baseline \
    --model "$model_name" \
    --features "$features" \
    --seq_len "$seq_len" \
    --label_len "$label_len" \
    --pred_len "$pred_len" \
    --output_len "$output_len" \
    --e_layers "$e_layers" \
    --factor "$factor" \
    --des 'Exp' \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --learning_rate 3e-5 \
    --num_workers "$num_workers" \
    --patch_len "$patch_len" \
    --train_test 0 \
    --subset_rand_ratio 1 \
    --itr 1 \
    --use_ims \
    --use_multi_gpu \
    2>&1 | tee baseline_output.log | grep -E 'mse:|mae:'

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 解析 Baseline 结果
# ─────────────────────────────────────────────────────────────────────────────
BL_MSE=$(grep -oP 'mse:\K[0-9.]+' baseline_output.log | tail -1)
BL_MAE=$(grep -oP 'mae:\K[0-9.]+' baseline_output.log | tail -1)
echo "Baseline parsed: mse=$BL_MSE, mae=$BL_MAE"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: 运行改进版（带 Attention-MI Alignment Loss）
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo ">>> Running: Improved (Attention-MI Alignment Loss)"
echo "============================================================"
torchrun --nnodes=1 --nproc_per_node=$N_GPU run.py \
    --task_name forecast \
    --is_training 1 \
    --seed 1 \
    --ckpt_path "$ckpt_path" \
    --root_path "$root_path" \
    --data_path ${data}.csv \
    --data "$data" \
    --model_id etth1_attn_mi_loss \
    --model "$model_name" \
    --features "$features" \
    --seq_len "$seq_len" \
    --label_len "$label_len" \
    --pred_len "$pred_len" \
    --output_len "$output_len" \
    --e_layers "$e_layers" \
    --factor "$factor" \
    --des 'Exp' \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --learning_rate 3e-5 \
    --num_workers "$num_workers" \
    --patch_len "$patch_len" \
    --train_test 0 \
    --subset_rand_ratio 1 \
    --itr 1 \
    --use_ims \
    --use_multi_gpu \
    --use_attn_mi_loss \
    --attn_mi_file "$ATTN_MI_FILE" \
    --attn_mi_loss_layers "$ATTN_MI_LAYERS" \
    --attn_mi_loss_weight "$ATTN_MI_WEIGHT" \
    --attn_mi_tau "$ATTN_MI_TAU" \
    $([ "$USE_LEARNABLE" -eq 1 ] && echo "--use_learnable_attn_mi_weight") \
    2>&1 | tee improved_output.log | grep -E 'mse:|mae:'

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: 解析改进版结果
# ─────────────────────────────────────────────────────────────────────────────
IM_MSE=$(grep -oP 'mse:\K[0-9.]+' improved_output.log | tail -1)
IM_MAE=$(grep -oP 'mae:\K[0-9.]+' improved_output.log | tail -1)
echo "Improved parsed: mse=$IM_MSE, mae=$IM_MAE"

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: 打印对比表格
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "                      实验结果对比"
echo "============================================================"
printf "%-20s %-15s %-15s %-15s\n" "Metric" "Baseline" "Improved" "Change"
echo "------------------------------------------------------------"

# Compute delta using parsed variables
if [ -n "$BL_MSE" ] && [ -n "$IM_MSE" ] && [ -n "$BL_MAE" ] && [ -n "$IM_MAE" ]; then
    DELTA_MSE=$(python3 -c "print(f'{float(\"$IM_MSE\") - float(\"$BL_MSE\"):+.6f}')")
    DELTA_MAE=$(python3 -c "print(f'{float(\"$IM_MAE\") - float(\"$BL_MAE\"):+.6f}')")
    IMP_MSE=$(python3 -c "print(f'{(float(\"$BL_MSE\") - float(\"$IM_MSE\")) / float(\"$BL_MSE\") * 100:.2f}%')")
    IMP_MAE=$(python3 -c "print(f'{(float(\"$BL_MAE\") - float(\"$IM_MAE\")) / float(\"$BL_MAE\") * 100:.2f}%')")
else
    DELTA_MSE="N/A"; DELTA_MAE="N/A"
    IMP_MSE="N/A"; IMP_MAE="N/A"
fi

printf "%-20s %-15s %-15s %-15s\n" "MSE" "$BL_MSE" "$IM_MSE" "$DELTA_MSE"
printf "%-20s %-15s %-15s %-15s\n" "MAE" "$BL_MAE" "$IM_MAE" "$DELTA_MAE"
echo "------------------------------------------------------------"
echo "  (Change = Improved - Baseline; negative means improvement)"
echo "  MSE improvement: $IMP_MSE"
echo "  MAE improvement: $IMP_MAE"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# 清理临时日志文件
# ─────────────────────────────────────────────────────────────────────────────
rm -f baseline_output.log improved_output.log

echo ""
echo "Attention-MI Alignment Loss 对比实验完成"
