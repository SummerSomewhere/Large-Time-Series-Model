#!/bin/bash
#
# 高/低 MI 组周期分量（S项）方差对比分析实验启动脚本
#
# 功能：
#   1. 提取所有 patch 的时间序列
#   2. 计算每个 patch 的 HSIC MI 分数
#   3. 将 patch 分为高 MI 组（>Q3）和低 MI 组（≤Q3）
#   4. 对每个 patch 的时间序列做周期分解（提取 S 项）
#   5. 计算每组 S 项的方差，绘制 boxplot/bar chart 对比图
#
# 输出：
#   - seasonal_variance_comparison.png — Boxplot + Bar Chart 对比
#   - seasonal_variance_comparison_violin.png — 小提琴图
#
# 用法（单卡）：
#   bash scripts/forecast/etth1_mi_seasonal_variance.sh
#
# 用法（多卡，必须用 torchrun）：
#   torchrun --nnodes=1 --nproc_per_node=4 experiments/etth1_mi_seasonal_variance.py \
#     --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
#     --root_path ./datasets/ --data_path ETTh1.csv --data ETTh1 \
#     --use_multi_gpu

_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$_ROOT" || exit 1

# ── 实验参数（与 etth1_psd_analysis.sh 保持一致） ──────────────────────────────
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
batch_size=64

subset_rand_ratio=1
out_dir=./results/seasonal_variance_etth1

# ── 周期分解参数 ──────────────────────────────────────────────────────────────
period=24          # ETTh1 主周期 = 24 小时
bandwidth=0.05     # 带通滤波器的频率带宽

# ── 多卡配置（物理 GPU 4–7）───────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nnodes=1 --nproc_per_node=4 experiments/etth1_mi_seasonal_variance.py \
  --ckpt_path $ckpt_path \
  --root_path $root_path \
  --data_path ${data}.csv \
  --data $data \
  --features $features \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --patch_len $patch_len \
  --e_layers $e_layers \
  --factor $factor \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --use_ims \
  --use_multi_gpu \
  --batch_size $batch_size \
  --num_workers $num_workers \
  --out_dir $out_dir \
  --subset_rand_ratio $subset_rand_ratio \
  --period $period \
  --bandwidth $bandwidth
