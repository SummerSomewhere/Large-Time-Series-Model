#!/bin/bash
#
# 平均自相关函数对比 (Mean ACF Profiles) 实验启动脚本
#
# 功能：
#   提取所有 patch 的时间序列，按 MI 分数分为高/低两组，
#   计算每组平均归一化 ACF 并对比，输出 4 张图：
#     1. acf_high_vs_low_mi.png         — 高/低 MI 组平均 ACF 对比（含置信区间）
#     2. acf_difference_high_minus_low.png — ACF 差值曲线（High-MI − Low-MI）
#     3. acf_decay_rate_comparison.png    — 衰减速度对比（半衰期 + 累积能量）
#     4. mi_score_histogram.png           — patch MI 分数直方图
#
# 数学定义：
#   rho(k) = sum_{t=1}^{n-k} (x_t - mu)(x_{t+k} - mu) / sum_{t=1}^{n} (x_t - mu)^2
#   其中 k 为滞后阶数，范围 0 <= k <= L//2
#
# 用法（单卡）：
#   bash scripts/forecast/etth1_mi_acf_analysis.sh
#
# 用法（多卡，必须用 torchrun）：
#   torchrun --nnodes=1 --nproc_per_node=4 experiments/etth1_mi_acf_analysis.py \
#     --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
#     --root_path ./datasets/ --data_path ETTm1.csv --data ETTm1 \
#     --use_multi_gpu

_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$_ROOT" || exit 1

# ── 实验参数 ─────────────────────────────────────────────────────────────────
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
out_dir=./results/acf_ettm1

# ── HSIC MI 文件（由 etth1_mi_hsic_peaks.py 生成）────────────────────────────
# 例如: ./results/mi_hsic_ettm1/global_mi_peaks_ettm1.json
hsic_file=./results/mi_hsic_etth1/global_mi_peaks_etth1.json

# 使用哪一层的 MI 曲线（-1=最后一层）
hsic_layer=-1

# ── 多卡配置（物理 GPU 4–7）───────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nnodes=1 --nproc_per_node=4 experiments/etth1_mi_acf_analysis.py \
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
  --hsic_file "$hsic_file" \
  --hsic_layer $hsic_layer
