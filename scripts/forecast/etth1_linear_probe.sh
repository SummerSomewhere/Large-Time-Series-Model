#!/bin/bash
#
# 线性探针（Linear Probe）实验启动脚本
#
# 功能：
#   Stage 1: 从原始 CSV 提取语义标签（STL 分解 / 时间编码 / 波动率 / 未来真值）
#   Stage 2: 冻结 Timer 前向传播，提取每层隐藏状态
#   Stage 3: 全量 Ridge Regression 基准扫描（R² per layer × semantic）
#   Stage 4: MI 分组对比实验（High-MI vs Low-MI probe R²）
#   Stage 5: 可视化（热力图 / 折线图 / Token 案例分析）
#
# 输出（results/linear_probe_etth1/）：
#   heatmap_r2_full.png    — 全量 R² 热力图
#   heatmap_r2_high.png    — High-MI R² 热力图
#   heatmap_r2_low.png    — Low-MI R² 热力图
#   high_low_comparison.png — 高低 MI 对比折线图
#   layer_bar_{sem}.png    — 重点语义柱状图
#   probe_results.json     — 完整结果 JSON
#
# 用法（多卡）：
#   bash scripts/forecast/etth1_linear_probe.sh
#

_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$_ROOT" || exit 1

# ── 模型与数据参数 ────────────────────────────────────────────────────────
model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
data=ETTh1
root_path=./datasets/
data_path=ETTh1.csv
num_workers=6
e_layers=8
factor=3
d_model=1024
d_ff=2048
n_heads=8
features=M
batch_size=256

# ── 线性探针参数 ──────────────────────────────────────────────────────────
test_ratio=0.2          # 测试集比例
val_ratio=0.2           # 验证集比例
alpha=1.0               # Ridge 正则化系数
period=24               # STL 周期（ETTh1 小时级周期）
max_batches=0    # 0=跑全部样本
out_dir=./results/linear_probe_etth1

# ── 语义标签（逗号分隔）───────────────────────────────────────────────────
# 可选: trend, seasonal, residual, volatility, y_next1, y_next24, hour_sin, hour_cos
semantics=trend,seasonal,residual,volatility,y_next1,y_next24,hour_sin,hour_cos

# ── 多卡配置 ─────────────────────────────────────────────────────────────
# 探针实验只需少量样本，使用单卡即可
# export CUDA_VISIBLE_DEVICES=4,5,6,7

# torchrun --nnodes=1 --nproc_per_node=4 experiments/etth1_linear_probe.py \
torchrun --nnodes=1 --nproc_per_node=1 experiments/etth1_linear_probe.py \
  --ckpt_path $ckpt_path \
  --root_path $root_path \
  --data_path ${data_path} \
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
  --batch_size $batch_size \
  --num_workers $num_workers \
  --out_dir $out_dir \
  --test_ratio $test_ratio \
  --val_ratio $val_ratio \
  --alpha $alpha \
  --period $period \
  --max_batches $max_batches \
  --semantics "$semantics" \
  --hsic_file ./results/mi_hsic_etth1/global_mi_peaks_etth1.json \
  --hsic_layer all
