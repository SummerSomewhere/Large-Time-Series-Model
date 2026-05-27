#!/bin/bash
#
# t-SNE 可视化实验启动脚本
#
# 功能：
#   提取所有 patch 的最后一层隐藏状态向量，
#   计算 HSIC MI 分数，
#   使用 t-SNE 降维并可视化 High-MI vs Low-MI Patch 的分布差异。
#
# 输出（3 张图）：
#   1. tsne_mi_scatter.png     — 主散点图（按 MI 分组着色 + 质心标注）
#   2. tsne_density.png        — 分组密度图（带等高线）
#   3. tsne_mi_gradient.png    — MI 渐变图（连续颜色映射）
#
# 用法（单卡）：
#   bash scripts/forecast/etth1_mi_tsne.sh
#
# 用法（多卡，必须用 torchrun）：
#   torchrun --nnodes=1 --nproc_per_node=4 experiments/etth1_mi_tsne.py \
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
out_dir=./results/tsne_etth1

# ── t-SNE 参数 ────────────────────────────────────────────────────────────────
max_batches=0               # 0=使用全部测试样本（不限制 batch 数量）
max_samples=0               # 0=使用全部样本（不采样）
tsne_perplexity=30.0      # t-SNE perplexity，控制局部/全局权衡
tsne_n_iter=1000           # t-SNE 迭代次数
tsne_random_state=42       # 随机种子，保证可复现

# ── 多卡配置（物理 GPU 4–7）───────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nnodes=1 --nproc_per_node=4 experiments/etth1_mi_tsne.py \
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
  --max_batches $max_batches \
  --max_samples $max_samples \
  --tsne_perplexity $tsne_perplexity \
  --tsne_n_iter $tsne_n_iter \
  --tsne_random_state $tsne_random_state
