#!/bin/bash
#
# Timer Layer-wise 线性探测实验
#
# 功能：
#   1. 加载已提取的 .pt 特征（跳过模型推理，节省时间）
#      或通过 forward hook 实时推理（无 --features_pt_path 时）
#   2. STL 分解生成 5 组语义标签
#   3. Ridge Regression 探测，计算每层对各标签的解释力
#   4. 可视化：Layer-wise R² 折线图 + 热力图
#
# 输出（results/layerwise_probe/）：
#   figA_layerwise_r2.png    — R² 折线图（每条线 = 一个语义标签）
#   figB_layerwise_mse.png    — MSE 对数折线图
#   figC_layerwise_heatmap.png — R² 热力图（层 × 标签，含数值）
#   tsne_*.png               — 各层 t-SNE 可视化
#   ridge_results.json       — 完整结果 JSON
#

_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$_ROOT" || exit 1

# ── 特征来源 ──────────────────────────────────────────────────────
# 方式 A（推荐）：加载已提取的 .pt 特征
#   由 etth1_layerwise_probe_features.py 生成
features_pt_path=./results/layerwise_probe_features/ETTh1_20000.pt
features_labels_json=./results/layerwise_probe_features/ETTh1_20000_labels.json

# 方式 B（备用）：不指定 features_pt_path，则通过 hook 实时推理
#   需要提供 --ckpt_path
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt

# ── 模型与数据参数 ─────────────────────────────────────────────────────
data=ETTh1
root_path=./datasets/
data_path=ETTh1.csv
e_layers=8
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
d_model=1024
d_ff=2048
n_heads=8
features=M
embed=timeF
freq=h
batch_size=64

# ── 线性探测参数 ──────────────────────────────────────────────────────
n_samples=0         # 0 = 使用全部样本
test_ratio=0.2     # 测试集比例
alpha=1.0          # Ridge 正则化系数
out_dir=./results/layerwise_probe

# ── 运行 ──────────────────────────────────────────────────────────────
if [ -n "$features_pt_path" ] && [ -f "$features_pt_path" ]; then
    echo "[Mode] 加载 .pt 特征: $features_pt_path"
    python experiments/etth1_linear_probe_patch_compare.py \
      --features_pt_path "$features_pt_path" \
      --labels_json "$features_labels_json" \
      --e_layers $e_layers \
      --pred_len $pred_len \
      --n_samples $n_samples \
      --test_ratio $test_ratio \
      --alpha $alpha \
      --out_dir $out_dir \
      --gpu_ids "0"
else
    echo "[Mode] hook 推理模式（无 --features_pt_path）"
    python experiments/etth1_linear_probe_patch_compare.py \
      --ckpt_path $ckpt_path \
      --root_path $root_path \
      --data_path $data_path \
      --data $data \
      --features $features \
      --embed $embed \
      --freq $freq \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --output_len $output_len \
      --patch_len $patch_len \
      --e_layers $e_layers \
      --factor 3 \
      --d_model $d_model \
      --d_ff $d_ff \
      --n_heads $n_heads \
      --batch_size $batch_size \
      --n_samples $n_samples \
      --test_ratio $test_ratio \
      --alpha $alpha \
      --out_dir $out_dir \
      --gpu_ids "0,1,2,3"
fi
