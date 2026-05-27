#!/bin/bash
#
# Timer Layer-wise Probing — 特征提取脚本
#
# 功能：
#   1. 通过 forward(output_hidden_states=True) 获取 Timer 所有 decoder 层的 hidden_states
#   2. 每层做 Mean Pooling over patches，得到 [n_samples, D] 特征向量
#   3. 对 y_true 做 STL 分解，生成 5 组语义标签
#   4. torch.save 保存 .pt 文件，供后续 Probe 训练使用
#
# 输出（results/layerwise_probe_features/）：
#   {data}_{n_samples}.pt    — 完整特征 + 标签 + config
#   {data}_{n_samples}_labels.json — 快速预览用标签 JSON
#

_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$_ROOT" || exit 1

# ── 模型与数据参数 ─────────────────────────────────────────────────────
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
e_layers=8
factor=3
d_model=1024
d_ff=2048
n_heads=8
features=M
embed=timeF
freq=h
batch_size=64
num_workers=4

# ── 特征提取参数 ───────────────────────────────────────────────────────
n_samples=20000
out_dir=./results/layerwise_probe_features
features_name="ETTh1_20000_full"  # 全程覆盖，非仅测试集

# ── 运行 ──────────────────────────────────────────────────────────────
python experiments/etth1_layerwise_probe_features.py \
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
  --factor $factor \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --batch_size $batch_size \
  --num_workers $num_workers \
  --n_samples $n_samples \
  --out_dir $out_dir \
  --features_name "${features_name}" \
  --gpu_ids "0,1,2,3"
