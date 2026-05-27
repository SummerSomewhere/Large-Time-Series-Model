#!/bin/bash
#
# Timer ETTh1 Curvature Analysis
#
# Implements the "Curvature" geometric metric from:
#   Skean et al. (2025) "Layer by Layer" / Hosseini & Fedorenko (2023)
#
# Curvature: measures how sharply token embeddings turn across consecutive
# positions in the hidden representation trajectory.
#   C = (1/(N-2)) * sum_k arccos( dot(v_{k+1}, v_k) / (||v_{k+1}|| * ||v_k||) )
# where v_k = z_{k+1} - z_k is the velocity between consecutive patches.
#
# Timer: seq_len=672, patch_len=96 → N=7 patches per sample.
#
# Outputs (results/curvature_etth1/):
#   figA_curvature_by_layer.png      — curvature vs layer depth (mean ± std)
#   figB_curvature_distribution.png  — per-sample violin distribution per layer
#   figC_curvature_vs_position.png   — heatmap: curvature per patch position
#   figD_normalized_comparison.png   — min-max normalized curvature vs layer depth
#   curvature_results.json           — raw values + config
#

_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$_ROOT" || exit 1

export CUDA_VISIBLE_DEVICES=0,1,2,3

# ── Timer + ETTh1 defaults ──────────────────────────────────────────────────
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
root_path=./datasets/
data_path=ETTh1.csv
data=ETTh1
seq_len=672
pred_len=96
label_len=576
output_len=96
patch_len=96
e_layers=8
factor=3
d_model=1024
d_ff=2048
n_heads=8
features=M
embed=timeF
freq=h
batch_size=64
num_workers=6
stride=1
subset_rand_ratio=1.0

# ── Curvature analysis params ───────────────────────────────────────────────
max_samples=2000
out_dir=./results/curvature_etth1
seed=42

python3 experiments/etth1_curvature_analysis.py \
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
  --dropout 0.1 \
  --activation gelu \
  --stride $stride \
  --subset_rand_ratio $subset_rand_ratio \
  --num_workers $num_workers \
  --batch_size $batch_size \
  --max_samples $max_samples \
  --out_dir $out_dir \
  --seed $seed \
  --device cuda
