#!/bin/bash
#
# Per-token HSIC (MI) Analysis for Timer on ETTh1
#
# Reference: "Information Peaks in Transformer Representations" (DR paper)
# Method: For each decoder layer l and each token t, compute:
#   HSIC_{l,t} = HSIC( h_x^{(l)}[:, t, :],  h_y^{(l)}[:, t, :] )
# where:
#   h_x^{(l)} = input x passes through patch embedding + layers 0..l
#   h_y^{(l)} = GT future y re-fed through the SAME decoder up to layer l
#
# Output:
#   global_token_hsic_peaks_etth1_token.json  — per-layer, per-token HSIC curves
#   plots/token_hsic_per_layer.png            — bar chart per layer
#   plots/token_hsic_overlay.png             — overlay all layers
#   plots/token_hsic_heatmap.png             — heatmap layers × tokens
#   plots/high_low_token_comparison.png      — high vs low MI token comparison
#
# Usage:
#   bash scripts/forecast/etth1_token_hsic_mi_peaks.sh
#
# Must be launched from project root.
#

set -e

# ── GPU setup ────────────────────────────────────────────────────────────────
# 5 GPUs: 0,1,2,3,4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# ── Hyperparameters ─────────────────────────────────────────────────────────
model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
data=ETTh1
root_path=./datasets/
num_workers=6
e_layers=8
factor=3
d_model=1024
d_ff=2048
n_heads=8
features=M
batch_size=64
# ────────────────────────────────────────────────────────────────────────────

# Output identifiers
model_id=ettm1_token
out_dir=./results/token_hsic_ETTm2
seed=42

torchrun --nnodes=1 --nproc_per_node=5 \
  experiments/etth1_token_hsic_mi_analysis.py \
  --ckpt_path "$ckpt_path" \
  --root_path "$root_path" \
  --data_path "${data}.csv" \
  --data "$data" \
  --features "$features" \
  --seq_len "$seq_len" \
  --label_len "$label_len" \
  --pred_len "$pred_len" \
  --output_len "$output_len" \
  --patch_len "$patch_len" \
  --e_layers "$e_layers" \
  --factor "$factor" \
  --d_model "$d_model" \
  --d_ff "$d_ff" \
  --n_heads "$n_heads" \
  --use_ims \
  --use_multi_gpu \
  --batch_size "$batch_size" \
  --num_workers "$num_workers" \
  --out_dir "$out_dir" \
  --model_id "$model_id" \
  --seed "$seed" \
  --use_stable_rank_sigma
