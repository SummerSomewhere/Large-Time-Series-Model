#!/bin/sh
#
# Entropy Analysis for Timer + ETTh1. Collects hidden representations after each
# decoder attention block and computes matrix alpha entropy per layer.
#
# Output:
#   entropy_results.json    — layer-wise entropy values and config
#   entropy_curve.png       — decoder layer entropy vs. layer depth
#   entropy_heatmap.png     — per-sample entropy heatmap (requires repitl)

_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$_ROOT" || exit 1

export CUDA_VISIBLE_DEVICES=2,3,4,5,6

# ----- Timer + ETTh1 defaults (matching etth1_mi_hsic_peaks.sh) -----
model_name=Timer
seq_len=672
pred_len=96
patch_len=96
d_model=1024
d_ff=2048
n_heads=8
dropout=0.1
factor=1
activation=gelu
e_layers=8
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
root_path=./datasets/
num_workers=6
# ------------------------------------------------------------------

alpha=1
normalization=maxEntropy
max_samples=20000
max_batches=
seed=42

out_dir=./results/entropy_etth2

python3 experiments/ts_concept_entropy_analysis.py \
  --ckpt_path $ckpt_path \
  --root_path $root_path \
  --data_path traffic.csv \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --patch_len $patch_len \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --dropout $dropout \
  --factor $factor \
  --activation $activation \
  --e_layers $e_layers \
  --num_workers $num_workers \
  --alpha $alpha \
  --normalization $normalization \
  --max_samples $max_samples \
  --seed $seed \
  --output_dir $out_dir \
  ${max_batches:+--max_batches $max_batches}
