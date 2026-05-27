#!/bin/bash
#
# Per-token L2 norm analysis grouped by high/low MI patches (Timer + ETTh1).
# Reads global_mi_peaks_etth1.json and produces norm distribution/scatter plots.
#

_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$_ROOT" || exit 1

seq_len=672
label_len=576
pred_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
data=ETTh1
root_path=./datasets/
data_path=${data}.csv
features=M
batch_size=64
num_workers=6
num_layers=8
d_model=1024
d_ff=2048
n_heads=8
dropout=0.1
activation=gelu
embed=timeF
freq=h

mi_file=./global_mi_peaks_etth1.json
out_dir=./results/token_norm_mi_analysis/

python3 experiments/etth1_token_norm_mi_analysis.py \
  --mi_file $mi_file \
  --model_path $ckpt_path \
  --root_path $root_path \
  --data_path $data_path \
  --data $data \
  --features $features \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --patch_len $patch_len \
  --batch_size $batch_size \
  --num_workers $num_workers \
  --num_layers $num_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --dropout $dropout \
  --activation $activation \
  --embed $embed \
  --freq $freq \
  --output_dir $out_dir \
  --seed 42 \
  --device cuda
