#!/bin/bash
#
# Timer Layer-wise Linear Probe + MI Patch Comparison
#
# Pipeline:
#   Step 1: Extract per-layer per-patch features via Timer forward hooks
#           (skipped if --features_pt_path already exists)
#   Step 2: STL decomposition generates 5+ semantic labels from raw time series
#   Step 3: Ridge Regression probing — measures each layer's explanatory power
#   Step 4: Compare R² between high-MI patches vs low-MI patches (requires MI peaks JSON)
#
# Outputs (results/layerwise_probe/):
#   figA_r2_per_label.png         — R² vs layer depth (one line per semantic label)
#   figB_mse_per_label.png        — log(MSE) vs layer depth
#   figC_r2_heatmap.png           — heatmap: layers × labels, colour = R²
#   figD_high_low_mi_compare.png  — R²: high-MI patches vs low-MI patches per layer
#   figE_patch_r2_heatmap.png     — per-patch R² for selected layers
#   ridge_results.json            — full structured results
#

_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$_ROOT" || exit 1

# ── Feature extraction step ──────────────────────────────────────────────────
# Skip if already extracted (set features_pt_path to existing file)
features_pt_path=./results/layerwise_probe_features/ETTh1_20000.pt
labels_json=./results/layerwise_probe_features/ETTh1_20000_labels.json

# ── Model & data defaults (matching etth1_mi_hsic_peaks.sh) ──────────────────
model_name=Timer
seq_len=672
label_len=576
pred_len=96
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
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
root_path=./datasets/
data=ETTh1
data_path=ETTh1.csv

# ── Linear probe parameters ──────────────────────────────────────────────────
n_samples=0          # 0 = use all available samples
test_ratio=0.2      # test set proportion
alpha=1.0           # Ridge regularization strength
mi_peaks_path=./global_mi_peaks_etth1.json   # optional: enables high/low MI comparison
stl_period=24       # seasonal period (24 for hourly ETTh1)
out_dir=./results/layerwise_probe

# ── Step 1: Feature extraction ──────────────────────────────────────────────
if [ -n "$features_pt_path" ] && [ -f "$features_pt_path" ]; then
    echo "[Step 1] Using pre-extracted features: $features_pt_path"
else
    echo "[Step 1] Extracting features (this may take a while)..."
    mkdir -p "$(dirname "$features_pt_path")"
    python3 experiments/etth1_layerwise_probe_features.py \
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
      --n_samples 20000 \
      --stl_period $stl_period \
      --out_dir ./results/layerwise_probe_features/ \
      --device cuda
    features_pt_path=./results/layerwise_probe_features/ETTh1_20000.pt
    labels_json=./results/layerwise_probe_features/ETTh1_20000_labels.json
fi

# ── Step 2: Linear probe + comparison ───────────────────────────────────────
echo ""
echo "[Step 2] Running linear probe + MI patch comparison..."
python3 experiments/etth1_linear_probe_patch_compare.py \
  --features_pt_path "$features_pt_path" \
  --labels_json "$labels_json" \
  --e_layers $e_layers \
  --pred_len $pred_len \
  --n_samples $n_samples \
  --test_ratio $test_ratio \
  --alpha $alpha \
  --mi_peaks_path "$mi_peaks_path" \
  --out_dir $out_dir \
  --seed 42 \
  --device cuda
