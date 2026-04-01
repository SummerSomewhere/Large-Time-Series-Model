#!/bin/sh
# ETTh1 test: baseline vs lightweight representation recycle (RR).
#
# RR: at encoder layers 0 and 7 (defaults), patches 3 and 6, two rounds per patch per layer:
#   h <- h + alpha_k * (TF_l(h) - h) then F.layer_norm, for k in {0.4, 0.2}.
#
# Mimics IMS + sl672 / patch 96 / e_layers=8.
#
# 中文: 基线 vs 第1+第7层、patch 3/6、两轮权重 0.4 与 0.2；打印 MSE/MAE 差。
#
# Usage:
#   bash scripts/forecast/ETTh1_mi_recycle_ablation.sh
#   RECYCLE_LAYERS=0,7 RECYCLE_ROUND_ALPHAS=0.4,0.2 RECYCLE_PATCHES=3,6 bash scripts/forecast/ETTh1_mi_recycle_ablation.sh
#
# HSIC_NPY: per-layer peak patches from hsic_mean.npy rows; still uses RECYCLE_LAYERS for which layers.
#
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

CKPT="${CKPT:-checkpoints/forecast_etth1_sr_1_Timer_ETTh1_ftM_sl672_ll576_pl96_pl96_dm1024_nh8_el8_dl1_df2048_fc3_ebtimeF_dtTrue_Exp26-03-30_16-57-39/checkpoint.pth}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/mi_recycle_ablation}"
RECYCLE_LAYERS="${RECYCLE_LAYERS:-0,7}"
RECYCLE_PATCHES="${RECYCLE_PATCHES:-6}"
RECYCLE_ROUND_ALPHAS="${RECYCLE_ROUND_ALPHAS:-0.4,0.2}"
HSIC_NPY="${HSIC_NPY:-}"
PEAK_MODE="${PEAK_MODE:-tukey}"

mkdir -p "$OUT_DIR"

BASE_ARGS="--task_name forecast --is_training 0 --is_finetuning 0 --model_id etth1_mi_recycle_eval --model Timer --features M --seq_len 672 --label_len 576 --pred_len 96 --output_len 96 --e_layers 8 --factor 3 --des mi_recycle_ablation --d_model 1024 --d_ff 2048 --n_heads 8 --dropout 0.1 --embed timeF --freq h --activation gelu --batch_size 32 --patch_len 96 --use_ims --data ETTh1 --root_path ./datasets/ETT-small/ --data_path ETTh1.csv --gpu 0 --num_workers 4 --itr 1"

echo "=== Baseline (no recycle) ==="
export FORECAST_TEST_METRICS_JSON="$OUT_DIR/baseline_metrics.json"
"$PYTHON" run.py $BASE_ARGS --ckpt_path "$CKPT" \
  --recycle_encoder_layer -1 --recycle_encoder_layers "" --recycle_patch_indices "" --recycle_round_alphas ""

echo "=== RR layers=${RECYCLE_LAYERS} patches=${RECYCLE_PATCHES} rounds=${RECYCLE_ROUND_ALPHAS} ==="
export FORECAST_TEST_METRICS_JSON="$OUT_DIR/recycle_metrics.json"
if [ -n "$HSIC_NPY" ] && [ -f "$HSIC_NPY" ]; then
  echo "Peaks from HSIC_NPY=$HSIC_NPY (PEAK_MODE=$PEAK_MODE)"
  "$PYTHON" run.py $BASE_ARGS --ckpt_path "$CKPT" \
    --recycle_encoder_layer -1 \
    --recycle_encoder_layers "$RECYCLE_LAYERS" \
    --recycle_hsic_mean_npy "$HSIC_NPY" \
    --recycle_peak_mode "$PEAK_MODE" \
    --recycle_round_alphas "$RECYCLE_ROUND_ALPHAS"
else
  "$PYTHON" run.py $BASE_ARGS --ckpt_path "$CKPT" \
    --recycle_encoder_layer -1 \
    --recycle_encoder_layers "$RECYCLE_LAYERS" \
    --recycle_patch_indices "$RECYCLE_PATCHES" \
    --recycle_round_alphas "$RECYCLE_ROUND_ALPHAS"
fi

echo "=== Comparison (ETTh1 test MSE / MAE) ==="
"$PYTHON" - "$OUT_DIR" << 'PY'
import json
import pathlib
import sys

d = pathlib.Path(sys.argv[1])
for name in ("baseline_metrics.json", "recycle_metrics.json"):
    p = d / name
    if not p.is_file():
        print(f"Missing {p}", file=sys.stderr)
        sys.exit(1)

def load(p):
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return None
    row = data[0]
    return float(row["mse"]), float(row["mae"]), int(row.get("output_len", 0))

b_mse, b_mae, ol = load(d / "baseline_metrics.json")
r_mse, r_mae, ol2 = load(d / "recycle_metrics.json")
print(f"output_len={ol or ol2}")
print(f"baseline  MSE={b_mse:.6f}  MAE={b_mae:.6f}")
print(f"recycle   MSE={r_mse:.6f}  MAE={r_mae:.6f}")
print(f"delta     MSE={r_mse - b_mse:+.6f}  MAE={r_mae - b_mae:+.6f}")
PY

echo "Metrics: $OUT_DIR/baseline_metrics.json , $OUT_DIR/recycle_metrics.json"
