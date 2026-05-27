#!/bin/sh
#
# Grid search over init scalars a,b for GeoHPE (same as ETTh1_geometric_hpe.sh; ab_fixed=1 → fixed during finetune):
#   a_init = --geometric_hpe_linear_k_a_init   (patch: v *= 1 + a*k)
#   b_init = --geometric_hpe_pe_curv_b_init    (PE: PE *= 1 + b*k_norm)
#
# Each run: finetune + test; MSE/MAE read from FORECAST_TEST_METRICS_JSON (rank 0).
# Results appended to CSV; best row printed at end.
#
# Usage (repo root = Large-Time-Series-Model):
#   bash scripts/forecast/ETTh1_geometric_hpe_grid_ab.sh
#
# Override grids (space-separated floats):
#   GRID_A="0.1 0.2 0.3" GRID_B="0.4 0.5" bash scripts/forecast/ETTh1_geometric_hpe_grid_ab.sh
#
# Optional env:
#   CUDA_VISIBLE_DEVICES, CKPT_PATH, DATA, OUT_DIR, GRID_CSV, FINETUNE_EPOCHS, BATCH_SIZE, LR, SEED
#
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

# Default search grids: a,b from 0.1 to 0.9 step 0.2 (5x5 = 25 runs unless overridden)
GRID_A="${GRID_A:-1 2 3 4 5}"
GRID_B="${GRID_B:-1 2 3 4 5}"

OUT_DIR="${OUT_DIR:-./outputs/geo_hpe_ab_grid}"
GRID_CSV="${GRID_CSV:-$OUT_DIR/grid_ab_mse.csv}"
mkdir -p "$OUT_DIR"

CKPT_PATH="${CKPT_PATH:-checkpoints/Timer_forecast_1.0.ckpt}"
DATA="${DATA:-ETTh1}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LR="${LR:-${LEARNING_RATE:-3e-5}}"
SEED="${SEED:-1}"

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96

if [ ! -f "$CKPT_PATH" ]; then
  echo "Checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi
if ! [ -f "./datasets/${DATA}.csv" ]; then
  echo "Dataset not found: ./datasets/${DATA}.csv" >&2
  exit 1
fi

if [ ! -f "$GRID_CSV" ]; then
  echo "a_init,b_init,mse,mae,metrics_json,status" > "$GRID_CSV"
fi

_run_one() {
  _a="$1"
  _b="$2"
  _atag=$(printf '%s' "$_a" | tr '.' 'p')
  _btag=$(printf '%s' "$_b" | tr '.' 'p')
  _mid="etth1_geohpe_ab_${_atag}_${_btag}"
  _met="${OUT_DIR}/metrics_a${_atag}_b${_btag}.json"
  export FORECAST_TEST_METRICS_JSON="$(pwd)/${_met}"

  echo ">>> grid run: a_init=${_a} b_init=${_b} model_id=${_mid}" >&2

  set +e
  torchrun --nnodes=1 --nproc_per_node=2 run.py \
    --task_name forecast \
    --is_finetuning 1\
    --is_training 1 \
    --seed "$SEED" \
    --ckpt_path "$CKPT_PATH" \
    --root_path ./datasets/ \
    --data_path "${DATA}.csv" \
    --data "$DATA" \
    --model_id "$_mid" \
    --model "$model_name" \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --output_len $output_len \
    --e_layers 8 \
    --factor 3 \
    --des "GeoHPE-ab${_atag}-b${_btag}" \
    --d_model 1024 \
    --d_ff 2048 \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --num_workers 4 \
    --patch_len $patch_len \
    --finetune_epochs "$FINETUNE_EPOCHS" \
    --geometric_hpe 1 \
    --geometric_hpe_periods 24,168 \
    --geometric_hpe_curv_phase_scale 1.0 \
    --geometric_hpe_curv_residual 0 \
    --geometric_hpe_res_lambda 0 \
    --geometric_hpe_linear_k_patch_scale 1 \
    --geometric_hpe_linear_k_a_init "$_a" \
    --geometric_hpe_pe_curv_weighted 1 \
    --geometric_hpe_pe_curv_b_init "$_b" \
    --geometric_hpe_ab_fixed 1 \
    --train_test 0 \
    --subset_rand_ratio 1 \
    --itr 1 \
    --gpu 0 \
    --use_ims \
    --use_multi_gpu
  _rc=$?
  set -e

  _mse="nan"
  _mae="nan"
  _st="ok"
  if [ "$_rc" -ne 0 ]; then
    _st="fail_rc${_rc}"
  elif [ -f "$_met" ]; then
    _parsed=$(python3 -c "
import json, sys
p = sys.argv[1]
try:
    with open(p) as f:
        j = json.load(f)
    r = j[0]
    print(r['mse'], r['mae'], flush=True)
except Exception:
    sys.exit(1)
" "$_met" 2>/dev/null) || _parsed=""
    if [ -n "$_parsed" ]; then
      # shellcheck disable=SC2086
      set -- $_parsed
      _mse="$1"
      _mae="$2"
    else
      _st="fail_metrics_parse"
    fi
  else
    _st="fail_no_json"
  fi

  echo "${_a},${_b},${_mse},${_mae},${_met},${_st}" >> "$GRID_CSV"
  echo "    -> mse=${_mse} mae=${_mae} status=${_st}" >&2
}

for a in $GRID_A; do
  for b in $GRID_B; do
    _run_one "$a" "$b"
  done
done

echo "" >&2
echo "Wrote grid results: $GRID_CSV" >&2
echo "Best MSE row (status=ok, min mse in column 3):" >&2
if command -v sort >/dev/null 2>&1; then
  tail -n +2 "$GRID_CSV" | awk -F, '$6=="ok" && $3!="nan" {print}' | sort -t, -k3 -g 2>/dev/null | head -1 >&2 || true
fi
