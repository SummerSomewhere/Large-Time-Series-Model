#!/bin/sh
#
# Same pipeline as mi_patch_curvature_etth1.sh, but the series is a synthetic sin wave
# written under outputs/synthetic_sin_mi/ (does not use ETTh1 or other dataset CSVs).
#
# Usage:
#   bash scripts/analysis/mi_patch_curvature_sin.sh
#   NUM_SAMPLES=5 SIN_LENGTH=10000 bash scripts/analysis/mi_patch_curvature_sin.sh
# After changing SIN_PERIOD, regenerate CSV: REGENERATE_SIN=1 bash scripts/analysis/mi_patch_curvature_sin.sh
#
# MI must come from HSIC (precomputed npy). Sin CSV is not ETTh test layout — use HSIC_NPY from a Timer run
# (e.g. run_mi_curvature_etth1_computed.sh on ETTh1, then point HSIC_NPY here), or run mi_hsic_layerwise_timer.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

# Default CSV path (generated once unless REGENERATE_SIN=1).
CSV_OUT_DIR="${CSV_OUT_DIR:-./outputs/synthetic_sin_mi}"
CSV="${CSV:-$CSV_OUT_DIR/sin_series.csv}"
OUT="${OUT:-./figures/mi_patch_curvature_sin}"
SEQ_LEN="${SEQ_LEN:-672}"
PATCH_LEN="${PATCH_LEN:-96}"
TARGET="${TARGET:-OT}"
NUM_SAMPLES="${NUM_SAMPLES:-25}"
WINDOW_HOP="${WINDOW_HOP:-$SEQ_LEN}"
HSIC_NPY="${HSIC_NPY:-}"
HSIC_SEARCH_ROOT="${HSIC_SEARCH_ROOT:-./outputs/mi_hsic_timer}"
LAYER_ROW="${LAYER_ROW:-0}"
TOP_K="${TOP_K:-2}"

# Synthetic series: y = sin(2*pi*t/SIN_PERIOD) + Normal(0, SIN_NOISE)
# Smaller SIN_PERIOD => more cycles per patch (e.g. PATCH_LEN=96 and SIN_PERIOD=16 => ~6 cycles/patch).
SIN_LENGTH="${SIN_LENGTH:-20000}"
SIN_PERIOD="${SIN_PERIOD:-16}"
SIN_SEED="${SIN_SEED:-0}"
SIN_NOISE="${SIN_NOISE:-0.0}"
REGENERATE_SIN="${REGENERATE_SIN:-0}"

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

mkdir -p "$CSV_OUT_DIR"

if [ ! -f "$CSV" ] || [ "$REGENERATE_SIN" = "1" ]; then
  echo "Writing synthetic sin CSV -> $CSV (length=$SIN_LENGTH period=$SIN_PERIOD)" >&2
  python -u -c "
import sys
import numpy as np
import pandas as pd

length = int(sys.argv[1])
period = float(sys.argv[2])
seed = int(sys.argv[3])
noise_std = float(sys.argv[4])
out_path = sys.argv[5]
rng = np.random.default_rng(seed)
t = np.arange(length, dtype=np.float64)
# Univariate sin; phase 0, unit amplitude (scale in analysis is arbitrary)
y = np.sin(2.0 * np.pi * t / period)
if noise_std > 0.0:
    y = y + rng.normal(0.0, noise_std, size=length)
df = pd.DataFrame({'date': np.arange(length, dtype=np.int64), 'OT': y})
df.to_csv(out_path, index=False)
print('Wrote', length, 'rows to', out_path, flush=True)
" "$SIN_LENGTH" "$SIN_PERIOD" "$SIN_SEED" "$SIN_NOISE" "$CSV"
fi

if ! [ -f "$CSV" ]; then
  echo "CSV missing: $CSV" >&2
  exit 1
fi

N_ROWS=$(python -c "import pandas as pd; import sys; print(len(pd.read_csv(sys.argv[1])))" "$CSV")
if [ "$N_ROWS" -lt "$SEQ_LEN" ]; then
  echo "Synthetic CSV has $N_ROWS rows but SEQ_LEN=$SEQ_LEN; increase SIN_LENGTH." >&2
  exit 1
fi
N_WIN_MAX=$(( (N_ROWS - SEQ_LEN) / WINDOW_HOP + 1 ))
if [ "$NUM_SAMPLES" -gt "$N_WIN_MAX" ]; then
  echo "NUM_SAMPLES=$NUM_SAMPLES capped to $N_WIN_MAX (N_ROWS=$N_ROWS, SEQ_LEN=$SEQ_LEN, WINDOW_HOP=$WINDOW_HOP)." >&2
fi
RUN_SAMPLES=$NUM_SAMPLES
if [ "$RUN_SAMPLES" -gt "$N_WIN_MAX" ]; then
  RUN_SAMPLES=$N_WIN_MAX
fi
if [ "$RUN_SAMPLES" -lt 1 ]; then
  echo "No valid windows (N_ROWS=$N_ROWS)." >&2
  exit 1
fi

mkdir -p "$OUT"

if [ -n "$HSIC_NPY" ] && [ -f "$HSIC_NPY" ]; then
  :
elif [ -d "$HSIC_SEARCH_ROOT" ]; then
  HSIC_RESOLVED=$(
    python -c "
import os, sys
root = sys.argv[1]
best = None
best_m = -1.0
if not os.path.isdir(root):
    sys.exit(0)
for dirpath, _dirnames, filenames in os.walk(root):
    if os.path.basename(dirpath) != 'aggregate':
        continue
    p = os.path.join(dirpath, 'hsic_mean.npy')
    if os.path.isfile(p):
        m = os.path.getmtime(p)
        if m > best_m:
            best_m, best = m, p
print(best or '')
" "$HSIC_SEARCH_ROOT"
  )
  if [ -n "$HSIC_RESOLVED" ] && [ -f "$HSIC_RESOLVED" ]; then
    HSIC_NPY=$HSIC_RESOLVED
    echo "Using HSIC_NPY (newest under $HSIC_SEARCH_ROOT): $HSIC_NPY" >&2
  fi
fi

BASE_ARGS="--csv $CSV --target_col $TARGET --seq_len $SEQ_LEN --patch_len $PATCH_LEN --out_dir $OUT --top_k $TOP_K --layer_row $LAYER_ROW"

if [ -n "$HSIC_NPY" ] && [ -f "$HSIC_NPY" ]; then
  BASE_ARGS="$BASE_ARGS --hsic_npy $HSIC_NPY"
else
  echo "Set HSIC_NPY to aggregate/hsic_mean.npy (from mi_hsic_layerwise_timer). Sin series cannot use --ckpt_path (needs ETTh test CSV)." >&2
  exit 1
fi

echo "synthetic_sin | CSV=$CSV | RUN_SAMPLES=$RUN_SAMPLES | OUT=$OUT" >&2

i=0
while [ "$i" -lt "$RUN_SAMPLES" ]; do
  T_START=$((i * WINDOW_HOP))
  echo ">>> sample $((i + 1))/$RUN_SAMPLES  t_start=$T_START" >&2
  python -u experiments/mi_patch_curvature_analysis.py $BASE_ARGS --t_start "$T_START" || exit 1
  i=$((i + 1))
done

echo "Done. $RUN_SAMPLES windows under $OUT (data: $CSV)" >&2
