#!/bin/sh
#
# ETTh1: high-MI vs low-MI patch STFT spectrum comparison (scipy.signal.stft).
# Reuses the same MI source as scripts/analysis/mi_patch_curvature_etth1.sh (--hsic_npy or --ckpt_path).
#
# Usage (from repo root):
#   bash scripts/analysis/mi_patch_stft_etth1.sh
#
# Env overrides:
#   CSV, OUT, T_START, SEQ_LEN, PATCH_LEN (default 24 — match finetuned Timer / HSIC npy),
#   HSIC_NPY (must be [L,P] with P = n_patches for this SEQ_LEN/PATCH_LEN — patch count changes
#     with patch_len; old npy from patch_len=96 is invalid for PATCH_LEN=24),
#   CKPT (live HSIC: always matches current --patch_len / Timer),
#   MI_GROUP_MODE: symmetric | high_vs_rest (default high_vs_rest: high MI = at/above HIGH_Q
#     percentile; roughly top (1-HIGH_Q) fraction of patches; all other finite patches = low MI),
#   LAYERS (default "all" = every encoder layer; e.g. "0,1,7" or "-1" for last layer only),
#   USE_FULL_DATASET (0/1: aggregate many windows — MI mean + STFT mean),
#   FULL_DATASET_MODE: exp_split (default) = same t0 grid as exp_forecast use_ims
#     (CIAutoRegressionDatasetBenchmark); sliding = full-CSV hop grid (WINDOW_HOP),
#   DATASET_FLAG: train | val | test (for exp_split; default test, matches data_provider),
#   DATA (default ETTh1), LOADER_STRIDE (default 1, exp_forecast --stride), SUBSET_RAND_RATIO,
#   WRITE_PER_WINDOW (0/1: with USE_FULL_DATASET=1, also write per-t0 figures — slow),
#   NUM_SAMPLES, WINDOW_HOP (only for sliding mode or non-full mode; default seq_len),
#   AGGREGATE_WINDOWS (0/1: only for non-full mode; mean STFT across num_samples windows),
#   HIGH_Q (default 0.75 — high-MI cutoff), LOW_Q (only for symmetric mode; default 0.25)
#

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7,8}"

CSV="${CSV:-./datasets/ETTh1.csv}"
OUT="${OUT:-./figures/mi_stft_etth1}"
SEQ_LEN="${SEQ_LEN:-672}"
PATCH_LEN="${PATCH_LEN:-24}"
T_START="${T_START:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
WINDOW_HOP="${WINDOW_HOP:-$SEQ_LEN}"

CKPT="${CKPT:-checkpoints/forecast_etth1_sr_1_Timer_ETTh1_ftM_sl672_ll576_pl96_pl24_dm1024_nh8_el8_dl1_df2048_fc3_ebtimeF_dtTrue_Exp26-04-07_12-53-00/checkpoint.pth}"
HSIC_NPY="${HSIC_NPY:-}"
HSIC_SEARCH_ROOT="${HSIC_SEARCH_ROOT:-./outputs/mi_hsic_timer}"
TARGET="${TARGET:-OT}"
LAYERS="${LAYERS:-all}"
USE_FULL_DATASET="${USE_FULL_DATASET:-1}"
FULL_DATASET_MODE="${FULL_DATASET_MODE:-exp_split}"
DATASET_FLAG="${DATASET_FLAG:-test}"
DATA="${DATA:-ETTh1}"
LOADER_STRIDE="${LOADER_STRIDE:-1}"
SUBSET_RAND_RATIO="${SUBSET_RAND_RATIO:-1}"
WRITE_PER_WINDOW="${WRITE_PER_WINDOW:-0}"
AGGREGATE_WINDOWS="${AGGREGATE_WINDOWS:-0}"
HIGH_Q="${HIGH_Q:-0.75}"
LOW_Q="${LOW_Q:-0.25}"
MI_GROUP_MODE="${MI_GROUP_MODE:-high_vs_rest}"
STFT_NPERSEG="${STFT_NPERSEG:-64}"

LABEL_LEN="${LABEL_LEN:-576}"
PRED_LEN="${PRED_LEN:-96}"
OUTPUT_LEN="${OUTPUT_LEN:-96}"
TIMER_FEATURES="${TIMER_FEATURES:-M}"

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

if ! [ -f "$CSV" ]; then
  echo "CSV not found: $CSV" >&2
  exit 1
fi

_resolve_newest_hsic() {
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
}

if [ -z "$HSIC_NPY" ] || [ ! -f "$HSIC_NPY" ]; then
  HSIC_NPY="$(_resolve_newest_hsic)"
fi

# Pass both npy and ckpt when available: Python uses npy only if P matches n_patches;
# otherwise falls back to live HSIC from CKPT (e.g. old aggregate with P=7 vs patch_len=24 → P=28).
EXTRA=""
if [ -f "$CKPT" ]; then
  CKPT_EXTRA="--ckpt_path $CKPT --gpu 0"
else
  CKPT_EXTRA=""
fi
if [ -n "$HSIC_NPY" ] && [ -f "$HSIC_NPY" ]; then
  echo "HSIC_NPY=$HSIC_NPY (if P matches grid, used; else CKPT fallback)" >&2
  EXTRA="--hsic_npy $HSIC_NPY $CKPT_EXTRA"
elif [ -n "$CKPT_EXTRA" ]; then
  echo "Using live Timer HSIC with CKPT=$CKPT" >&2
  EXTRA="$CKPT_EXTRA"
else
  echo "ERROR: Set HSIC_NPY or CKPT, or run mi_hsic_layerwise_timer first." >&2
  exit 1
fi

mkdir -p "$OUT"

python -u experiments/mi_patch_stft_compare.py \
  --csv "$CSV" \
  --target_col "$TARGET" \
  --t_start "$T_START" \
  --num_samples "$NUM_SAMPLES" \
  --window_hop "$WINDOW_HOP" \
  --seq_len "$SEQ_LEN" \
  --patch_len "$PATCH_LEN" \
  --label_len "$LABEL_LEN" \
  --pred_len "$PRED_LEN" \
  --output_len "$OUTPUT_LEN" \
  --use_ims 1 \
  --timer_patch_len "$PATCH_LEN" \
  --timer_features "$TIMER_FEATURES" \
  --layers "$LAYERS" \
  --use_full_dataset "$USE_FULL_DATASET" \
  --full_dataset_mode "$FULL_DATASET_MODE" \
  --dataset_flag "$DATASET_FLAG" \
  --data "$DATA" \
  --loader_stride "$LOADER_STRIDE" \
  --subset_rand_ratio "$SUBSET_RAND_RATIO" \
  --write_per_window "$WRITE_PER_WINDOW" \
  --aggregate_windows "$AGGREGATE_WINDOWS" \
  --out_dir "$OUT" \
  --mi_group_mode "$MI_GROUP_MODE" \
  --high_quantile "$HIGH_Q" \
  --low_quantile "$LOW_Q" \
  --stft_nperseg "$STFT_NPERSEG" \
  $EXTRA

echo "Done. Figures under $OUT" >&2
