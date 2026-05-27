#!/bin/sh
#
# ETTh1：一条流水线 = 准备 HSIC/MI 源 → 单次 Python 调用（多窗口：MI→映射到 patch→画图/CSV）。
# 对应 experiments/mi_patch_curvature_analysis.py 的 --num_samples / --window_hop。
#
# 用法（仓库根目录）：
#   bash scripts/analysis/mi_patch_curvature_etth1.sh
#
# MI 来源（与旧版一致）：
#   MI_MODE=auto（默认）：若 CKPT 存在则 live Timer+HSIC；否则用 HSIC_NPY 或 HSIC_SEARCH_ROOT 下最新的
#   aggregate/hsic_mean.npy；仍没有且 AUTO_RUN_HSIC=1 且默认 ckpt 存在则先跑 mi_hsic_layerwise_timer.sh。
#
# 覆盖示例：
#   CKPT=/path/to/model.ckpt bash scripts/analysis/mi_patch_curvature_etth1.sh
#   MI_MODE=npy HSIC_NPY=.../hsic_mean.npy bash ...
#   AUTO_RUN_HSIC=0 bash ...   # 不自动跑 layerwise
#

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

CSV="${CSV:-./datasets/ETTh1.csv}"
OUT="${OUT:-./figures/mi_patch_curvature_etth1}"
SEQ_LEN="${SEQ_LEN:-672}"
PATCH_LEN="${PATCH_LEN:-96}"
TARGET="${TARGET:-OT}"
NUM_SAMPLES="${NUM_SAMPLES:-25}"
WINDOW_HOP="${WINDOW_HOP:-$SEQ_LEN}"
T_START="${T_START:-0}"

CKPT="${CKPT:-checkpoints/Timer_forecast_1.0.ckpt}"
LABEL_LEN="${LABEL_LEN:-576}"
PRED_LEN="${PRED_LEN:-96}"
OUTPUT_LEN="${OUTPUT_LEN:-96}"
TIMER_FEATURES="${TIMER_FEATURES:-M}"

MI_MODE="${MI_MODE:-auto}"
HSIC_NPY="${HSIC_NPY:-}"
HSIC_SEARCH_ROOT="${HSIC_SEARCH_ROOT:-./outputs/mi_hsic_timer}"
DATA="${DATA:-ETTh1}"
AUTO_RUN_HSIC="${AUTO_RUN_HSIC:-1}"
GPU_ID="${GPU_ID:-4}"
LAYER_ROW="${LAYER_ROW:-0}"
TOP_K="${TOP_K:-2}"

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

if ! [ -f "$CSV" ]; then
  echo "CSV not found: $CSV" >&2
  exit 1
fi

N_ROWS=$(python -c "import pandas as pd; import sys; print(len(pd.read_csv(sys.argv[1])))" "$CSV")
if [ "$N_ROWS" -lt "$SEQ_LEN" ]; then
  echo "CSV has $N_ROWS rows but SEQ_LEN=$SEQ_LEN; cannot form one window." >&2
  exit 1
fi
N_WIN_MAX=$(( (N_ROWS - SEQ_LEN) / WINDOW_HOP + 1 ))
if [ "$NUM_SAMPLES" -gt "$N_WIN_MAX" ]; then
  echo "NUM_SAMPLES=$NUM_SAMPLES capped to $N_WIN_MAX (series length $N_ROWS, SEQ_LEN=$SEQ_LEN, WINDOW_HOP=$WINDOW_HOP)." >&2
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

BASE_ARGS="--csv $CSV --target_col $TARGET --seq_len $SEQ_LEN --patch_len $PATCH_LEN --out_dir $OUT --top_k $TOP_K --layer_row $LAYER_ROW --t_start $T_START --num_samples $RUN_SAMPLES --window_hop $WINDOW_HOP"

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

_try_resolve_hsic_npy() {
  if [ -n "$HSIC_NPY" ] && [ -f "$HSIC_NPY" ]; then
    return 0
  fi
  if [ -d "$HSIC_SEARCH_ROOT" ]; then
    HSIC_RESOLVED=$(_resolve_newest_hsic)
    if [ -n "$HSIC_RESOLVED" ] && [ -f "$HSIC_RESOLVED" ]; then
      HSIC_NPY=$HSIC_RESOLVED
      echo "Using HSIC_NPY (newest under $HSIC_SEARCH_ROOT): $HSIC_NPY" >&2
    fi
  fi
}

_run_layerwise_hsic() {
  _lw_ckpt="${1:-checkpoints/Timer_forecast_1.0.ckpt}"
  if [ ! -f "$_lw_ckpt" ]; then
    return 1
  fi
  _root=$(dirname "$CSV")
  _csv=$(basename "$CSV")
  echo ">>> [auto] Running mi_hsic_layerwise_timer.sh → aggregate/hsic_mean.npy (CKPT=$_lw_ckpt)..." >&2
  GPU_ID="$GPU_ID" \
    CKPT="$_lw_ckpt" \
    DATA="$DATA" \
    ROOT="$_root" \
    CSV="$_csv" \
    SEQ_LEN="$SEQ_LEN" \
    LABEL_LEN="$LABEL_LEN" \
    PRED_LEN="$PRED_LEN" \
    OUTPUT_LEN="$OUTPUT_LEN" \
    PATCH_LEN="$PATCH_LEN" \
    bash scripts/mi_hsic_layerwise_timer.sh
  return 0
}

USE_LIVE=0
if [ "$MI_MODE" = "npy" ]; then
  USE_LIVE=0
elif [ "$MI_MODE" = "live" ]; then
  if [ ! -f "$CKPT" ]; then
    echo "MI_MODE=live requires an existing CKPT file: $CKPT" >&2
    exit 1
  fi
  USE_LIVE=1
else
  if [ -f "$CKPT" ]; then
    USE_LIVE=1
  fi
fi

if [ "$USE_LIVE" = "1" ]; then
  BASE_ARGS="$BASE_ARGS --ckpt_path $CKPT --label_len $LABEL_LEN --pred_len $PRED_LEN --output_len $OUTPUT_LEN --use_ims 1 --timer_patch_len $PATCH_LEN --timer_features $TIMER_FEATURES"
  echo "MI mode: live Timer+HSIC (--ckpt_path $CKPT)" >&2
else
  _try_resolve_hsic_npy
  if [ -z "$HSIC_NPY" ] || [ ! -f "$HSIC_NPY" ]; then
    if [ "$AUTO_RUN_HSIC" = "1" ] && [ -f "checkpoints/Timer_forecast_1.0.ckpt" ]; then
      _run_layerwise_hsic "checkpoints/Timer_forecast_1.0.ckpt"
      HSIC_NPY=""
      _try_resolve_hsic_npy
    fi
  fi
  if [ -z "$HSIC_NPY" ] || [ ! -f "$HSIC_NPY" ]; then
    echo "Could not find HSIC. Options:" >&2
    echo "  - Place checkpoint at checkpoints/Timer_forecast_1.0.ckpt (or set CKPT=...) for live HSIC or auto layerwise." >&2
    echo "  - Or set HSIC_NPY=.../aggregate/hsic_mean.npy, or run scripts/mi_hsic_layerwise_timer.sh first." >&2
    echo "  - Or MI_MODE=npy with HSIC_NPY set." >&2
    exit 1
  fi
  BASE_ARGS="$BASE_ARGS --hsic_npy $HSIC_NPY"
  echo "MI mode: precomputed npy (--hsic_npy $HSIC_NPY)" >&2
fi

echo "One Python run: RUN_SAMPLES=$RUN_SAMPLES WINDOW_HOP=$WINDOW_HOP T_START=$T_START SEQ_LEN=$SEQ_LEN OUT=$OUT" >&2
python -u experiments/mi_patch_curvature_analysis.py $BASE_ARGS

echo "Done. Up to $RUN_SAMPLES windows processed; outputs under: $OUT" >&2
