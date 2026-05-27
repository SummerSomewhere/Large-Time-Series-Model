#!/bin/sh
# Timer attention visualization — ETTh1 真实数据默认（第二个入口脚本）。
# Usage: ./visualize_timer_attention2.sh
# 数据: 默认读 DATA_ROOT/ETTh1.csv（test loader）；正弦合成: USE_SYNTHETIC=1 ./visualize_timer_attention2.sh
# GPU: CUDA_VISIBLE_DEVICES=0,1 ./visualize_timer_attention2.sh
# CKPT: CKPT=path/to.ckpt ./visualize_timer_attention2.sh
# 输出: OUT=./attention_maps_timer_etth1（默认）；与脚本1区分
# IMS: 需 seq_len = label_len + pred_len（此处 6+6=12）
# patch_len 须与权重一致：官方 Timer_forecast_1.0 多为 96，此时请改 seq/label/pred 与 patch_len 与训练一致
# 其它: RUN_ID=... 、见 visualize_timer_attention.py --help

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

CKPT="${CKPT:-checkpoints/Timer_forecast_1.0.ckpt}"
OUT="${OUT:-./attention_maps_timer_etth1}"
DATA_ROOT="${DATA_ROOT:-./datasets/ETT-small/}"

EXTRA_PY=()
if [ -n "$RUN_ID" ]; then
  EXTRA_PY=(--run_id "$RUN_ID")
fi
# Default: real ETTh1 CSV. Synthetic sine only when USE_SYNTHETIC is set (non-empty).
if [ -n "$USE_SYNTHETIC" ]; then
  EXTRA_PY+=(--synthetic_sin --sin_n_vars 1)
fi

exec python visualize_timer_attention.py \
  --ckpt_path "$CKPT" \
  --root_path "$DATA_ROOT" \
  --data_path ETTh1.csv \
  --data ETTh1 \
  --seq_len 672 \
  --label_len 576 \
  --pred_len 96 \
  --output_len 96 \
  --patch_len 96 \
  --d_model 1024 \
  --d_ff 2048 \
  --e_layers 8 \
  --n_heads 8 \
  --factor 3 \
  --output_dir "$OUT" \
  --gpu 0 \
  "${EXTRA_PY[@]}" \
  "$@"
