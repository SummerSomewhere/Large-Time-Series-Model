#!/bin/sh
# Layer-wise HSIC (MI proxy) for Timer on ETTh1/ETTm1 test set (arXiv:2506.02867 style).
#
# IMPORTANT: --patch_len (and seq_len / IMS flags) MUST match the checkpoint.
#   - checkpoints/Timer_forecast_1.0.ckpt uses patch_len=96 (value_embedding [1024,96]).
#   - A patch_len=1 ckpt needs PATCH_LEN=1 and matching sl/ll/pl below.
#
# Usage from repo root:
#   bash scripts/mi_hsic_layerwise_timer.sh
# Override:
#   CKPT=checkpoints/yours.ckpt MI_RUN=2 bash scripts/mi_hsic_layerwise_timer.sh
# Patch-1 preset:
#   PATCH_LEN=1 SEQ_LEN=96 LABEL_LEN=48 bash scripts/mi_hsic_layerwise_timer.sh
# GPU: default physical card 4; override with GPU_ID=0 etc.
# Per-sample: default --max_samples=100; each sample gets layer_XX_mi.png, heatmaps, summary PNGs.
#
set -e
cd "$(dirname "$0")/.."

# Expose only one physical GPU to PyTorch (it becomes cuda:0 inside the process).
GPU_ID="${GPU_ID:-4}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

CKPT="${CKPT:-checkpoints/Timer_forecast_1.0.ckpt}"
MI_RUN="${MI_RUN:-${MI_HSIC_RUN:-1}}"
DATA="${DATA:-ETTh1}"
ROOT="${ROOT:-./datasets/ETT-small/}"
CSV="${CSV:-${DATA}.csv}"

# Defaults aligned with Timer_forecast_1.0.ckpt + ETTh1 IMS periodic-style runs (sl672_ll576_pl96, patch 96).
SEQ_LEN="${SEQ_LEN:-672}"
LABEL_LEN="${LABEL_LEN:-576}"
PRED_LEN="${PRED_LEN:-96}"
OUTPUT_LEN="${OUTPUT_LEN:-96}"
PATCH_LEN="${PATCH_LEN:-96}"

# Set PERIODIC=1 when loading a finetuned checkpoint that has periodic_embedding_branch.
PERIODIC="${PERIODIC:-0}"
_extra_periodic=""
if [ "$PERIODIC" = "1" ]; then
  _extra_periodic="--periodic_embedding_branch 1"
fi

python experiments/mi_hsic_layerwise_timer.py \
  --gpu 0 \
  --ckpt_path "$CKPT" \
  --root_path "$ROOT" \
  --data_path "$CSV" \
  --data "$DATA" \
  --features M \
  --seq_len "$SEQ_LEN" \
  --label_len "$LABEL_LEN" \
  --pred_len "$PRED_LEN" \
  --output_len "$OUTPUT_LEN" \
  --use_ims \
  --patch_len "$PATCH_LEN" \
  --d_model 1024 \
  --d_ff 2048 \
  --e_layers 8 \
  --n_heads 8 \
  --factor 3 \
  --batch_size 32 \
  --max_batches 80 \
  --num_workers 4 \
  --run_id "$MI_RUN" \
  --output_dir ./outputs/mi_hsic_timer \
  $_extra_periodic \
  "$@"
