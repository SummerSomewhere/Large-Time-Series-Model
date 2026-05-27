#!/bin/bash
#
# Token-Level L2 Norm vs HSIC Analysis on Synthetic Concepts (Timer)
#
# Investigates whether high-HSIC token positions have different L2 norm
# distributions than low-HSIC positions in Timer representations.
#
# Steps:
#   1. Load synthetic concept dataset
#   2. Extract per-layer per-token hidden states via Timer
#   3. Compute HSIC per token position (token vs concept param)
#   4. Split tokens by top/bottom K HSIC → compare norm distributions
#   5. Mann-Whitney U test + Cliff's Delta effect size
#
# Usage:
#   bash probe/ts_concept_norm_mi_pipeline.sh
#
#   Or with custom parameters:
#   SEQ_LEN=512 PATCH_LEN=96 MAX_SAMPLES=1024 TOP_K=4 \
#   bash probe/ts_concept_norm_mi_pipeline.sh
#

set -e

_ROOT="$(cd "$(dirname "$0")/.." && pwd)" && cd "$_ROOT" || exit 1

# ── Configuration ────────────────────────────────────────────────────────────────
CKPT_PATH="${CKPT_PATH:-checkpoints/Timer_forecast_1.0.ckpt}"
DATASET_PATH="${DATASET_PATH:-./results/synthetic/concepts_dataset.pt}"

SEQ_LEN="${SEQ_LEN:-512}"
PATCH_LEN="${PATCH_LEN:-96}"
STRIDE="${STRIDE:-96}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_SAMPLES="${MAX_SAMPLES:-1024}"
TOP_K="${TOP_K:-4}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/synthetic/norm_mi_analysis/}"

DEVICE="${DEVICE:-cuda}"
if [ "$DEVICE" = "cuda" ] && ! command -v nvidia-smi &> /dev/null; then
    DEVICE="cpu"
fi

# ── Run ────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "[Norm vs HSIC Analysis] Timer + Synthetic Concepts"
echo "============================================================"
echo "  CKPT_PATH   : $CKPT_PATH"
echo "  DATASET_PATH: $DATASET_PATH"
echo "  SEQ_LEN     : $SEQ_LEN"
echo "  PATCH_LEN   : $PATCH_LEN"
echo "  STRIDE      : $STRIDE"
echo "  BATCH_SIZE  : $BATCH_SIZE"
echo "  MAX_SAMPLES : $MAX_SAMPLES"
echo "  TOP_K       : $TOP_K"
echo "  SEED        : $SEED"
echo "  DEVICE      : $DEVICE"
echo "  OUTPUT_DIR  : $OUTPUT_DIR"
echo "============================================================"

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$PIPELINE_DIR/ts_concept_token_norm_mi_analysis.py" \
    --ckpt_path "$CKPT_PATH" \
    --dataset_path "$DATASET_PATH" \
    --seq_len "$SEQ_LEN" \
    --patch_len "$PATCH_LEN" \
    --stride "$STRIDE" \
    --batch_size "$BATCH_SIZE" \
    --max_samples "$MAX_SAMPLES" \
    --top_k "$TOP_K" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/run_log.txt"

echo ""
echo "============================================================"
echo "Done. Results: $OUTPUT_DIR/"
echo "============================================================"
