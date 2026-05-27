#!/usr/bin/env bash
# =============================================================================
# MI Token Analysis Pipeline
# Three modes:
#   [Default]  High vs Low MI token probe across layers
#   --within_layer   Within-layer percentile analysis (top/bottom N% tokens)
#   --feature_entropy Matrix-based feature entropy per layer (Layer-by-Layer paper)
#
# This script runs the complete pipeline:
#   1. Generate synthetic concept dataset (Step 1, if not exists)
#   2. Extract representations + token-level hidden states via Timer (Step 2)
#   3. Run analysis (Step 4, mode controlled by flags)
#
# Usage:
#   bash probe/ts_concept_mi_token_pipeline.sh
#   bash probe/ts_concept_mi_token_pipeline.sh --top_k 6 --probe_epochs 300
#   bash probe/ts_concept_mi_token_pipeline.sh --within_layer --percentile 10
#   bash probe/ts_concept_mi_token_pipeline.sh --feature_entropy --entropy_alpha 1.0
# =============================================================================

set -e

# ── Default config ────────────────────────────────────────────────────────────
CKPT_PATH="${CKPT_PATH:-checkpoints/Timer_forecast_1.0.ckpt}"
DATASET_DIR="./results/synthetic/"
DATASET_PATH="${DATASET_DIR}concepts_dataset.pt"
REPR_DIR="${DATASET_DIR}representations/"
LP_DIR="${DATASET_DIR}mi_token_analysis/"
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Extraction config (Timer defaults)
SEQ_LEN="${SEQ_LEN:-512}"
PATCH_LEN="${PATCH_LEN:-96}"
STRIDE="${STRIDE:-96}"
BATCH_SIZE="${BATCH_SIZE:-256}"
POOLING_MODE="mean"
DEVICE="${DEVICE:-cuda}"

# Probe config
TOP_K="${TOP_K:-1}"
PERCENTILE="${PERCENTILE:-10.0}"
WITHIN_LAYER="${WITHIN_LAYER:-0}"
FEATURE_ENTROPY="${FEATURE_ENTROPY:-0}"
ENTROPY_ALPHA="${ENTROPY_ALPHA:-1.0}"
PROBE_EPOCHS="${PROBE_EPOCHS:-100}"
PROBE_LR="${PROBE_LR:-1e-3}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-700}"   # 7 concepts × 100 samples

# ── Parse optional args ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --top_k)           TOP_K="$2";        shift 2 ;;
        --percentile)      PERCENTILE="$2";   shift 2 ;;
        --within_layer)    WITHIN_LAYER=1;    shift   ;;
        --feature_entropy)  FEATURE_ENTROPY=1; shift   ;;
        --entropy_alpha)   ENTROPY_ALPHA="$2"; shift 2 ;;
        --probe_epochs)     PROBE_EPOCHS="$2"; shift 2 ;;
        --probe_lr)        PROBE_LR="$2";     shift 2 ;;
        --seed)            SEED="$2";          shift 2 ;;
        --device)          DEVICE="$2";        shift 2 ;;
        --ckpt_path)       CKPT_PATH="$2";     shift 2 ;;
        --max_samples)     MAX_SAMPLES="$2";   shift 2 ;;
        --seq_len)         SEQ_LEN="$2";       shift 2 ;;
        --patch_len)       PATCH_LEN="$2";    shift 2 ;;
        --stride)          STRIDE="$2";        shift 2 ;;
        --batch_size)      BATCH_SIZE="$2";    shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ── Output dirs ───────────────────────────────────────────────────────────────
mkdir -p "$DATASET_DIR"
mkdir -p "$REPR_DIR"
mkdir -p "$LP_DIR"

echo "============================================================"
echo "MI Token Analysis Pipeline (High vs Low MI Token Probe)"
echo "Model: Timer"
echo "============================================================"
echo "  CKPT_PATH    : $CKPT_PATH"
echo "  DATASET_PATH : $DATASET_PATH"
echo "  REPR_DIR     : $REPR_DIR"
echo "  LP_DIR       : $LP_DIR"
echo "  SEQ_LEN      : $SEQ_LEN"
echo "  PATCH_LEN    : $PATCH_LEN"
echo "  TOP_K          : $TOP_K"
echo "  PERCENTILE     : $PERCENTILE"
echo "  WITHIN_LAYER   : $WITHIN_LAYER"
echo "  FEATURE_ENTROPY : $FEATURE_ENTROPY"
echo "  ENTROPY_ALPHA   : $ENTROPY_ALPHA"
echo "  PROBE_EPOCHS   : $PROBE_EPOCHS"
echo "  PROBE_LR       : $PROBE_LR"
echo "  SEED         : $SEED"
echo "  DEVICE       : $DEVICE"
echo "============================================================"

# ── Step 1: Generate synthetic dataset ───────────────────────────────────────
if [[ ! -f "$DATASET_PATH" ]]; then
    echo ""
    echo "============================================================"
    echo "[Step 1] Generating synthetic concept dataset"
    echo "============================================================"
    python3 "$PIPELINE_DIR/ts_concept_synthetic_dataset.py" \
        --n_samples 100 \
        --seq_len $SEQ_LEN \
        --seed "$SEED" \
        --output_dir "$DATASET_DIR"
else
    echo ""
    echo "[Step 1] Dataset already exists: $DATASET_PATH, skipping."
fi

# ── Step 2: Extract representations + token states ────────────────────────────
echo ""
echo "============================================================"
echo "[Step 2] Extracting Timer representations + token-level hidden states"
echo "============================================================"

EXTRACT_CMD="python3 $PIPELINE_DIR/ts_concept_representation_extraction.py \
    --ckpt_path '$CKPT_PATH' \
    --dataset_path '$DATASET_PATH' \
    --output_dir '$REPR_DIR' \
    --seq_len $SEQ_LEN \
    --patch_len $PATCH_LEN \
    --stride $STRIDE \
    --batch_size $BATCH_SIZE \
    --pooling_mode '$POOLING_MODE' \
    --device $DEVICE \
    --seed $SEED \
    --save_token_reps"

if [[ -n "$MAX_SAMPLES" ]]; then
    EXTRACT_CMD="$EXTRACT_CMD --max_samples $MAX_SAMPLES"
fi

eval "$EXTRACT_CMD"

# ── Step 4: MI Token Analysis ───────────────────────────────────────────────
echo ""
echo "============================================================"
if [[ "$WITHIN_LAYER" == "1" ]]; then
    echo "[Step 4] Within-Layer Percentile Analysis (top/bottom ${PERCENTILE}% tokens)"
elif [[ "$FEATURE_ENTROPY" == "1" ]]; then
    echo "[Step 4] Feature Entropy Analysis (Layer-by-Layer paper)"
else
    echo "[Step 4] MI Token Analysis (High vs Low MI Token Probe)"
fi
echo "============================================================"

ANALYSIS_CMD="python3 $PIPELINE_DIR/ts_concept_mi_token_analysis.py \
    --rep_dir '$REPR_DIR' \
    --output_dir '$LP_DIR' \
    --probe_epochs $PROBE_EPOCHS \
    --probe_lr $PROBE_LR \
    --seed $SEED \
    --device $DEVICE"

if [[ "$WITHIN_LAYER" == "1" ]]; then
    ANALYSIS_CMD="$ANALYSIS_CMD --within_layer --percentile $PERCENTILE"
elif [[ "$FEATURE_ENTROPY" == "1" ]]; then
    ANALYSIS_CMD="$ANALYSIS_CMD --feature_entropy --entropy_alpha $ENTROPY_ALPHA"
else
    ANALYSIS_CMD="$ANALYSIS_CMD --top_k $TOP_K"
fi

if [[ -n "$MAX_SAMPLES" ]]; then
    ANALYSIS_CMD="$ANALYSIS_CMD --max_samples $MAX_SAMPLES"
fi

eval "$ANALYSIS_CMD"

echo ""
echo "============================================================"
echo "All steps complete!"
echo "  Output dir: $LP_DIR"
echo "============================================================"
