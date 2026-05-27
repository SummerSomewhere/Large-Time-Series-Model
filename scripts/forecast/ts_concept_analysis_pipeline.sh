#!/bin/bash
#
# Synthetic Time Series Concept Analysis Pipeline
#
# This script runs the full 4-step pipeline for analyzing Timer representations
# on synthetic time series concepts:
#
#   Step 1: Generate synthetic time series concepts
#   Step 2: Extract Timer hidden states and pool them
#   Step 3: Train linear probes to predict concept parameters
#   Step 4: Multi-dimensional analysis (CKA, UMAP, PCA, Silhouette)
#
# Usage:
#   bash scripts/forecast/ts_concept_analysis_pipeline.sh
#
#   Or with custom parameters:
#   CKPT_PATH=checkpoints/Timer_forecast_1.0.ckpt \
#   N_SAMPLES=1000 SEQ_LEN=256 \
#   bash scripts/forecast/ts_concept_analysis_pipeline.sh
#
# Each step can be run independently:
#   python3 experiments/ts_concept_synthetic_dataset.py ...
#   python3 experiments/ts_concept_representation_extraction.py ...
#   python3 experiments/ts_concept_linear_probing.py ...
#   python3 experiments/ts_concept_multidim_analysis.py ...
#

set -e

_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$_ROOT" || exit 1

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Timer model checkpoint (use 'random' for uninitialized model)
CKPT_PATH="${CKPT_PATH:-checkpoints/Timer_forecast_1.0.ckpt}"

# Synthetic dataset parameters
N_SAMPLES="${N_SAMPLES:-1000}"
SEQ_LEN="${SEQ_LEN:-256}"
SEED="${SEED:-42}"

# Timer model parameters (must match checkpoint)
D_MODEL="${D_MODEL:-1024}"
D_FF="${D_FF:-2048}"
E_LAYERS="${E_LAYERS:-8}"
N_HEADS="${N_HEADS:-8}"
PATCH_LEN="${PATCH_LEN:-96}"

# Step 2: representation extraction
BATCH_SIZE="${BATCH_SIZE:-256}"
POOLING_MODE="${POOLING_MODE:-mean}"

# Step 3: linear probing
LP_EPOCHS="${LP_EPOCHS:-100}"
LP_LR="${LP_LR:-0.001}"
LP_BATCH_SIZE="${LP_BATCH_SIZE:-256}"

# Step 4: analysis
DIM_METHOD="${DIM_METHOD:-pca}"
MAX_SAMPLES_CKA="${MAX_SAMPLES_CKA:-1500}"
MAX_SAMPLES_VIZ="${MAX_SAMPLES_VIZ:-500}"

# GPU device (default: use GPUs 4, 5, 6)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6}"
DEVICE="${DEVICE:-cuda}"
if [ "$DEVICE" = "cuda" ] && ! command -v nvidia-smi &> /dev/null; then
    DEVICE="cpu"
fi

# Output root
OUTPUT_ROOT="${OUTPUT_ROOT:-./results/synthetic}"

# ─────────────────────────────────────────────────────────────────────────────
# Derived paths
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_DIR="$OUTPUT_ROOT"
REPR_DIR="$OUTPUT_ROOT/representations"
LP_DIR="$OUTPUT_ROOT/linear_probing"
MDA_DIR="$OUTPUT_ROOT/multidim_analysis"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Synthetic Dataset Generation
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "[Step 1] Synthetic Time Series Concept Generation"
echo "============================================================"
echo "  N_SAMPLES   : $N_SAMPLES"
echo "  SEQ_LEN     : $SEQ_LEN"
echo "  SEED        : $SEED"
echo "  GPU         : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  OUTPUT      : $SYNTHETIC_DIR"
echo "============================================================"

python3 experiments/ts_concept_synthetic_dataset.py \
    --n_samples "$N_SAMPLES" \
    --seq_len "$SEQ_LEN" \
    --seed "$SEED" \
    --output_dir "$SYNTHETIC_DIR"

DATASET_PATH="$SYNTHETIC_DIR/concepts_dataset.pt"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Representation Extraction & Pooling
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "[Step 2] Timer Representation Extraction & Pooling"
echo "============================================================"
echo "  CKPT_PATH   : $CKPT_PATH"
echo "  D_MODEL     : $D_MODEL"
echo "  E_LAYERS    : $E_LAYERS"
echo "  PATCH_LEN   : $PATCH_LEN"
echo "  BATCH_SIZE  : $BATCH_SIZE"
echo "  POOLING     : $POOLING_MODE"
echo "  DEVICE      : $DEVICE (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "============================================================"

python3 experiments/ts_concept_representation_extraction.py \
    --ckpt_path "$CKPT_PATH" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$REPR_DIR" \
    --d_model "$D_MODEL" \
    --d_ff "$D_FF" \
    --e_layers "$E_LAYERS" \
    --n_heads "$N_HEADS" \
    --patch_len "$PATCH_LEN" \
    --batch_size "$BATCH_SIZE" \
    --pooling_mode "$POOLING_MODE" \
    --device "$DEVICE" \
    --seed "$SEED"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Linear Probing Training
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "[Step 3] Linear Probing Training"
echo "============================================================"
echo "  EPOCHS      : $LP_EPOCHS"
echo "  LR          : $LP_LR"
echo "  BATCH_SIZE  : $LP_BATCH_SIZE"
echo "  TRAIN_RATIO : 0.8"
echo "  OUTPUT      : $LP_DIR"
echo "============================================================"

python3 experiments/ts_concept_linear_probing.py \
    --rep_dir "$REPR_DIR" \
    --output_dir "$LP_DIR" \
    --epochs "$LP_EPOCHS" \
    --lr "$LP_LR" \
    --batch_size "$LP_BATCH_SIZE" \
    --train_ratio 0.8 \
    --seed "$SEED" \
    --device "$DEVICE" \
    --per_concept

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Multi-Dimensional Analysis
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "[Step 4] Multi-Dimensional Analysis"
echo "============================================================"
echo "  DIM_METHOD  : $DIM_METHOD"
echo "  MAX_CKA     : $MAX_SAMPLES_CKA"
echo "  MAX_VIZ     : $MAX_SAMPLES_VIZ"
echo "  OUTPUT      : $MDA_DIR"
echo "============================================================"

python3 experiments/ts_concept_multidim_analysis.py \
    --rep_dir "$REPR_DIR" \
    --output_dir "$MDA_DIR" \
    --dim_method "$DIM_METHOD" \
    --max_samples_cka "$MAX_SAMPLES_CKA" \
    --max_samples_viz "$MAX_SAMPLES_VIZ" \
    --seed "$SEED" \
    --device "$DEVICE"

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "Pipeline Complete"
echo "============================================================"
echo "  Step 1 (Dataset)  : $DATASET_PATH"
echo "  Step 2 (Repr)     : $REPR_DIR/layer_representations.pt"
echo "  Step 3 (LP)       : $LP_DIR/linear_probing_results.pt"
echo "  Step 4 (Analysis) : $MDA_DIR/"
echo "============================================================"
echo ""
echo "Key outputs:"
echo "  $SYNTHETIC_DIR/concepts_visualization.png"
echo "  $REPR_DIR/repr_stats.png"
echo "  $LP_DIR/layer_mse_curve.png"
echo "  $MDA_DIR/cka_heatmap.png"
echo "  $MDA_DIR/layer_embeddings_${DIM_METHOD}.png"
echo "  $MDA_DIR/silhouette_scores.png"
echo ""
