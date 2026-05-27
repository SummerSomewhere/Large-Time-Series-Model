#!/bin/bash
#
# Timer Probe Experiment Pipeline
#
# Full 4-step pipeline for probing Timer representations on synthetic
# time series concepts:
#
#   Step 1: Generate synthetic time series concepts (7 types)
#   Step 2: Extract Timer hidden states and pool them
#   Step 3: Train linear probes to predict concept parameters
#   Step 4: Multi-dimensional analysis (CKA, UMAP, PCA, Silhouette)
#   Step 5: Per-token KSG MI analysis with SPI-based token ordering
#
# Usage:
#   bash probe/ts_concept_analysis_pipeline.sh
#
#   Or with custom parameters:
#   CKPT_PATH=checkpoints/Timer_forecast_1.0.ckpt N_SAMPLES=1000 SEQ_LEN=512 \
#   bash probe/ts_concept_analysis_pipeline.sh
#
# Individual steps:
#   python probe/ts_concept_synthetic_dataset.py ...
#   python probe/ts_concept_representation_extraction.py ... --use_dataparallel  # Step 2 multi-GPU
#   python probe/ts_concept_linear_probing.py ... --torchrun  # Step 3 multi-GPU
#   python probe/ts_concept_multidim_analysis.py ...
#   python probe/ts_concept_ksg_mi_analysis.py ...  # Step 5, requires --save_token_reps from Step 2
#

set -e

_ROOT="$(cd "$(dirname "$0")/.." && pwd)" && cd "$_ROOT" || exit 1

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Configuration
# ─────────────────────────────────────────────────────────────────────────────

CKPT_PATH="${CKPT_PATH:-checkpoints/Timer_forecast_1.0.ckpt}"

# Synthetic dataset parameters
N_SAMPLES="${N_SAMPLES:-1000}"
SEQ_LEN="${SEQ_LEN:-512}"
SEED="${SEED:-42}"

# Timer-specific parameters
PATCH_LEN="${PATCH_LEN:-96}"
STRIDE="${STRIDE:-96}"
D_MODEL="${D_MODEL:-1024}"
D_FF="${D_FF:-2048}"
E_LAYERS="${E_LAYERS:-8}"
FACTOR="${FACTOR:-3}"
N_HEADS="${N_HEADS:-16}"

# Step 2: representation extraction
BATCH_SIZE="${BATCH_SIZE:-256}"
POOLING_MODE="${POOLING_MODE:-mean}"
USE_DATAPARALLEL="${USE_DATAPARALLEL:-true}"

# Step 3: linear probing
LP_EPOCHS="${LP_EPOCHS:-100}"
LP_LR="${LP_LR:-0.001}"
LP_BATCH_SIZE="${LP_BATCH_SIZE:-256}"
LP_TORCHRUN="${LP_TORCHRUN:-true}"

# Step 4: analysis
DIM_METHOD="${DIM_METHOD:-pca}"
MAX_SAMPLES_CKA="${MAX_SAMPLES_CKA:-1500}"
MAX_SAMPLES_VIZ="${MAX_SAMPLES_VIZ:-500}"

# Step 5: per-token KSG MI analysis (SPI-based)
KSG_TOP_K="${KSG_TOP_K:-1}"
KSG_PERCENTILE="${KSG_PERCENTILE:-25}"
KSG_PCA_DIM="${KSG_PCA_DIM:-32}"
KSG_K_NEIGHBORS="${KSG_K_NEIGHBORS:-3}"
KSG_SPI_BIAS="${KSG_SPI_BIAS:-0.15}"
KSG_MAX_SAMPLES="${KSG_MAX_SAMPLES:-1000}"
KSG_N_SAMPLES_KSG="${KSG_N_SAMPLES_KSG:-1000}"

# GPU device — use specific GPUs: 0,1,3,4,5,6,7
GPU_DEVICES="${GPU_DEVICES:-0,1,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
DEVICE="${DEVICE:-cuda}"
if [ "$DEVICE" = "cuda" ] && ! command -v nvidia-smi &> /dev/null; then
    DEVICE="cpu"
fi

# Output root
OUTPUT_ROOT="${OUTPUT_ROOT:-./results/timer_probe}"

# ─────────────────────────────────────────────────────────────────────────────
# Derived paths
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_DIR="$OUTPUT_ROOT"
REPR_DIR="$OUTPUT_ROOT/representations"
LP_DIR="$OUTPUT_ROOT/linear_probing"
MDA_DIR="$OUTPUT_ROOT/multidim_analysis"
KSA_DIR="$OUTPUT_ROOT/ksg_mi_token"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Synthetic Dataset Generation
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "[Step 1] Synthetic Time Series Concept Generation (Timer)"
echo "============================================================"
echo "  N_SAMPLES   : $N_SAMPLES"
echo "  SEQ_LEN     : $SEQ_LEN  (Timer context length)"
echo "  SEED        : $SEED"
echo "  GPU         : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  OUTPUT      : $SYNTHETIC_DIR"
echo "============================================================"

python3 probe/ts_concept_synthetic_dataset.py \
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
echo "  CKPT_PATH    : $CKPT_PATH"
echo "  SEQ_LEN      : $SEQ_LEN"
echo "  PATCH_LEN    : $PATCH_LEN"
echo "  D_MODEL      : $D_MODEL"
echo "  N_HEADS      : $N_HEADS"
echo "  E_LAYERS     : $E_LAYERS"
echo "  POOLING      : $POOLING_MODE"
echo "  DATAPARALLEL : $USE_DATAPARALLEL"
echo "  DEVICE       : $DEVICE (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "============================================================"

python3 probe/ts_concept_representation_extraction.py \
    --ckpt_path "$CKPT_PATH" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$REPR_DIR" \
    --seq_len "$SEQ_LEN" \
    --patch_len "$PATCH_LEN" \
    --stride "$STRIDE" \
    --d_model "$D_MODEL" \
    --d_ff "$D_FF" \
    --n_heads "$N_HEADS" \
    --e_layers "$E_LAYERS" \
    --factor "$FACTOR" \
    --batch_size "$BATCH_SIZE" \
    --pooling_mode "$POOLING_MODE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --save_token_reps \
    $([ "$USE_DATAPARALLEL" = "true" ] && echo "--use_dataparallel")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Linear Probing Training
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "[Step 3] Linear Probing Training (Timer)"
echo "============================================================"
echo "  EPOCHS       : $LP_EPOCHS"
echo "  LR           : $LP_LR"
echo "  BATCH_SIZE   : $LP_BATCH_SIZE"
echo "  TRAIN_RATIO  : 0.8"
echo "  TORCHRUN     : $LP_TORCHRUN"
echo "  OUTPUT       : $LP_DIR"
echo "============================================================"

python3 probe/ts_concept_linear_probing.py \
    --rep_dir "$REPR_DIR" \
    --output_dir "$LP_DIR" \
    --epochs "$LP_EPOCHS" \
    --lr "$LP_LR" \
    --batch_size "$LP_BATCH_SIZE" \
    --train_ratio 0.8 \
    --seed "$SEED" \
    --device "$DEVICE" \
    --per_concept \
    $([ "$LP_TORCHRUN" = "true" ] && echo "--torchrun")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Multi-Dimensional Analysis
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "[Step 4] Multi-Dimensional Analysis (Timer)"
echo "============================================================"
echo "  DIM_METHOD  : $DIM_METHOD"
echo "  MAX_CKA     : $MAX_SAMPLES_CKA"
echo "  MAX_VIZ     : $MAX_SAMPLES_VIZ"
echo "  OUTPUT      : $MDA_DIR"
echo "============================================================"

python3 probe/ts_concept_multidim_analysis.py \
    --rep_dir "$REPR_DIR" \
    --output_dir "$MDA_DIR" \
    --dim_method "$DIM_METHOD" \
    --max_samples_cka "$MAX_SAMPLES_CKA" \
    --max_samples_viz "$MAX_SAMPLES_VIZ" \
    --seed "$SEED" \
    --device "$DEVICE"

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Per-Token KSG MI Analysis (SPI-Based Ordering)
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "[Step 5] Per-Token KSG MI Analysis (SPI-Based)"
echo "============================================================"
echo "  KSG_TOP_K        : $KSG_TOP_K"
echo "  KSG_PERCENTILE  : $KSG_PERCENTILE%"
echo "  KSG_PCA_DIM      : $KSG_PCA_DIM"
echo "  KSG_K_NEIGHBORS  : $KSG_K_NEIGHBORS"
echo "  KSG_SPI_BIAS     : $KSG_SPI_BIAS"
echo "  KSG_MAX_SAMPLES  : $KSG_MAX_SAMPLES"
echo "  OUTPUT           : $KSA_DIR"
echo "============================================================"

python3 probe/ts_concept_ksg_mi_analysis.py \
    --rep_dir "$REPR_DIR" \
    --output_dir "$KSA_DIR" \
    --top_k "$KSG_TOP_K" \
    --percentile "$KSG_PERCENTILE" \
    --pca_dim "$KSG_PCA_DIM" \
    --k_neighbors "$KSG_K_NEIGHBORS" \
    --spi_bias_factor "$KSG_SPI_BIAS" \
    --max_samples "$KSG_MAX_SAMPLES" \
    --max_samples_ksg "$KSG_N_SAMPLES_KSG" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --num_workers 7

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "Pipeline Complete (Timer Probe)"
echo "============================================================"
echo "  Step 1 (Dataset)  : $DATASET_PATH"
echo "  Step 2 (Repr)    : $REPR_DIR/layer_representations.pt"
echo "  Step 3 (LP)      : $LP_DIR/run_*/linear_probing_results.pt"
echo "  Step 4 (Analysis): $MDA_DIR/"
echo "  Step 5 (KSG MI):  $KSA_DIR/"
echo "============================================================"
echo ""
echo "Key outputs:"
echo "  $SYNTHETIC_DIR/concepts_visualization.png"
echo "  $REPR_DIR/repr_stats.png"
echo "  $LP_DIR/run_*/layer_mse_curve.png"
echo "  $MDA_DIR/cka_heatmap.png"
echo "  $MDA_DIR/layer_embeddings_${DIM_METHOD}.png"
echo "  $MDA_DIR/silhouette_scores.png"
echo "  $KSA_DIR/ksg_spi_summary.png"
echo "  $KSA_DIR/ksg_spi_profile.png"
echo "  $KSA_DIR/ksg_within_layer_summary_25pct.png"
echo "  $KSA_DIR/ksg_within_layer_concept_grid_25pct.png"
echo ""
