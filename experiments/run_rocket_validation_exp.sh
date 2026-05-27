#!/usr/bin/env bash
# =============================================================================
# Timer MI ROCKET Validation + Change-Point Alignment Experiment
# =============================================================================
# Validates MI scores using:
#   1. ROCKET Feature Discriminability (aeon/MiniRocket)
#   2. Change-Point Alignment (aeon/ClaSPSegmenter)
#
# Datasets: ETTh1, ETTh2, ETTm1, ETTm2
# =============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="${ROOT_DIR}/experiments"
OUT_DIR="${ROOT_DIR}/results/timer_mi_rocket_validation"

# ── Model ──────────────────────────────────────────────────────────────────────
CKPT="${ROOT_DIR}/checkpoints/Timer_forecast_1.0.ckpt"
D_MODEL=1024
D_FF=2048
E_LAYERS=8
N_HEADS=8
DROPOUT=0.1

# ── Data ──────────────────────────────────────────────────────────────────────
SEQ_LEN=672
PRED_LEN=96
PATCH_LEN=96
BATCH_SIZE=64

# ── MI Matrix ─────────────────────────────────────────────────────────────────
# Primary path (your actual MI output directory)
MI_DIR="${ROOT_DIR}/outputs/timer_mi_ksg_pca/Timer_MI_20260516_133108"

# ── Validation ────────────────────────────────────────────────────────────────
MI_PERCENTILE=50
N_KERNELS=10000
N_CP_SAMPLES=500

GPU=${GPU:-0}
SEED=${SEED:-42}

echo "======================================================================"
echo "Timer MI ROCKET Validation + Change-Point Alignment"
echo "======================================================================"
echo "  ROOT_DIR        : ${ROOT_DIR}"
echo "  OUT_DIR         : ${OUT_DIR}"
echo "  MI_DIR          : ${MI_DIR}"
echo "  D_MODEL         : ${D_MODEL}"
echo "  SEQ_LEN         : ${SEQ_LEN}"
echo "  PRED_LEN        : ${PRED_LEN}"
echo "  MI_PERCENTILE   : ${MI_PERCENTILE}"
echo "  N_KERNELS       : ${N_KERNELS}"
echo "  GPU             : ${GPU}"
echo "======================================================================"

mkdir -p "${OUT_DIR}"

# Check dependencies
if ! python -c "import aeon" 2>/dev/null; then
    echo "WARNING: 'aeon' not installed. Run: uv pip install aeon"
fi

run_experiment() {
    local DATA_NAME="$1"
    local DATA_PATH="$2"
    local DATA_TYPE="$3"
    local FREQ="$4"

    echo ""
    echo ">>> [${DATA_NAME}]"
    python "${EXP_DIR}/timer_mi_rocket_validation.py" \
        --mi_dir "${MI_DIR}" \
        --data_path "${DATA_PATH}" \
        --data_type "${DATA_TYPE}" \
        --output_dir "${OUT_DIR}/${DATA_NAME}" \
        --ckpt_path "${CKPT}" \
        --d_model "${D_MODEL}" \
        --d_ff "${D_FF}" \
        --e_layers "${E_LAYERS}" \
        --n_heads "${N_HEADS}" \
        --dropout "${DROPOUT}" \
        --seq_len "${SEQ_LEN}" \
        --pred_len "${PRED_LEN}" \
        --patch_len "${PATCH_LEN}" \
        --batch_size "${BATCH_SIZE}" \
        --mi_percentile "${MI_PERCENTILE}" \
        --n_kernels "${N_KERNELS}" \
        --n_cp_samples "${N_CP_SAMPLES}" \
        --seed "${SEED}" \
        --gpu "${GPU}" \
        --freq "${FREQ}"
}

# ── ETTh1 (hourly) ────────────────────────────────────────────────────────────
run_experiment "ETTh1" "${ROOT_DIR}/datasets/ETTh1.csv" "ETTh1" "h"

# ── ETTh2 (hourly) ────────────────────────────────────────────────────────────
run_experiment "ETTh2" "${ROOT_DIR}/datasets/ETTh2.csv" "ETTh2" "h"

# ── ETTm1 (15min) ─────────────────────────────────────────────────────────────
run_experiment "ETTm1" "${ROOT_DIR}/datasets/ETTm1.csv" "ETTm1" "t"

# ── ETTm2 (15min) ─────────────────────────────────────────────────────────────
run_experiment "ETTm2" "${ROOT_DIR}/datasets/ETTm2.csv" "ETTm2" "t"

echo ""
echo "======================================================================"
echo "All experiments complete. Results in: ${OUT_DIR}/"
echo "======================================================================"
