#!/usr/bin/env bash
# =============================================================================
# Timer Token-Level MI Masked Knowledge Distillation Comparison
# =============================================================================
# Compares FIVE training strategies:
#   1. From scratch (no teacher)
#   2. Uniform token-level distillation
#   3. Layer-level MI-weighted distillation (token-level)
#   4. Token-level high-MI mask distillation
#   5. Token-level high-MI mask + Layer-level MI-weighted distillation
#
# Datasets: ETTh1, ETTh2, ETTm1, ETTm2
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="${ROOT_DIR}/experiments"
OUT_DIR="${ROOT_DIR}/results/timer_token_mi_distillation"

# ── MI & Checkpoint ──────────────────────────────────────────────────────────
CKPT="${ROOT_DIR}/checkpoints/Timer_forecast_1.0.ckpt"
MI_DIR="${ROOT_DIR}/outputs/timer_mi_ksg_pca/Timer_MI_20260516_133108"

# ── Model (Teacher: Timer) ───────────────────────────────────────────────────
D_MODEL_T=1024
D_FF_T=2048
E_LAYERS=8
N_HEADS=8
DROPOUT=0.1

# ── Model (Student) ──────────────────────────────────────────────────────────
S_D_MODEL=256
S_D_FF=512
S_N_LAYERS=4
S_N_HEADS=4
S_DROPOUT=0.1

# ── Data ─────────────────────────────────────────────────────────────────────
SEQ_LEN=672
PRED_LEN=96
PATCH_LEN=96
BATCH_SIZE=32

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS=15
LR=1e-3
WEIGHT_DECAY=1e-4
PATIENCE=4
ALPHA=1.0

GPU=${GPU:-3}
SEED=${SEED:-42}

echo "======================================================================"
echo "Timer Token-Level MI Masked Knowledge Distillation"
echo "======================================================================"
echo "  ROOT_DIR   : ${ROOT_DIR}"
echo "  OUT_DIR    : ${OUT_DIR}"
echo "  MI_DIR     : ${MI_DIR}"
echo "  CKPT       : ${CKPT}"
echo "  SEQ_LEN    : ${SEQ_LEN}"
echo "  PRED_LEN   : ${PRED_LEN}"
echo "  PATCH_LEN  : ${PATCH_LEN}"
echo "  Teacher    : d_model=${D_MODEL_T}, d_ff=${D_FF_T}, e_layers=${E_LAYERS}"
echo "  Student    : d_model=${S_D_MODEL}, d_ff=${S_D_FF}, s_layers=${S_N_LAYERS}"
echo "  EPOCHS     : ${EPOCHS}"
echo "  LR         : ${LR}"
echo "  ALPHA      : ${ALPHA}"
echo "  GPU        : ${GPU}"
echo "  SEED       : ${SEED}"
echo "======================================================================"

mkdir -p "${OUT_DIR}"

run_one() {
  local DATASET=$1
  local CSV=$2
  local FREQ=$3

  echo ""
  echo ">>> [${DATASET}]"
  python "${EXP_DIR}/timer_token_mi_distillation_comparison.py" \
      --mi_dir "${MI_DIR}" \
      --root_path "${ROOT_DIR}/datasets/" \
      --data "${DATASET}" \
      --data_path "${CSV}" \
      --out_dir "${OUT_DIR}" \
      --ckpt_path "${CKPT}" \
      --d_model_t "${D_MODEL_T}" \
      --d_ff_t "${D_FF_T}" \
      --e_layers "${E_LAYERS}" \
      --n_heads "${N_HEADS}" \
      --s_d_model "${S_D_MODEL}" \
      --s_d_ff "${S_D_FF}" \
      --s_n_layers "${S_N_LAYERS}" \
      --s_n_heads "${S_N_HEADS}" \
      --s_dropout "${S_DROPOUT}" \
      --seq_len "${SEQ_LEN}" \
      --pred_len "${PRED_LEN}" \
      --patch_len "${PATCH_LEN}" \
      --batch_size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --patience "${PATIENCE}" \
      --alpha "${ALPHA}" \
      --gpu "${GPU}" \
      --model_id "${DATASET}"
}

# ── ETTh1 ────────────────────────────────────────────────────────────────────
run_one "ETTh1" "ETTh1.csv" "h"

# ── ETTh2 ────────────────────────────────────────────────────────────────────
run_one "ETTh2" "ETTh2.csv" "h"

# ── ETTm1 ────────────────────────────────────────────────────────────────────
run_one "ETTm1" "ETTm1.csv" "t"

# ── ETTm2 ────────────────────────────────────────────────────────────────────
run_one "ETTm2" "ETTm2.csv" "t"

echo ""
echo "======================================================================"
echo "All datasets complete. Results saved in: ${OUT_DIR}"
echo "======================================================================"
