#!/usr/bin/env bash
# =============================================================================
# Timer MI-guided Sample-Level Sampling (Scheme B)
# =============================================================================
# This script runs the *sample-window sampler* comparison:
#   - uniform sampler
#   - high-MI-driven sampler
#   - low-MI-driven sampler
#   - random-patches-driven sampler (same patch count control)
#
# Output directory structure (Nature-style, clean):
#   results/timer_mi_sample_sampling/<dataset>/<sampler_mode>/...
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="${ROOT_DIR}/experiments"
OUT_DIR="${ROOT_DIR}/results/timer_mi_sample_sampling"

CKPT="${ROOT_DIR}/checkpoints/Timer_forecast_1.0.ckpt"
MI_DIR="${ROOT_DIR}/outputs/timer_mi_ksg_pca/Timer_MI_20260516_133108"

# Teacher (Timer)
D_MODEL_T=1024
D_FF_T=2048
E_LAYERS=8
N_HEADS=8
DROPOUT=0.1

# Student
S_D_MODEL=256
S_D_FF=512
S_N_LAYERS=4
S_N_HEADS=4
S_DROPOUT=0.1

# Data
SEQ_LEN=672
PRED_LEN=96
PATCH_LEN=96
BATCH_SIZE=32

# Training
EPOCHS=15
LR=1e-3
WEIGHT_DECAY=1e-4

# Sampling curriculum
WARMUP_EPOCHS=2
ALPHA_START=2.0
ALPHA_END=0.0
EPS=1e-6

GPU=${GPU:-2}
SEED=${SEED:-42}

mkdir -p "${OUT_DIR}"

run_one () {
  local DATASET=$1
  local CSV=$2
  local MODE=$3

  echo ""
  echo ">>> [${DATASET}] sampler_mode=${MODE}"
  python "${EXP_DIR}/timer_mi_sample_sampling_comparison.py" \
    --mi_dir "${MI_DIR}" \
    --root_path "${ROOT_DIR}/datasets/" \
    --data "${DATASET}" \
    --data_path "${CSV}" \
    --out_dir "${OUT_DIR}/${DATASET}/${MODE}" \
    --ckpt_path "${CKPT}" \
    --d_model_t "${D_MODEL_T}" \
    --d_ff_t "${D_FF_T}" \
    --e_layers "${E_LAYERS}" \
    --n_heads "${N_HEADS}" \
    --dropout "${DROPOUT}" \
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
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --alpha_start "${ALPHA_START}" \
    --alpha_end "${ALPHA_END}" \
    --eps "${EPS}" \
    --sampler_mode "${MODE}" \
    --gpu "${GPU}" \
    --seed "${SEED}" \
    --model_id "${DATASET}_${MODE}"
}

for DATASET in ETTh1 ETTh2 ETTm1 ETTm2; do
  CSV="${DATASET}.csv"
  for MODE in uniform high low random; do
    run_one "${DATASET}" "${CSV}" "${MODE}"
  done
done

echo ""
echo "======================================================================"
echo "All runs complete. Results saved in: ${OUT_DIR}"
echo "======================================================================"
