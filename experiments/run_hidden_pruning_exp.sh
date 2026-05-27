#!/usr/bin/env bash
# =============================================================================
# Timer MI-Guided Hidden-State Pruning Experiment
# =============================================================================
# Compares: zero hidden states vs zero attention scores for low-MI tokens.
#
# Datasets: ETTh1, ETTh2, ETTm1, ETTm2
# =============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="${ROOT_DIR}/experiments"
OUT_DIR="${ROOT_DIR}/results/timer_mi_hidden_pruning"
MI_DIR="${ROOT_DIR}/outputs/timer_mi_ksg_pca/Timer_MI_20260516_133108"

# ── Model ──────────────────────────────────────────────────────────────────────
CKPT="${ROOT_DIR}/checkpoints/Timer_forecast_1.0.ckpt"
D_MODEL=1024
D_FF=2048
E_LAYERS=8
N_HEADS=8
DROPOUT=0.1

# ── Data ─────────────────────────────────────────────────────────────────────
SEQ_LEN=672
PRED_LEN=96
PATCH_LEN=96
BATCH_SIZE=64

# ── Experiment ────────────────────────────────────────────────────────────────
MASK_RATIOS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

GPU=${GPU:-0}
SEED=${SEED:-42}

echo "======================================================================"
echo "Timer MI Hidden-State Pruning Experiment"
echo "======================================================================"
echo "  ROOT_DIR   : ${ROOT_DIR}"
echo "  OUT_DIR    : ${OUT_DIR}"
echo "  MI_DIR     : ${MI_DIR}"
echo "  SEQ_LEN    : ${SEQ_LEN}"
echo "  PRED_LEN   : ${PRED_LEN}"
echo "  GPU        : ${GPU}"
echo "======================================================================"

mkdir -p "${OUT_DIR}"

# ── ETTh1 ────────────────────────────────────────────────────────────────────
echo ""
echo ">>> [ETTh1]"
python "${EXP_DIR}/timer_mi_hidden_pruning_experiment.py" \
    --mi_dir "${MI_DIR}" \
    --data_path "${ROOT_DIR}/datasets/ETTh1.csv" \
    --data_type ETTh1 \
    --output_dir "${OUT_DIR}/ETTh1" \
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
    --mask_ratios "${MASK_RATIOS[@]}" \
    --seed "${SEED}" \
    --gpu "${GPU}" \
    --all_layers

# ── ETTh2 ────────────────────────────────────────────────────────────────────
echo ""
echo ">>> [ETTh2]"
python "${EXP_DIR}/timer_mi_hidden_pruning_experiment.py" \
    --mi_dir "${MI_DIR}" \
    --data_path "${ROOT_DIR}/datasets/ETTh2.csv" \
    --data_type ETTh2 \
    --output_dir "${OUT_DIR}/ETTh2" \
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
    --mask_ratios "${MASK_RATIOS[@]}" \
    --seed "${SEED}" \
    --gpu "${GPU}" \
    --all_layers

# ── ETTm1 ────────────────────────────────────────────────────────────────────
echo ""
echo ">>> [ETTm1]"
python "${EXP_DIR}/timer_mi_hidden_pruning_experiment.py" \
    --mi_dir "${MI_DIR}" \
    --data_path "${ROOT_DIR}/datasets/ETTm1.csv" \
    --data_type ETTm1 \
    --output_dir "${OUT_DIR}/ETTm1" \
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
    --mask_ratios "${MASK_RATIOS[@]}" \
    --seed "${SEED}" \
    --gpu "${GPU}" \
    --all_layers

# ── ETTm2 ────────────────────────────────────────────────────────────────────
echo ""
echo ">>> [ETTm2]"
python "${EXP_DIR}/timer_mi_hidden_pruning_experiment.py" \
    --mi_dir "${MI_DIR}" \
    --data_path "${ROOT_DIR}/datasets/ETTm2.csv" \
    --data_type ETTm2 \
    --output_dir "${OUT_DIR}/ETTm2" \
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
    --mask_ratios "${MASK_RATIOS[@]}" \
    --seed "${SEED}" \
    --gpu "${GPU}" \
    --all_layers

echo ""
echo "======================================================================"
echo "All experiments complete. Results in: ${OUT_DIR}/"
echo "======================================================================"
