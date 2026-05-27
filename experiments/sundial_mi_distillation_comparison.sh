#!/usr/bin/env bash
# =============================================================================
# Sundial MI-Guided Knowledge Distillation Comparison Experiment
# =============================================================================
# 对齐 timer_mi_distillation_comparison.sh 的结构和参数风格。
#
# 对比三种训练方式：
#   1. 从零训练学生模型 (no teacher)
#   2. 普通知识蒸馏 (均匀权重)
#   3. MI 引导知识蒸馏 (按层 MI 加权)
#
# Datasets: ETTh1, ETTh2, ETTm1, ETTm2
# ==============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="${ROOT_DIR}/experiments"
OUT_DIR="${ROOT_DIR}/results/sundial_mi_distillation"

# ── MI & Checkpoint ─────────────────────────────────────────────────────────
CKPT="${ROOT_DIR}/checkpoints/sundial-base-128m.pt"
MI_DIR="${ROOT_DIR}/results/sundial_mi_ksg_pca/Sundial_MI_20260520_120000"

# ── Model (Teacher: Sundial) ─────────────────────────────────────────────────
# Sundial base: d_model=768, d_ff=3072, layers=12, heads=12
D_MODEL_T=768
D_FF_T=3072
E_LAYERS=12
N_HEADS=12

# ── Model (Student) ──────────────────────────────────────────────────────────
S_D_MODEL=256
S_D_FF=512
S_N_LAYERS=4
S_N_HEADS=4
S_DROPOUT=0.1

# ── Data ─────────────────────────────────────────────────────────────────────
# Sundial 使用的序列长度 (512) 与 Timer (672) 不同
SEQ_LEN=512
PRED_LEN=96
LABEL_LEN=48
PATCH_LEN=16
BATCH_SIZE=32

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS=15
LR=1e-3
WEIGHT_DECAY=1e-4
PATIENCE=4
ALPHA=1.0

GPU=${GPU:-0}
SEED=${SEED:-42}

echo "======================================================================"
echo "Sundial MI-Guided Knowledge Distillation Comparison"
echo "======================================================================"
echo "  ROOT_DIR   : ${ROOT_DIR}"
echo "  OUT_DIR    : ${OUT_DIR}"
echo "  MI_DIR     : ${MI_DIR}"
echo "  CKPT       : ${CKPT}"
echo "  SEQ_LEN    : ${SEQ_LEN}"
echo "  PRED_LEN   : ${PRED_LEN}"
echo "  LABEL_LEN  : ${LABEL_LEN}"
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

# ── ETTh1 ────────────────────────────────────────────────────────────────────
echo ""
echo ">>> [ETTh1]"
python "${EXP_DIR}/sundial_mi_distillation_comparison.py" \
    --mi_dir "${MI_DIR}" \
    --data_path "${ROOT_DIR}/datasets/ETTh1.csv" \
    --data_type ETTh1 \
    --output_dir "${OUT_DIR}" \
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
    --label_len "${LABEL_LEN}" \
    --patch_len "${PATCH_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --patience "${PATIENCE}" \
    --alpha "${ALPHA}" \
    --seed "${SEED}" \
    --gpu "${GPU}" \
    --model_id etth1

# ── ETTh2 ────────────────────────────────────────────────────────────────────
echo ""
echo ">>> [ETTh2]"
python "${EXP_DIR}/sundial_mi_distillation_comparison.py" \
    --mi_dir "${MI_DIR}" \
    --data_path "${ROOT_DIR}/datasets/ETTh2.csv" \
    --data_type ETTh2 \
    --output_dir "${OUT_DIR}" \
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
    --label_len "${LABEL_LEN}" \
    --patch_len "${PATCH_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --patience "${PATIENCE}" \
    --alpha "${ALPHA}" \
    --seed "${SEED}" \
    --gpu "${GPU}" \
    --model_id etth2

# ── ETTm1 ────────────────────────────────────────────────────────────────────
echo ""
echo ">>> [ETTm1]"
python "${EXP_DIR}/sundial_mi_distillation_comparison.py" \
    --mi_dir "${MI_DIR}" \
    --data_path "${ROOT_DIR}/datasets/ETTm1.csv" \
    --data_type ETTm1 \
    --output_dir "${OUT_DIR}" \
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
    --label_len "${LABEL_LEN}" \
    --patch_len "${PATCH_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --patience "${PATIENCE}" \
    --alpha "${ALPHA}" \
    --seed "${SEED}" \
    --gpu "${GPU}" \
    --model_id ettm1

# ── ETTm2 ────────────────────────────────────────────────────────────────────
echo ""
echo ">>> [ETTm2]"
python "${EXP_DIR}/sundial_mi_distillation_comparison.py" \
    --mi_dir "${MI_DIR}" \
    --data_path "${ROOT_DIR}/datasets/ETTm2.csv" \
    --data_type ETTm2 \
    --output_dir "${OUT_DIR}" \
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
    --label_len "${LABEL_LEN}" \
    --patch_len "${PATCH_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --patience "${PATIENCE}" \
    --alpha "${ALPHA}" \
    --seed "${SEED}" \
    --gpu "${GPU}" \
    --model_id ettm2

echo ""
echo "======================================================================"
echo "All datasets complete. Results saved in: ${OUT_DIR}"
echo "======================================================================"
