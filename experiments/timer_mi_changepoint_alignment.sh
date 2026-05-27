#!/bin/bash
# Timer MI-变点对齐 & STL 分析实验启动脚本
#
# Usage:
#   bash experiments/timer_mi_changepoint_alignment.sh

set -e

# ── 实验配置 ──────────────────────────────────────────────────────────────
DATA=${1:-ETTh1}
SEQ_LEN=${2:-672}
PATCH_LEN=${3:-96}
PRED_LEN=${4:-96}

DATA_PATH="${DATA}.csv"
CKPT_PATH="./checkpoints/Timer_forecast_1.0.ckpt"
ROOT_PATH="./datasets/"
OUT_DIR="./results/timer_mi_changepoint/"
MI_RESULT_DIR="./outputs/timer_mi_ksg_pca/Timer_MI_20260516_133108"

# 模型配置
D_MODEL=1024
D_FF=2048
E_LAYERS=8
N_HEADS=8

# 分析配置
STL_PERIOD=24
TOP_K_RATIO=0.15
MAX_SAMPLES=2000
MAX_SAMPLES_STL=2000
GPU=0

# ── 运行 ──────────────────────────────────────────────────────────────────
echo "========================================"
echo "  Timer MI-变点对齐 & STL 分析"
echo "========================================"
echo "  Dataset:     ${DATA}"
echo "  seq_len:     ${SEQ_LEN}"
echo "  patch_len:   ${PATCH_LEN}"
echo "  pred_len:    ${PRED_LEN}"
echo "  stl_period:  ${STL_PERIOD}"
echo "  max_samples: ${MAX_SAMPLES}"
echo "  gpu:         ${GPU}"
echo "========================================"

python experiments/timer_mi_changepoint_alignment.py \
    --root_path "${ROOT_PATH}" \
    --data_path "${DATA_PATH}" \
    --seq_len "${SEQ_LEN}" \
    --pred_len "${PRED_LEN}" \
    --patch_len "${PATCH_LEN}" \
    --batch_size 64 \
    --num_workers 4 \
    --max_samples "${MAX_SAMPLES}" \
    --freq h \
    --data_type "${DATA}" \
    --ckpt_path "${CKPT_PATH}" \
    --d_model "${D_MODEL}" \
    --d_ff "${D_FF}" \
    --e_layers "${E_LAYERS}" \
    --n_heads "${N_HEADS}" \
    --dropout 0.1 \
    --mi_result_dir "${MI_RESULT_DIR}" \
    --model_id "${DATA,,}" \
    --stl_period "${STL_PERIOD}" \
    --top_k_ratio "${TOP_K_RATIO}" \
    --max_samples_stl "${MAX_SAMPLES_STL}" \
    --gpu "${GPU}" \
    --out_dir "${OUT_DIR}"
