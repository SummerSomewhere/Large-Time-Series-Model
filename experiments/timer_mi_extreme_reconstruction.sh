#!/bin/bash
#
# Timer 极端对比掩码重建实验启动脚本
#
# Usage:
#   bash experiments/timer_mi_extreme_reconstruction.sh
#   bash experiments/timer_mi_extreme_reconstruction.sh <gpu_id> [task]
#
# Tasks:
#   etth1       - ETTh1 dataset (default)
#   etth2       - ETTh2 dataset
#   ettm1       - ETTm1 dataset
#   weather     - Weather dataset
#   electricity - Electricity dataset
#   all         - Run all datasets

set -e

GPU_ID="${1:-0}"
TASK="${2:-etth1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ─── 路径配置 ─────────────────────────────────────────────────────────────────
CKPT_PATH="${ROOT_DIR}/checkpoints/Timer_forecast_1.0.ckpt"
MI_RESULT_DIR="${ROOT_DIR}/results/global_mi_peaks_etth1.json"
OUT_DIR="${ROOT_DIR}/outputs/timer_mi_ksg_pca/Timer_MI_20260516_133108"

# ─── 数据集配置 ──────────────────────────────────────────────────────────────
SEQ_LEN=672
PRED_LEN=96
PATCH_LEN=96
BATCH_SIZE=64
MAX_SAMPLES=500
N_VISUAL=3
MASK_RATIO=0.5

# ─── 模型配置 ────────────────────────────────────────────────────────────────
D_MODEL=1024
D_FF=2048
E_LAYERS=8
N_HEADS=8
DROPOUT=0.1

# ─── 实验函数 ────────────────────────────────────────────────────────────────

run_etth1() {
    echo "========================================"
    echo "  Extreme Contrast Reconstruction: ETTh1"
    echo "========================================"
    python "${ROOT_DIR}/experiments/timer_mi_extreme_reconstruction.py" \
        --mi_result_dir "${MI_RESULT_DIR}" \
        --root_path "${ROOT_DIR}/datasets/" \
        --data_path "ETTh1.csv" \
        --data_type "ETTh1" \
        --seq_len ${SEQ_LEN} \
        --pred_len ${PRED_LEN} \
        --patch_len ${PATCH_LEN} \
        --batch_size ${BATCH_SIZE} \
        --max_samples ${MAX_SAMPLES} \
        --mask_ratio ${MASK_RATIO} \
        --ckpt_path "${CKPT_PATH}" \
        --d_model ${D_MODEL} \
        --d_ff ${D_FF} \
        --e_layers ${E_LAYERS} \
        --n_heads ${N_HEADS} \
        --dropout ${DROPOUT} \
        --gpu ${GPU_ID} \
        --out_dir "${OUT_DIR}/etth1" \
        --model_id "etth1" \
        --n_visual ${N_VISUAL} \
        --freq "h"
}

run_etth2() {
    echo "========================================"
    echo "  Extreme Contrast Reconstruction: ETTh2"
    echo "========================================"
    python "${ROOT_DIR}/experiments/timer_mi_extreme_reconstruction.py" \
        --mi_result_dir "${MI_RESULT_DIR}" \
        --root_path "${ROOT_DIR}/datasets/" \
        --data_path "ETTh2.csv" \
        --data_type "ETTh2" \
        --seq_len ${SEQ_LEN} \
        --pred_len ${PRED_LEN} \
        --patch_len ${PATCH_LEN} \
        --batch_size ${BATCH_SIZE} \
        --max_samples ${MAX_SAMPLES} \
        --mask_ratio ${MASK_RATIO} \
        --ckpt_path "${CKPT_PATH}" \
        --d_model ${D_MODEL} \
        --d_ff ${D_FF} \
        --e_layers ${E_LAYERS} \
        --n_heads ${N_HEADS} \
        --dropout ${DROPOUT} \
        --gpu ${GPU_ID} \
        --out_dir "${OUT_DIR}/etth2" \
        --model_id "etth2" \
        --n_visual ${N_VISUAL} \
        --freq "h"
}

run_ettm1() {
    echo "========================================"
    echo "  Extreme Contrast Reconstruction: ETTm1"
    echo "========================================"
    python "${ROOT_DIR}/experiments/timer_mi_extreme_reconstruction.py" \
        --mi_result_dir "${MI_RESULT_DIR}" \
        --root_path "${ROOT_DIR}/datasets/" \
        --data_path "ETTm1.csv" \
        --data_type "ETTm1" \
        --seq_len ${SEQ_LEN} \
        --pred_len ${PRED_LEN} \
        --patch_len 48 \
        --batch_size ${BATCH_SIZE} \
        --max_samples ${MAX_SAMPLES} \
        --mask_ratio ${MASK_RATIO} \
        --ckpt_path "${CKPT_PATH}" \
        --d_model ${D_MODEL} \
        --d_ff ${D_FF} \
        --e_layers ${E_LAYERS} \
        --n_heads ${N_HEADS} \
        --dropout ${DROPOUT} \
        --gpu ${GPU_ID} \
        --out_dir "${OUT_DIR}/ettm1" \
        --model_id "ettm1" \
        --n_visual ${N_VISUAL} \
        --freq "t"
}

run_weather() {
    echo "========================================"
    echo "  Extreme Contrast Reconstruction: Weather"
    echo "========================================"
    python "${ROOT_DIR}/experiments/timer_mi_extreme_reconstruction.py" \
        --mi_result_dir "${MI_RESULT_DIR}" \
        --root_path "${ROOT_DIR}/datasets/" \
        --data_path "weather.csv" \
        --data_type "custom" \
        --seq_len ${SEQ_LEN} \
        --pred_len ${PRED_LEN} \
        --patch_len ${PATCH_LEN} \
        --batch_size ${BATCH_SIZE} \
        --max_samples ${MAX_SAMPLES} \
        --mask_ratio ${MASK_RATIO} \
        --ckpt_path "${CKPT_PATH}" \
        --d_model ${D_MODEL} \
        --d_ff ${D_FF} \
        --e_layers ${E_LAYERS} \
        --n_heads ${N_HEADS} \
        --dropout ${DROPOUT} \
        --gpu ${GPU_ID} \
        --out_dir "${OUT_DIR}/weather" \
        --model_id "weather" \
        --n_visual ${N_VISUAL} \
        --freq "h"
}

run_electricity() {
    echo "========================================"
    echo "  Extreme Contrast Reconstruction: Electricity"
    echo "========================================"
    python "${ROOT_DIR}/experiments/timer_mi_extreme_reconstruction.py" \
        --mi_result_dir "${MI_RESULT_DIR}" \
        --root_path "${ROOT_DIR}/datasets/" \
        --data_path "electricity.csv" \
        --data_type "custom" \
        --seq_len ${SEQ_LEN} \
        --pred_len ${PRED_LEN} \
        --patch_len ${PATCH_LEN} \
        --batch_size ${BATCH_SIZE} \
        --max_samples ${MAX_SAMPLES} \
        --mask_ratio ${MASK_RATIO} \
        --ckpt_path "${CKPT_PATH}" \
        --d_model ${D_MODEL} \
        --d_ff ${D_FF} \
        --e_layers ${E_LAYERS} \
        --n_heads ${N_HEADS} \
        --dropout ${DROPOUT} \
        --gpu ${GPU_ID} \
        --out_dir "${OUT_DIR}/electricity" \
        --model_id "electricity" \
        --n_visual ${N_VISUAL} \
        --freq "h"
}

# ─── 主入口 ──────────────────────────────────────────────────────────────────

case "${TASK}" in
    etth1)
        run_etth1
        ;;
    etth2)
        run_etth2
        ;;
    ettm1)
        run_ettm1
        ;;
    weather)
        run_weather
        ;;
    electricity)
        run_electricity
        ;;
    all)
        run_etth1
        run_etth2
        run_ettm1
        run_weather
        run_electricity
        ;;
    *)
        echo "Usage: bash $0 [GPU_ID] [TASK]"
        echo ""
        echo "Tasks:"
        echo "  etth1       - ETTh1 dataset (default)"
        echo "  etth2       - ETTh2 dataset"
        echo "  ettm1       - ETTm1 dataset"
        echo "  weather     - Weather dataset"
        echo "  electricity - Electricity dataset"
        echo "  all        - Run all datasets"
        echo ""
        echo "Examples:"
        echo "  bash $0 0 etth1       # Run ETTh1 on GPU 0"
        echo "  bash $0 1 all         # Run all on GPU 1"
        echo "  bash $0 0             # Run ETTh1 on GPU 0 (default)"
        exit 1
        ;;
esac
