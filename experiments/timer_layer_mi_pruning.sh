#!/bin/bash
#
# Timer MI 变化率引导的层剪枝实验启动脚本
#
# 使用方法:
#   bash experiments/timer_layer_mi_pruning.sh
#   bash experiments/timer_layer_mi_pruning.sh <gpu_id>
#

set -e

GPU_ID="${1:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ─── 配置 ────────────────────────────────────────────────────────────────────
CKPT_PATH="${ROOT_DIR}/checkpoints/Timer_forecast_1.0.ckpt"
MI_RESULT_DIR="${ROOT_DIR}/outputs/timer_mi_ksg_pca/Timer_MI_20260516_133108"
OUT_DIR="${ROOT_DIR}/results/timer_layer_mi_pruning"

# ─── 数据集配置 ──────────────────────────────────────────────────────────────
SEQ_LEN=672
PRED_LEN=96
LABEL_LEN=48
PATCH_LEN=96
BATCH_SIZE=64
MAX_SAMPLES=0

# ─── 方法配置 ────────────────────────────────────────────────────────────────
# 方法一：百分位数阈值（数据驱动）
PERCENTILES="10,20,30,40,50,60,70,80,90,95"

# ─── 模型配置 ────────────────────────────────────────────────────────────────
D_MODEL=1024
D_FF=2048
E_LAYERS=8
N_HEADS=8
DROPOUT=0.1

# ─── 实验函数 ────────────────────────────────────────────────────────────────

run_etth1() {
    echo "========================================"
    echo "  Timer Layer MI Pruning: ETTh1"
    echo "========================================"
    python "${ROOT_DIR}/experiments/timer_layer_mi_pruning.py" \
        --mi_result_dir "${MI_RESULT_DIR}" \
        --root_path "${ROOT_DIR}/datasets/" \
        --data_path "ETTh1.csv" \
        --data "ETTh1" \
        --data_type "ETTh1" \
        --seq_len ${SEQ_LEN} \
        --pred_len ${PRED_LEN} \
        --label_len ${LABEL_LEN} \
        --patch_len ${PATCH_LEN} \
        --batch_size ${BATCH_SIZE} \
        --max_samples ${MAX_SAMPLES} \
        --percentiles "${PERCENTILES}" \
        --ckpt_path "${CKPT_PATH}" \
        --d_model ${D_MODEL} \
        --d_ff ${D_FF} \
        --e_layers ${E_LAYERS} \
        --n_heads ${N_HEADS} \
        --dropout ${DROPOUT} \
        --gpu ${GPU_ID} \
        --out_dir "${OUT_DIR}" \
        --model_id "etth1" \
        --freq "h"
}

run_etth2() {
    echo "========================================"
    echo "  Timer Layer MI Pruning: ETTh2"
    echo "========================================"
    python "${ROOT_DIR}/experiments/timer_layer_mi_pruning.py" \
        --mi_result_dir "${MI_RESULT_DIR}" \
        --root_path "${ROOT_DIR}/datasets/" \
        --data_path "ETTh2.csv" \
        --data "ETTh2" \
        --data_type "ETTh2" \
        --seq_len ${SEQ_LEN} \
        --pred_len ${PRED_LEN} \
        --label_len ${LABEL_LEN} \
        --patch_len ${PATCH_LEN} \
        --batch_size ${BATCH_SIZE} \
        --max_samples ${MAX_SAMPLES} \
        --percentiles "${PERCENTILES}" \
        --ckpt_path "${CKPT_PATH}" \
        --d_model ${D_MODEL} \
        --d_ff ${D_FF} \
        --e_layers ${E_LAYERS} \
        --n_heads ${N_HEADS} \
        --dropout ${DROPOUT} \
        --gpu ${GPU_ID} \
        --out_dir "${OUT_DIR}" \
        --model_id "etth2" \
        --freq "h"
}

run_ettm1() {
    echo "========================================"
    echo "  Timer Layer MI Pruning: ETTm1"
    echo "========================================"
    python "${ROOT_DIR}/experiments/timer_layer_mi_pruning.py" \
        --mi_result_dir "${MI_RESULT_DIR}" \
        --root_path "${ROOT_DIR}/datasets/" \
        --data_path "ETTm1.csv" \
        --data "ETTm1" \
        --data_type "ETTm1" \
        --seq_len ${SEQ_LEN} \
        --pred_len ${PRED_LEN} \
        --label_len ${LABEL_LEN} \
        --patch_len ${PATCH_LEN} \
        --batch_size ${BATCH_SIZE} \
        --max_samples ${MAX_SAMPLES} \
        --percentiles "${PERCENTILES}" \
        --ckpt_path "${CKPT_PATH}" \
        --d_model ${D_MODEL} \
        --d_ff ${D_FF} \
        --e_layers ${E_LAYERS} \
        --n_heads ${N_HEADS} \
        --dropout ${DROPOUT} \
        --gpu ${GPU_ID} \
        --out_dir "${OUT_DIR}" \
        --model_id "ettm1" \
        --freq "t"
}

run_weather() {
    echo "========================================"
    echo "  Timer Layer MI Pruning: Weather"
    echo "========================================"
    python "${ROOT_DIR}/experiments/timer_layer_mi_pruning.py" \
        --mi_result_dir "${MI_RESULT_DIR}" \
        --root_path "${ROOT_DIR}/datasets/" \
        --data_path "weather.csv" \
        --data "custom" \
        --data_type "custom" \
        --seq_len ${SEQ_LEN} \
        --pred_len ${PRED_LEN} \
        --label_len ${LABEL_LEN} \
        --patch_len ${PATCH_LEN} \
        --batch_size ${BATCH_SIZE} \
        --max_samples ${MAX_SAMPLES} \
        --percentiles "${PERCENTILES}" \
        --ckpt_path "${CKPT_PATH}" \
        --d_model ${D_MODEL} \
        --d_ff ${D_FF} \
        --e_layers ${E_LAYERS} \
        --n_heads ${N_HEADS} \
        --dropout ${DROPOUT} \
        --gpu ${GPU_ID} \
        --out_dir "${OUT_DIR}" \
        --model_id "weather" \
        --freq "h"
}

run_electricity() {
    echo "========================================"
    echo "  Timer Layer MI Pruning: Electricity"
    echo "========================================"
    python "${ROOT_DIR}/experiments/timer_layer_mi_pruning.py" \
        --mi_result_dir "${MI_RESULT_DIR}" \
        --root_path "${ROOT_DIR}/datasets/" \
        --data_path "electricity.csv" \
        --data "custom" \
        --data_type "custom" \
        --seq_len ${SEQ_LEN} \
        --pred_len ${PRED_LEN} \
        --label_len ${LABEL_LEN} \
        --patch_len ${PATCH_LEN} \
        --batch_size ${BATCH_SIZE} \
        --max_samples ${MAX_SAMPLES} \
        --percentiles "${PERCENTILES}" \
        --ckpt_path "${CKPT_PATH}" \
        --d_model ${D_MODEL} \
        --d_ff ${D_FF} \
        --e_layers ${E_LAYERS} \
        --n_heads ${N_HEADS} \
        --dropout ${DROPOUT} \
        --gpu ${GPU_ID} \
        --out_dir "${OUT_DIR}" \
        --model_id "electricity" \
        --freq "h"
}

# ─── 主入口 ──────────────────────────────────────────────────────────────────

TASK="${2:-etth1}"

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
        echo "  etth1      - ETTh1 dataset (default)"
        echo "  etth2      - ETTh2 dataset"
        echo "  ettm1      - ETTm1 dataset"
        echo "  weather    - Weather dataset"
        echo "  electricity- Electricity dataset"
        echo "  all        - Run all datasets"
        echo ""
        echo "Examples:"
        echo "  bash $0 0 etth1    # Run ETTh1 on GPU 0"
        echo "  bash $0 1 all     # Run all datasets on GPU 1"
        echo "  bash $0 0         # Run ETTh1 on GPU 0 (default)"
        exit 1
        ;;
esac
