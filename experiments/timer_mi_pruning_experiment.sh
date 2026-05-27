#!/usr/bin/env bash
#
# Timer MI引导Attention剪枝实验 启动脚本
#
# 剪枝策略: 低MI Token的Attention分数置零
# 数据集: ETTh1 (默认) / ETTm1
#
# Usage:
#   bash experiments/timer_mi_pruning_experiment.sh
#   bash experiments/timer_mi_pruning_experiment.sh --data etth2
#   bash experiments/timer_mi_pruning_experiment.sh --gpu 1
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA="${1:-etth1}"
GPU="${2:-0}"

case "${DATA}" in
    etth1)
        DATA_PATH="./datasets/ETTh1.csv"
        DATA_TYPE="ETTh1"
        MI_DIR="./outputs/timer_mi_ksg_pca/Timer_MI_20260516_133108"
        SEQ_LEN=672
        PRED_LEN=96
        PATCH_LEN=96
        ;;
    etth2)
        DATA_PATH="./datasets/ETTh2.csv"
        DATA_TYPE="ETTh2"
        MI_DIR="./timer_mi_ksg_pca/Timer_MI_20260514_103927"
        SEQ_LEN=672
        PRED_LEN=96
        PATCH_LEN=96
        ;;
    ettm1)
        DATA_PATH="./datasets/ETTm1.csv"
        DATA_TYPE="ETTm1"
        MI_DIR="./timer_mi_ksg_pca/Timer_MI_20260515_050931"
        SEQ_LEN=672
        PRED_LEN=96
        PATCH_LEN=96
        ;;
    custom)
        DATA_PATH="./datasets/ETTh1.csv"
        DATA_TYPE="custom"
        MI_DIR="./timer_mi_ksg_pca/Timer_MI_20260514_073311"
        SEQ_LEN=672
        PRED_LEN=96
        PATCH_LEN=96
        ;;
    *)
        echo "Unknown dataset: ${DATA}"
        echo "Available: etth1 (default), etth2, ettm1, custom"
        exit 1
        ;;
esac

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./results/timer_mi_pruning_${DATA}_${TIMESTAMP}/"

echo "============================================================"
echo "Timer MI引导Attention剪枝实验"
echo "============================================================"
echo "  数据集      : ${DATA} (${DATA_PATH})"
echo "  数据类型    : ${DATA_TYPE}"
echo "  MI目录      : ${MI_DIR}"
echo "  序列长度    : ${SEQ_LEN}"
echo "  预测长度    : ${PRED_LEN}"
echo "  Patch长度   : ${PATCH_LEN}"
echo "  输出目录    : ${OUTPUT_DIR}"
echo "  GPU设备     : cuda:${GPU}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_ROOT}"

python experiments/timer_mi_pruning_experiment.py \
    --mi_dir "${MI_DIR}" \
    --data_path "${DATA_PATH}" \
    --data_type "${DATA_TYPE}" \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --patch_len ${PATCH_LEN} \
    --ckpt_path "checkpoints/Timer_forecast_1.0.ckpt" \
    --d_model 1024 \
    --d_ff 2048 \
    --e_layers 8 \
    --n_heads 8 \
    --dropout 0.1 \
    --batch_size 64 \
    --all_layers \
    --mask_ratios 0.0 0.1 0.2 0.3 0.4 0.5 \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42 \
    --gpu ${GPU} \
    --freq "h"

echo ""
echo "============================================================"
echo "实验完成！"
echo "结果目录: ${OUTPUT_DIR}"
echo "============================================================"
