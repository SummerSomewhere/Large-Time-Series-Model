#!/bin/bash
#
# Timer MI-Guided Feature Recycling 实验启动脚本
#
# 模仿 timer_mi_changepoint_alignment.sh，使用 timer_mi_ksg_pca 生成的 MI 结果
# 进行特征回收实验。
#
# Usage:
#   bash experiments/timer_mi_feature_recycling.sh [DATA] [SEQ_LEN] [PATCH_LEN] [PRED_LEN]
#
# Example:
#   bash experiments/timer_mi_feature_recycling.sh ETTh1 672 96 96
#

set -e

# ── 实验配置 ──────────────────────────────────────────────────────────────
DATA=${1:-ETTh1}
SEQ_LEN=${2:-672}
PATCH_LEN=${3:-96}
PRED_LEN=${4:-96}

DATA_PATH="${DATA}.csv"
CKPT_PATH="./checkpoints/Timer_forecast_1.0.ckpt"
ROOT_PATH="./datasets/"
OUT_DIR="./results/timer_mi_feature_recycling/"

# MI 结果目录（由 timer_mi_ksg_pca.py 生成）
MI_RESULT_DIR="./outputs/timer_mi_ksg_pca/Timer_MI_20260516_133108"

# 模型配置
D_MODEL=1024
D_FF=2048
E_LAYERS=8
N_HEADS=8

# 特征回收配置
RECYCLE_LAYER=-1          # -1 表示使用最后一层
BLEND_RATIO=1.0           # 替换时高 MI 表征的混合比例
DO_LAYER_SWEEP=false       # 是否做逐层 sweep（耗时较长）
MAX_EVAL_BATCHES=100      # 最大评估 batch 数（0=全部）
GPU=0

# ── 构建 Python 参数 ─────────────────────────────────────────────────────
PY_ARGS=""
if [ "${DO_LAYER_SWEEP}" = "true" ] || [ "${DO_LAYER_SWEEP}" = "1" ]; then
    PY_ARGS="${PY_ARGS} --do_layer_sweep"
fi

# ── 运行 ─────────────────────────────────────────────────────────────────
echo "========================================"
echo "  Timer MI-Guided Feature Recycling"
echo "========================================"
echo "  Dataset:         ${DATA}"
echo "  seq_len:         ${SEQ_LEN}"
echo "  patch_len:       ${PATCH_LEN}"
echo "  pred_len:        ${PRED_LEN}"
echo "  recycle_layer:   ${RECYCLE_LAYER} (auto = last layer)"
echo "  blend_ratio:     ${BLEND_RATIO}"
echo "  layer_sweep:     ${DO_LAYER_SWEEP}"
echo "  max_eval_batches:${MAX_EVAL_BATCHES}"
echo "  gpu:             ${GPU}"
echo "  mi_result_dir:   ${MI_RESULT_DIR}"
echo "========================================"

python experiments/timer_mi_feature_recycling.py \
    --root_path "${ROOT_PATH}" \
    --data_path "${DATA_PATH}" \
    --data "${DATA}" \
    --seq_len "${SEQ_LEN}" \
    --pred_len "${PRED_LEN}" \
    --patch_len "${PATCH_LEN}" \
    --batch_size 64 \
    --num_workers 4 \
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
    --recycle_layer "${RECYCLE_LAYER}" \
    --blend_ratio "${BLEND_RATIO}" \
    --max_eval_batches "${MAX_EVAL_BATCHES}" \
    --gpu "${GPU}" \
    --out_dir "${OUT_DIR}" \
    ${PY_ARGS}
