#!/bin/bash
# =============================================================================
# Timer MI引导Token剪枝实验
#
# 使用预计算的 MI 曲线，对指定层进行 Top-K Token 物理剪枝
# 评估 MSE 损失和推理加速效果
#
# Pipeline:
#   1. 读取 timer_mi_ksg_pca.py 生成的 global_mi_peaks_*.json
#   2. 加载 Timer 模型，手动实现 forward（支持逐层 token 剪枝）
#   3. 逐层剪枝：保留 top-K% 高MI token，评估 MSE 和推理时间
#   4. 全层剪枝：所有层同时按相同比例剪枝
#
# Prerequisite:
#   先运行 timer_mi_ksg_pca.sh 生成 MI 结果：
#     bash scripts/forecast/timer_mi_ksg_pca.sh
#
# Usage:
#   bash scripts/forecast/timer_mi_pruning.sh
# =============================================================================

set -e

# ── Hyperparameters ──────────────────────────────────────────────────────────
CKPT_PATH=${CKPT_PATH:-checkpoints/Timer_forecast_1.0.ckpt}
ROOT_PATH=${ROOT_PATH:-./datasets/}
DATA_PATH=${DATA_PATH:-ETTh1.csv}
DATA_TYPE=${DATA_TYPE:-ETTh1}

SEQ_LEN=${SEQ_LEN:-672}
PRED_LEN=${PRED_LEN:-96}
PATCH_LEN=${PATCH_LEN:-96}
STRIDE=${STRIDE:-96}

E_LAYERS=${E_LAYERS:-8}

BATCH_SIZE=${BATCH_SIZE:-32}

# MI 结果目录（timer_mi_ksg_pca.py 输出）
MI_RESULT_DIR=${MI_RESULT_DIR:-./outputs/timer_mi_ksg_pca/Timer_MI_20260516_133108}

# 剪枝配置
DROP_RATES=${DROP_RATES:-0.0,0.1,0.2,0.3,0.4,0.5}

# 微调配置
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-1}
FINETUNE_LR=${FINETUNE_LR:-0.001}

GPU=${GPU:-0}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/timer_mi_pruning}
MODEL_ID=${MODEL_ID:-etth1}

# ── GPU setup ────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=$GPU

# ── Navigate to project root ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "  Timer MI引导Token剪枝实验"
echo "=========================================================="
echo "  ckpt_path      : $CKPT_PATH"
echo "  data_path      : $DATA_PATH"
echo "  seq_len        : $SEQ_LEN"
echo "  pred_len       : $PRED_LEN"
echo "  patch_len      : $PATCH_LEN"
echo "  e_layers       : $E_LAYERS"
echo "  mi_result_dir  : $MI_RESULT_DIR"
echo "  drop_rates     : $DROP_RATES"
echo "  finetune_epochs: $FINETUNE_EPOCHS"
echo "  finetune_lr    : $FINETUNE_LR"
echo "  gpu            : $GPU"
echo "  output_dir     : $OUTPUT_DIR"
echo "=========================================================="

python experiments/timer_mi_pruning_experiment.py \
    --ckpt_path "$CKPT_PATH" \
    --root_path "$ROOT_PATH" \
    --data_path "$DATA_PATH" \
    --seq_len $SEQ_LEN \
    --pred_len $PRED_LEN \
    --patch_len $PATCH_LEN \
    --stride $STRIDE \
    --e_layers $E_LAYERS \
    --batch_size $BATCH_SIZE \
    --gpu $GPU \
    --output_dir "$OUTPUT_DIR" \
    --mi_dir "$MI_RESULT_DIR" \
    --drop_rates "$DROP_RATES" \
    --finetune_epochs $FINETUNE_EPOCHS \
    --finetune_lr $FINETUNE_LR

echo ""
echo "=========================================================="
echo "  Done. Results saved to: $OUTPUT_DIR"
echo "=========================================================="
