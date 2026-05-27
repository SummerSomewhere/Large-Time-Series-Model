#!/bin/bash
# =============================================================================
# Timer MI引导Token剪枝实验
#
# 模仿 MOMENT mi_pruning_experiment.py，为 Timer 定制。
# 不进行预测头微调（Timer的预测头已预训练）。
#
# Pipeline:
#   1. 从 timer_mi_ksg_pca.py 生成的 JSON 读取每层 I(H,Y) 曲线
#   2. 加载 Timer 模型，在指定层根据 MI 分数做 Top-K Token 剪枝
#   3. 评估不同剪枝率下的 MSE、MAE、推理时间
#
# Prerequisite:
#   先运行 timer_mi_ksg_pca.sh 生成 MI 结果：
#     bash scripts/forecast/timer_mi_ksg_pca.sh
#
# Usage:
#   bash scripts/forecast/timer_mi_pruning_experiment.sh
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

D_MODEL=${D_MODEL:-1024}
D_FF=${D_FF:-2048}
E_LAYERS=${E_LAYERS:-8}
N_HEADS=${N_HEADS:-8}
DROPOUT=${DROPOUT:-0.1}

BATCH_SIZE=${BATCH_SIZE:-64}

# MI 结果目录（timer_mi_ksg_pca.py 输出）
MI_RESULT_DIR=${MI_RESULT_DIR:-./outputs/timer_mi_ksg_pca/Timer_MI_20260516_133108}

MODEL_ID=${MODEL_ID:-etth1}
GPU=${GPU:-0}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/timer_mi_pruning}

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
echo "  gpu            : $GPU"
echo "  output_dir     : $OUTPUT_DIR"
echo "=========================================================="

python experiments/timer_mi_pruning_experiment.py \
    --ckpt_path "$CKPT_PATH" \
    --root_path "$ROOT_PATH" \
    --data_path "$DATA_PATH" \
    --data_type "$DATA_TYPE" \
    --seq_len $SEQ_LEN \
    --pred_len $PRED_LEN \
    --patch_len $PATCH_LEN \
    --stride $STRIDE \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --e_layers $E_LAYERS \
    --n_heads $N_HEADS \
    --dropout $DROPOUT \
    --batch_size $BATCH_SIZE \
    --gpu $GPU \
    --mi_result_dir "$MI_RESULT_DIR" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================================="
echo "  Done. Results saved to: $OUTPUT_DIR"
echo "=========================================================="
