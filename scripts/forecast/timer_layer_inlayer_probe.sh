#!/bin/bash
# =============================================================================
# Timer 层间 + 层内 Probe 分析（含 STL 时序分解）
#
# 新增模块一：时序分解（STL Probe）
#   - 利用 STL 分解将原始序列拆解为 Trend / Seasonal / Residual
#   - 对每个分量分别执行层间 Probe + 层内 Probe
#   - 证明模型在不同层对不同时序特征的敏感度不同
#
# Pipeline:
#   1. 从 timer_mi_ksg_pca.py 生成的 JSON 读取每层 I(H,Y) 曲线
#   2. 加载 Timer 模型，提取测试集每层 token hidden state
#   3. 对原始序列做 STL 分解 → Trend / Seasonal / Residual
#   4. 对每个分量跑层间 Probe + 层内 Probe
#   5. 跨分量对比绘图
#
# Prerequisite:
#   先运行 timer_mi_ksg_pca.sh 生成 JSON：
#     bash scripts/forecast/timer_mi_ksg_pca.sh
#
# Usage:
#   bash scripts/forecast/timer_layer_inlayer_probe.sh
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

# STL 参数
STL_PERIOD=${STL_PERIOD:-24}       # ETTh1 hourly: daily=24, weekly=168
STL_TARGET_VAR=${STL_TARGET_VAR:-0} # 分解第几个变量（默认第 1 列）

# Probe 参数
PROBE_EPOCHS=${PROBE_EPOCHS:-100}
PROBE_LR=${PROBE_LR:-1e-3}

MODEL_ID=${MODEL_ID:-etth1}
GPU=${GPU:-0}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/timer_layer_inlayer_probe}

# ── GPU setup ────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=$GPU

# ── Navigate to project root ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "  Timer Layer-wise & In-layer Probe (with STL)"
echo "=========================================================="
echo "  ckpt_path      : $CKPT_PATH"
echo "  data_path      : $DATA_PATH"
echo "  seq_len        : $SEQ_LEN"
echo "  pred_len       : $PRED_LEN"
echo "  patch_len      : $PATCH_LEN"
echo "  e_layers       : $E_LAYERS"
echo "  mi_result_dir  : $MI_RESULT_DIR"
echo "  stl_period     : $STL_PERIOD"
echo "  stl_target_var : $STL_TARGET_VAR"
echo "  probe_epochs   : $PROBE_EPOCHS"
echo "  gpu            : $GPU"
echo "=========================================================="

python experiments/timer_layer_inlayer_probe.py \
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
    --freq h \
    --out_dir "$OUTPUT_DIR" \
    --model_id $MODEL_ID \
    --mi_result_dir "$MI_RESULT_DIR" \
    --probe_epochs $PROBE_EPOCHS \
    --probe_lr $PROBE_LR \
    --stl_period $STL_PERIOD \
    --stl_target_var $STL_TARGET_VAR

echo ""
echo "=========================================================="
echo "  Done. Results saved to: $OUTPUT_DIR"
echo "=========================================================="
