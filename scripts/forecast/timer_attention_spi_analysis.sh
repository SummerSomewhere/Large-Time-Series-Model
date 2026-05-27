#!/bin/bash
# =============================================================================
# Timer Attention Hub 验证实验 — 多层版
#
# 从 timer_mi_ksg_pca.py 生成的 JSON 读取 I(h_x, h_y) 曲线，
# 与注意力入度做相关性分析。
#
# Pipeline:
#   1. 从 JSON 读取每层的 MI 曲线（I(h_x, h_y) 或 I(X, H)）
#   2. 加载预训练 Timer，通过 attention hook 提取所有层的
#      自注意力权重矩阵 [B, H, S, S]
#   3. 对注意力头维度求均值，再沿 Query 维对列求和，计算 in-degree
#   4. 将 MI 曲线 tile 到每个样本，对齐后计算 Pearson/Spearman 相关系数
#   5. 绘制散点图和多层对比柱状图
#
# Prerequisite:
#   先运行 timer_mi_ksg_pca.sh 生成 JSON：
#     bash scripts/forecast/timer_mi_ksg_pca.sh
#
# Usage:
#   bash scripts/forecast/timer_attention_spi_analysis.sh
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
MI_METRIC=${MI_METRIC:-IHY}   # IHY 或 IXH

MODEL_ID=${MODEL_ID:-etth1}
GPU=${GPU:-0}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/timer_attention_spi}

# ── GPU setup ────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=$GPU

# ── Navigate to project root ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "  Timer Attention Hub — Multi-Layer Analysis"
echo "  Metric: I(h_x, h_y) vs In-degree Attention"
echo "=========================================================="
echo "  ckpt_path     : $CKPT_PATH"
echo "  data_path     : $DATA_PATH"
echo "  seq_len       : $SEQ_LEN"
echo "  pred_len      : $PRED_LEN"
echo "  patch_len     : $PATCH_LEN"
echo "  stride        : $STRIDE"
echo "  e_layers      : $E_LAYERS"
echo "  mi_result_dir : $MI_RESULT_DIR"
echo "  mi_metric     : $MI_METRIC"
echo "  batch_size    : $BATCH_SIZE"
echo "  gpu           : $GPU"
echo "=========================================================="

python experiments/timer_attention_spi_analysis.py \
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
    --mi_metric $MI_METRIC

echo ""
echo "=========================================================="
echo "  Done. Results saved to: $OUTPUT_DIR"
echo "=========================================================="
