#!/bin/bash
# =============================================================================
# Timer 层间 MI 分析 (KSG Estimator + PCA)
#
# Pipeline:
#   1. Load pre-trained Timer checkpoint
#   2. Extract per-layer hidden states on test set
#   3. Compute layer-wise, patch-wise MI with future sequence
#   4. Save MI matrix and identify high-MI patches
#   5. Output JSON for alignment loss
#
# Features:
#   - Each subplot includes a per-layer mean MI bar chart for clear comparison
#   - SPI = I(H,Y) / (I(X,H) + bias), where bias = 0.15 * mean(I(X,H))
#
# Usage:
#   bash scripts/forecast/timer_mi_ksg_pca.sh
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
STRIDE=${STRIDE:-1}

D_MODEL=${D_MODEL:-1024}
D_FF=${D_FF:-2048}
E_LAYERS=${E_LAYERS:-8}
N_HEADS=${N_HEADS:-8}
DROPOUT=${DROPOUT:-0.1}

BATCH_SIZE=${BATCH_SIZE:-1024}
SAMPLE_RATIO=${SAMPLE_RATIO:-0.1}
K_NEIGHBORS=${K_NEIGHBORS:-3}
PCA_DIM=${PCA_DIM:-32}

MODEL_ID=${MODEL_ID:-etth1}
GPU=${GPU:-0}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/timer_mi_ksg_pca}

# ── GPU setup ────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=$GPU

# ── Navigate to project root ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "  Timer MI Analysis (KSG Estimator + PCA)"
echo "=========================================================="
echo "  ckpt_path   : $CKPT_PATH"
echo "  data_path   : $DATA_PATH"
echo "  seq_len     : $SEQ_LEN"
echo "  pred_len    : $PRED_LEN"
echo "  patch_len   : $PATCH_LEN"
echo "  stride      : $STRIDE"
echo "  e_layers    : $E_LAYERS"
echo "  pca_dim     : $PCA_DIM"
echo "  batch_size  : $BATCH_SIZE"
echo "  sample_ratio: $SAMPLE_RATIO"
echo "  gpu         : $GPU"
echo "=========================================================="

python experiments/timer_mi_ksg_pca.py \
    --ckpt_path "$CKPT_PATH" \
    --root_path "$ROOT_PATH" \
    --data_path "$DATA_PATH" \
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
    --sample_ratio $SAMPLE_RATIO \
    --k_neighbors $K_NEIGHBORS \
    --pca_dim $PCA_DIM \
    --gpu $GPU \
    --out_dir "$OUTPUT_DIR" \
    --data_type "$DATA_TYPE" \
    --freq h \
    --model_id $MODEL_ID

echo ""
echo "=========================================================="
echo "  Done. Results saved to: $OUTPUT_DIR"
echo "=========================================================="
