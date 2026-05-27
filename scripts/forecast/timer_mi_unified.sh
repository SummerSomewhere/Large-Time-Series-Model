#!/bin/bash
# =============================================================================
# Timer MI Analysis — Unified launcher (KSG / fastMI rpy2 / fastMI pure Python)
#
# Supported --mi_method values:
#   ksg           — KSG kNN estimator (default, no extra deps)
#   fastmi_rpy2   — fastMI via R package (requires rpy2 + fastMI R package)
#   fastmi_python — fastMI pure Python (copula + FFT, no extra deps)
#   all           — run ksg + fastmi_python (+ fastmi_rpy2 if available)
#
# Examples:
#   bash scripts/forecast/timer_mi_unified.sh --mi_method ksg
#   bash scripts/forecast/timer_mi_unified.sh --mi_method fastmi_python
#   bash scripts/forecast/timer_mi_unified.sh --mi_method all
# =============================================================================

set -e

# ── Hyperparameters ──────────────────────────────────────────────────────────
CKPT_PATH=${CKPT_PATH:-checkpoints/Timer_forecast_1.0.ckpt}
ROOT_PATH=${ROOT_PATH:-./datasets/}
DATA=${DATA:-ETTh1}
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

BATCH_SIZE=${BATCH_SIZE:-64}
NUM_BATCHES=${NUM_BATCHES:-8}
K_NEIGHBORS=${K_NEIGHBORS:-5}
PCA_DIM=${PCA_DIM:-32}
FASTMI_GRID_SIZE=${FASTMI_GRID_SIZE:-256}

MI_METHOD=${MI_METHOD:-all}
MODEL_ID=${MODEL_ID:-etth1}
GPU=${GPU:-0}
OUTPUT_DIR=${OUTPUT_DIR:-./results/timer_mi_unified}

# ── GPU setup ────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=$GPU

# ── Navigate to project root ───────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================================="
echo "  Timer MI Analysis — Unified"
echo "=============================================================="
echo "  mi_method     : $MI_METHOD"
echo "  ckpt_path     : $CKPT_PATH"
echo "  data          : $DATA"
echo "  data_path     : $DATA_PATH"
echo "  seq_len       : $SEQ_LEN"
echo "  pred_len      : $PRED_LEN"
echo "  patch_len     : $PATCH_LEN"
echo "  e_layers      : $E_LAYERS"
echo "  pca_dim       : $PCA_DIM"
echo "  k_neighbors   : $K_NEIGHBORS"
echo "  grid_size     : $FASTMI_GRID_SIZE"
echo "  batch_size    : $BATCH_SIZE"
echo "  num_batches   : $NUM_BATCHES"
echo "  gpu           : $GPU"
echo "  output_dir    : $OUTPUT_DIR"
echo "=============================================================="

python experiments/timer_mi_unified.py \
    --ckpt_path "$CKPT_PATH" \
    --root_path "$ROOT_PATH" \
    --data "$DATA" \
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
    --num_batches $NUM_BATCHES \
    --k_neighbors $K_NEIGHBORS \
    --pca_dim $PCA_DIM \
    --fastmi_grid_size $FASTMI_GRID_SIZE \
    --mi_method "$MI_METHOD" \
    --gpu $GPU \
    --out_dir "$OUTPUT_DIR" \
    --freq h \
    --model_id "$MODEL_ID"

echo ""
echo "=============================================================="
echo "  Done. Results saved to: $OUTPUT_DIR"
echo "=============================================================="
