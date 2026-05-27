#!/bin/sh
#
# Pipeline: (1) finetune native Timer → test  (2) finetune Geo-HPE + v·(1+a·k) → test  (3) attention figure.
#
# Steps (1)(2): each run.py uses --is_finetuning 1, so run.py runs exp.finetune() then exp.test()
# (metrics under ./test_results/). Step (3) only loads CKPT_GEO and plots (no second test).
#
# Usage:
#   bash ./scripts/forecast/ETTh1_finetune_then_attention_figure.sh
#
# Optional:
#   SKIP_FINETUNE=1     — only step (3); set CKPT_GEO or use latest dir for MODEL_ID_GEO (L×H figure uses geo ckpt only).
#   USE_TORCHRUN=0      — single-process python (one GPU). Default 1: torchrun + DDP on NPROC_PER_NODE GPUs.
#   NPROC_PER_NODE      — torchrun workers (default 3; align with CUDA_VISIBLE_DEVICES count).
#   PRETRAIN_CKPT=...   — base Timer weight (default: checkpoints/Timer_forecast_1.0.ckpt).

# Default: physical GPUs 3,4,5 (processes see cuda:0,1,2).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
# Use all visible GPUs via torchrun (override USE_TORCHRUN=0 for single-GPU python).
USE_TORCHRUN="${USE_TORCHRUN:-1}"

cd "$(dirname "$0")/../.." || exit 1

# ========= shared with ETTh1.sh / ETTh1_geometric_hpe.sh =========
PRETRAIN_CKPT="${PRETRAIN_CKPT:-checkpoints/Timer_forecast_1.0.ckpt}"
ROOT="${ROOT:-./datasets}"
DATA_CSV="${DATA_CSV:-weather.csv}"
DATA_NAME="${DATA_NAME:-weather}"
SEED="${SEED:-1}"
SUBSET_RAND_RATIO="${SUBSET_RAND_RATIO:-1}"
GPU="${GPU:-0}"
FEATURES="${FEATURES:-M}"
DROPOUT="${DROPOUT:-0.1}"
ACTIVATION="${ACTIVATION:-gelu}"
EMBED="${EMBED:-timeF}"
FREQ="${FREQ:-h}"
STRIDE="${STRIDE:-1}"

SEQ_LEN="${SEQ_LEN:-672}"
PATCH_LEN="${PATCH_LEN:-96}"
LABEL_LEN="${LABEL_LEN:-576}"
PRED_LEN="${PRED_LEN:-96}"
OUTPUT_LEN="${OUTPUT_LEN:-96}"

D_MODEL="${D_MODEL:-1024}"
D_FF="${D_FF:-2048}"
E_LAYERS="${E_LAYERS:-8}"
N_HEADS="${N_HEADS:-8}"
FACTOR="${FACTOR:-3}"
DES_BASE="${DES_BASE:-ETTh1ft}"

FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PATIENCE="${PATIENCE:-15}"

MODEL_ID_TIMER="${MODEL_ID_TIMER:-etth1_timer_native}"
MODEL_ID_GEO="${MODEL_ID_GEO:-etth1_timer_v1plusak}"

GEOMETRIC_HPE_PERIODS="${GEOMETRIC_HPE_PERIODS:-24,168}"
GEOMETRIC_HPE_CURV_PHASE_SCALE="${GEOMETRIC_HPE_CURV_PHASE_SCALE:-1.0}"
GEOMETRIC_HPE_PE_CURV_WEIGHTED="${GEOMETRIC_HPE_PE_CURV_WEIGHTED:-0}"
GEOMETRIC_HPE_LINEAR_K_A_INIT="${GEOMETRIC_HPE_LINEAR_K_A_INIT:-0.01}"
GEOMETRIC_HPE_PE_CURV_B_INIT="${GEOMETRIC_HPE_PE_CURV_B_INIT:-0.01}"
GEOMETRIC_HPE_AB_FIXED="${GEOMETRIC_HPE_AB_FIXED:-0}"

export OUT_DIR="${OUT_DIR:-./figure_attention_geo_layers_heads}"
export OUT_NAME="${OUT_NAME:-attention_layers_heads.png}"

SKIP_FINETUNE="${SKIP_FINETUNE:-0}"

latest_ckpt_pth() {
  mid="$1"
  d=$(ls -td checkpoints/forecast_"${mid}"_Timer_"${DATA_NAME}"_* 2>/dev/null | head -1)
  if [ -z "$d" ] || [ ! -f "$d/checkpoint.pth" ]; then
    echo ""
    return 1
  fi
  echo "$d/checkpoint.pth"
}

# run_train <model_id> <des> -- extra args ...  (pass-through to run.py)
run_train() {
  mid="$1"
  des="$2"
  shift 2

  if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "Missing pretrained checkpoint: $PRETRAIN_CKPT" >&2
    exit 1
  fi

  common="--task_name forecast \
    --is_finetuning 1 \
    --is_training 1 \
    --seed $SEED \
    --ckpt_path $PRETRAIN_CKPT \
    --root_path $ROOT \
    --data_path $DATA_CSV \
    --data $DATA_NAME \
    --model_id $mid \
    --model Timer \
    --features $FEATURES \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --output_len $OUTPUT_LEN \
    --e_layers $E_LAYERS \
    --factor $FACTOR \
    --des $des \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --dropout $DROPOUT \
    --activation $ACTIVATION \
    --embed $EMBED \
    --freq $FREQ \
    --stride $STRIDE \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_workers $NUM_WORKERS \
    --patch_len $PATCH_LEN \
    --finetune_epochs $FINETUNE_EPOCHS \
    --patience $PATIENCE \
    --train_test 0 \
    --subset_rand_ratio $SUBSET_RAND_RATIO \
    --itr 1 \
    --gpu $GPU \
    --use_ims"

  if [ "$USE_TORCHRUN" = "1" ]; then
    # shellcheck disable=SC2086
    torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" run.py $common --use_multi_gpu "$@"
  else
    # shellcheck disable=SC2086
    python run.py $common "$@"
  fi
}

set -e

if [ "$SKIP_FINETUNE" != "1" ]; then
  echo "========== [1/3] Finetune native Timer → checkpoint + test =========="
  run_train "$MODEL_ID_TIMER" "${DES_BASE}_native" --geometric_hpe 0

  echo "========== [2/3] Finetune Timer + v(1+a·k) → checkpoint + test =========="
  run_train "$MODEL_ID_GEO" "${DES_BASE}_v1plusak" \
    --geometric_hpe 1 \
    --geometric_hpe_periods "$GEOMETRIC_HPE_PERIODS" \
    --geometric_hpe_curv_phase_scale "$GEOMETRIC_HPE_CURV_PHASE_SCALE" \
    --geometric_hpe_curv_residual 0 \
    --geometric_hpe_res_lambda 0 \
    --geometric_hpe_linear_k_patch_scale 1 \
    --geometric_hpe_linear_k_a_init "$GEOMETRIC_HPE_LINEAR_K_A_INIT" \
    --geometric_hpe_pe_curv_weighted "$GEOMETRIC_HPE_PE_CURV_WEIGHTED" \
    --geometric_hpe_pe_curv_b_init "$GEOMETRIC_HPE_PE_CURV_B_INIT" \
    --geometric_hpe_ab_fixed "$GEOMETRIC_HPE_AB_FIXED"
fi

CKPT_TIMER="${CKPT_TIMER:-$(latest_ckpt_pth "$MODEL_ID_TIMER")}"
CKPT_GEO="${CKPT_GEO:-$(latest_ckpt_pth "$MODEL_ID_GEO")}"

if [ -z "$CKPT_TIMER" ] || [ ! -f "$CKPT_TIMER" ]; then
  echo "Could not resolve CKPT_TIMER. Finetune first or: export CKPT_TIMER=.../checkpoint.pth" >&2
  exit 1
fi
if [ -z "$CKPT_GEO" ] || [ ! -f "$CKPT_GEO" ]; then
  echo "Could not resolve CKPT_GEO. Finetune first or: export CKPT_GEO=.../checkpoint.pth" >&2
  exit 1
fi

export CKPT_TIMER CKPT_GEO
export ROOT DATA_CSV DATA_NAME
export SEQ_LEN PATCH_LEN LABEL_LEN PRED_LEN OUTPUT_LEN
export D_MODEL D_FF E_LAYERS N_HEADS FACTOR
export DROPOUT ACTIVATION EMBED FREQ STRIDE FEATURES
export SEED SUBSET_RAND_RATIO NUM_WORKERS GPU
export GEOMETRIC_HPE_PERIODS GEOMETRIC_HPE_CURV_PHASE_SCALE
export GEOMETRIC_HPE_CURV_RESIDUAL="${GEOMETRIC_HPE_CURV_RESIDUAL:-0}"
export GEOMETRIC_HPE_RES_LAMBDA="${GEOMETRIC_HPE_RES_LAMBDA:-0}"
export GEOMETRIC_HPE_LINEAR_K_A_INIT GEOMETRIC_HPE_PE_CURV_WEIGHTED
export GEOMETRIC_HPE_PE_CURV_B_INIT GEOMETRIC_HPE_AB_FIXED
export METRICS_MAX_BATCHES="${METRICS_MAX_BATCHES:-0}"

echo "========== [3/3] Attention figure (L×H, Geo v·(1+a·k), PE not scaled) =========="
echo "CKPT_GEO=$CKPT_GEO"

export CKPT_PATH="${CKPT_GEO}"
export RUN_FINETUNE=0
bash ./scripts/forecast/ETTh1_attention_figure.sh
