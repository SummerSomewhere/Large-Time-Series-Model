#!/bin/sh
#
# Attention compare figure — parameter defaults follow scripts/forecast/ETTh1.sh line-by-line where applicable.
# (Figure script uses single-process python + test batch_size=1; ETTh1.sh uses torchrun + train batch_size.)
#
# ETTh1.sh reference (abbreviated):
#   CUDA_VISIBLE_DEVICES=4,5  torchrun ... run.py  --seed 1  --root_path ./datasets  --data_path weather.csv
#   --data weather  --seq_len 672  --label_len 576  --pred_len 96  --output_len 96  --patch_len 96
#   --e_layers 8  --factor 3  --des Exp  --d_model 1024  --d_ff 2048  --batch_size 1024  --learning_rate 3e-5
#   --num_workers 4  --subset_rand_ratio 1  --gpu 0  --use_ims  --use_multi_gpu
#
# Full pipeline: scripts/forecast/ETTh1_finetune_then_attention_figure.sh

# Match ETTh1.sh: Physical GPUs 3,4 — processes see cuda:0,1 under torchrun; single python uses first visible GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

cd "$(dirname "$0")/../.." || exit 1

# ========= checkpoints (ETTh1.sh: ckpt_path=checkpoints/Timer_forecast_1.0.ckpt) =========
CKPT_TIMER_DEFAULT="checkpoints/Timer_forecast_1.0.ckpt"
CKPT_GEO_DEFAULT="checkpoints/Timer_forecast_1.0.ckpt"
CKPT_TIMER="${CKPT_TIMER:-$CKPT_TIMER_DEFAULT}"
CKPT_GEO="${CKPT_GEO:-$CKPT_GEO_DEFAULT}"

# ========= same variable names / values as ETTh1.sh =========
ROOT="${ROOT:-./datasets}"
DATA_CSV="${DATA_CSV:-weather.csv}"
DATA_NAME="${DATA_NAME:-weather}"
FEATURES="${FEATURES:-M}"

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
DROPOUT="${DROPOUT:-0.1}"
ACTIVATION="${ACTIVATION:-gelu}"
EMBED="${EMBED:-timeF}"
FREQ="${FREQ:-h}"
STRIDE="${STRIDE:-1}"
SUBSET_RAND_RATIO="${SUBSET_RAND_RATIO:-1}"

SEED="${SEED:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU="${GPU:-0}"

# ETTh1.sh uses --learning_rate 3e-5 for training; not used by inference figure (documented for parity).
# LEARNING_RATE="${LEARNING_RATE:-3e-5}"

# ========= optional extras (not in ETTh1.sh) =========
PERIODIC_EMBEDDING_BRANCH="${PERIODIC_EMBEDDING_BRANCH:-0}"
PERIODIC_EMB_BANK_DIM="${PERIODIC_EMB_BANK_DIM:-0}"

GEOMETRIC_HPE_PERIODS="${GEOMETRIC_HPE_PERIODS:-24,168}"
GEOMETRIC_HPE_CURV_PHASE_SCALE="${GEOMETRIC_HPE_CURV_PHASE_SCALE:-1.0}"
GEOMETRIC_HPE_CURV_RESIDUAL="${GEOMETRIC_HPE_CURV_RESIDUAL:-0}"
GEOMETRIC_HPE_RES_LAMBDA="${GEOMETRIC_HPE_RES_LAMBDA:-0}"
GEOMETRIC_HPE_LINEAR_K_A_INIT="${GEOMETRIC_HPE_LINEAR_K_A_INIT:-0.01}"
GEOMETRIC_HPE_PE_CURV_WEIGHTED="${GEOMETRIC_HPE_PE_CURV_WEIGHTED:-0}"
GEOMETRIC_HPE_PE_CURV_B_INIT="${GEOMETRIC_HPE_PE_CURV_B_INIT:-0.01}"
GEOMETRIC_HPE_AB_FIXED="${GEOMETRIC_HPE_AB_FIXED:-0}"

OUT_DIR="${OUT_DIR:-./figure_attention_etth1_compare}"
OUT_NAME="${OUT_NAME:-figure_timer_vs_geo_tsfm_etth1.png}"
DPI="${DPI:-200}"

BATCH_INDEX="${BATCH_INDEX:-0}"
BATCH_EFF_INDEX="${BATCH_EFF_INDEX:-0}"
METRICS_MAX_BATCHES="${METRICS_MAX_BATCHES:-0}"

if [ ! -f "$CKPT_TIMER" ] || [ ! -f "$CKPT_GEO" ]; then
  echo "Error: checkpoint file not found (paths relative to repo root)." >&2
  echo "  CKPT_TIMER=$CKPT_TIMER" >&2
  echo "  CKPT_GEO=$CKPT_GEO" >&2
  exit 1
fi

set -e

python figure_timer_geo_attention_etth1.py \
  --ckpt_timer "$CKPT_TIMER" \
  --ckpt_geo "$CKPT_GEO" \
  --root_path "$ROOT" \
  --data_path "$DATA_CSV" \
  --data "$DATA_NAME" \
  --features "$FEATURES" \
  --seq_len "$SEQ_LEN" \
  --patch_len "$PATCH_LEN" \
  --label_len "$LABEL_LEN" \
  --pred_len "$PRED_LEN" \
  --output_len "$OUTPUT_LEN" \
  --d_model "$D_MODEL" \
  --d_ff "$D_FF" \
  --e_layers "$E_LAYERS" \
  --n_heads "$N_HEADS" \
  --factor "$FACTOR" \
  --dropout "$DROPOUT" \
  --activation "$ACTIVATION" \
  --embed "$EMBED" \
  --freq "$FREQ" \
  --stride "$STRIDE" \
  --subset_rand_ratio "$SUBSET_RAND_RATIO" \
  --seed "$SEED" \
  --num_workers "$NUM_WORKERS" \
  --periodic_embedding_branch "$PERIODIC_EMBEDDING_BRANCH" \
  --periodic_emb_bank_dim "$PERIODIC_EMB_BANK_DIM" \
  --geometric_hpe_periods "$GEOMETRIC_HPE_PERIODS" \
  --geometric_hpe_curv_phase_scale "$GEOMETRIC_HPE_CURV_PHASE_SCALE" \
  --geometric_hpe_curv_residual "$GEOMETRIC_HPE_CURV_RESIDUAL" \
  --geometric_hpe_res_lambda "$GEOMETRIC_HPE_RES_LAMBDA" \
  --geometric_hpe_linear_k_a_init "$GEOMETRIC_HPE_LINEAR_K_A_INIT" \
  --geometric_hpe_pe_curv_weighted "$GEOMETRIC_HPE_PE_CURV_WEIGHTED" \
  --geometric_hpe_pe_curv_b_init "$GEOMETRIC_HPE_PE_CURV_B_INIT" \
  --geometric_hpe_ab_fixed "$GEOMETRIC_HPE_AB_FIXED" \
  --batch_index "$BATCH_INDEX" \
  --batch_eff_index "$BATCH_EFF_INDEX" \
  --metrics_max_batches "$METRICS_MAX_BATCHES" \
  --output_dir "$OUT_DIR" \
  --output_name "$OUT_NAME" \
  --dpi "$DPI" \
  --gpu "$GPU"

echo "Done. Figure: $OUT_DIR/$OUT_NAME"
