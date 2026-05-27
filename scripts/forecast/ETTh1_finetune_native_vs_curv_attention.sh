#!/bin/sh
#
# Mimics scripts/forecast/ETTh1.sh: torchrun finetune + test, then one merged attention figure.
#
# (1) Native Timer (--geometric_hpe 0): finetune → test (run.py).
# (2) Geometric-HPE + v'=v(1+a*k_norm), k = mean |Δ²x| per patch (same as ETTh1_geometric_hpe.sh /
#     layers/Embed.py). a learnable, init 0 (--geometric_hpe_linear_k_a_init 0, --geometric_hpe_ab_fixed 0).
#     PE term PE*(1+b*k_norm) when --geometric_hpe_pe_curv_weighted 1 (default here, same as ETTh1_geometric_hpe.sh).
#     Each finetune epoch prints scalar a via exp_forecast._log_patch_curv_scale_a_epoch.
# (3) figure_timer_native_vs_curv_attention_merged.py: one PNG with two L×H grids + patch k footer.
#
# Usage:
#   bash ./scripts/forecast/ETTh1_finetune_native_vs_curv_attention.sh
#
# Optional:
#   SKIP_FINETUNE=1 — only step (3); set CKPT_NATIVE / CKPT_GEO or rely on latest checkpoints dirs.
#   DATA_NAME / ROOT / PRETRAIN_CKPT / USE_TORCHRUN / NPROC_PER_NODE — override defaults.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
USE_TORCHRUN="${USE_TORCHRUN:-1}"

cd "$(dirname "$0")/../.." || exit 1

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
pretrain_ckpt="${PRETRAIN_CKPT:-checkpoints/Timer_forecast_1.0.ckpt}"
data="${DATA_NAME:-ETTh1}"
root_path="${ROOT:-./datasets}"
data_csv="${DATA_CSV:-${data}.csv}"

SEED="${SEED:-1}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PATIENCE="${PATIENCE:-15}"
GPU="${GPU:-0}"
FEATURES="${FEATURES:-M}"
DROPOUT="${DROPOUT:-0.1}"
ACTIVATION="${ACTIVATION:-gelu}"
EMBED="${EMBED:-timeF}"
FREQ="${FREQ:-h}"
STRIDE="${STRIDE:-1}"
SUBSET_RAND_RATIO="${SUBSET_RAND_RATIO:-1}"
E_LAYERS="${E_LAYERS:-8}"
N_HEADS="${N_HEADS:-8}"
FACTOR="${FACTOR:-3}"
D_MODEL="${D_MODEL:-1024}"
D_FF="${D_FF:-2048}"

MODEL_ID_NATIVE="${MODEL_ID_NATIVE:-etth1_timer_native_merged}"
MODEL_ID_GEO="${MODEL_ID_GEO:-etth1_timer_geo_curv_merged}"

# Geo-HPE: match ETTh1_geometric_hpe.sh for v/PE scaling; a,b learnable (ab_fixed=0), a_init=0.
GEOMETRIC_HPE_PERIODS="${GEOMETRIC_HPE_PERIODS:-24,168}"
GEOMETRIC_HPE_PE_CURV_WEIGHTED="${GEOMETRIC_HPE_PE_CURV_WEIGHTED:-1}"
GEOMETRIC_HPE_PE_CURV_B_INIT="${GEOMETRIC_HPE_PE_CURV_B_INIT:-0.01}"

OUT_DIR="${OUT_DIR:-./figure_attention_native_vs_curv_merged}"
OUT_NAME="${OUT_NAME:-attention_native_vs_curv_merged.png}"
BATCH_INDEX="${BATCH_INDEX:-0}"
BATCH_EFF_INDEX="${BATCH_EFF_INDEX:-0}"
METRICS_MAX_BATCHES="${METRICS_MAX_BATCHES:-0}"
SKIP_FINETUNE="${SKIP_FINETUNE:-0}"

latest_ckpt_pth() {
  mid="$1"
  d=$(ls -td checkpoints/forecast_"${mid}"_Timer_"${data}"_* 2>/dev/null | head -1)
  if [ -z "$d" ] || [ ! -f "$d/checkpoint.pth" ]; then
    echo ""
    return 1
  fi
  echo "$d/checkpoint.pth"
}

run_train() {
  mid="$1"
  des="$2"
  shift 2

  if [ ! -f "$pretrain_ckpt" ]; then
    echo "Missing pretrained checkpoint: $pretrain_ckpt" >&2
    exit 1
  fi

  common="--task_name forecast \
    --is_finetuning 1 \
    --is_training 1 \
    --seed $SEED \
    --ckpt_path $pretrain_ckpt \
    --root_path $root_path \
    --data_path $data_csv \
    --data $data \
    --model_id $mid \
    --model $model_name \
    --features $FEATURES \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --output_len $output_len \
    --e_layers $E_LAYERS \
    --n_heads $N_HEADS \
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
    --patch_len $patch_len \
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
  echo "========== [1/3] Native Timer finetune + test =========="
  run_train "$MODEL_ID_NATIVE" "ExpNative" --geometric_hpe 0

  echo "========== [2/3] Geo-HPE Timer (v'=v(1+a*k_norm), a init 0, learnable) finetune + test =========="
  run_train "$MODEL_ID_GEO" "ExpGeoCurv" \
    --geometric_hpe 1 \
    --geometric_hpe_periods "$GEOMETRIC_HPE_PERIODS" \
    --geometric_hpe_curv_phase_scale 1.0 \
    --geometric_hpe_curv_residual 0 \
    --geometric_hpe_res_lambda 0 \
    --geometric_hpe_linear_k_patch_scale 1 \
    --geometric_hpe_linear_k_a_init 0 \
    --geometric_hpe_pe_curv_weighted "$GEOMETRIC_HPE_PE_CURV_WEIGHTED" \
    --geometric_hpe_pe_curv_b_init "$GEOMETRIC_HPE_PE_CURV_B_INIT" \
    --geometric_hpe_ab_fixed 0
fi

CKPT_NATIVE="${CKPT_NATIVE:-$(latest_ckpt_pth "$MODEL_ID_NATIVE")}"
CKPT_GEO="${CKPT_GEO:-$(latest_ckpt_pth "$MODEL_ID_GEO")}"

if [ -z "$CKPT_NATIVE" ] || [ ! -f "$CKPT_NATIVE" ]; then
  echo "Could not resolve CKPT_NATIVE. Finetune first or: export CKPT_NATIVE=.../checkpoint.pth" >&2
  exit 1
fi
if [ -z "$CKPT_GEO" ] || [ ! -f "$CKPT_GEO" ]; then
  echo "Could not resolve CKPT_GEO. Finetune first or: export CKPT_GEO=.../checkpoint.pth" >&2
  exit 1
fi

echo "========== [3/3] Merged L×H attention figure + patch k =========="
echo "CKPT_NATIVE=$CKPT_NATIVE"
echo "CKPT_GEO=$CKPT_GEO"

SKIP_IMS_IN_FIGURE="${SKIP_IMS_IN_FIGURE:-0}"
EXTRA_FIG=""
if [ "$SKIP_IMS_IN_FIGURE" = "1" ]; then
  EXTRA_FIG="--skip_ims_metrics"
fi

# shellcheck disable=SC2086
python figure_timer_native_vs_curv_attention_merged.py \
  --ckpt_native "$CKPT_NATIVE" \
  --ckpt_geo "$CKPT_GEO" \
  --root_path "$root_path" \
  --data_path "$data_csv" \
  --data "$data" \
  --seq_len "$seq_len" \
  --label_len "$label_len" \
  --pred_len "$pred_len" \
  --output_len "$output_len" \
  --patch_len "$patch_len" \
  --e_layers "$E_LAYERS" \
  --n_heads "$N_HEADS" \
  --factor "$FACTOR" \
  --d_model "$D_MODEL" \
  --d_ff "$D_FF" \
  --subset_rand_ratio "$SUBSET_RAND_RATIO" \
  --seed "$SEED" \
  --num_workers "$NUM_WORKERS" \
  --geometric_hpe_pe_curv_weighted "$GEOMETRIC_HPE_PE_CURV_WEIGHTED" \
  --geometric_hpe_pe_curv_b_init "$GEOMETRIC_HPE_PE_CURV_B_INIT" \
  --geometric_hpe_ab_fixed 0 \
  --geometric_hpe_linear_k_a_init 0 \
  --batch_index "$BATCH_INDEX" \
  --batch_eff_index "$BATCH_EFF_INDEX" \
  --metrics_max_batches "$METRICS_MAX_BATCHES" \
  --output_dir "$OUT_DIR" \
  --output_name "$OUT_NAME" \
  --gpu "$GPU" \
  $EXTRA_FIG

echo "Done. Figure: $OUT_DIR/$OUT_NAME"
