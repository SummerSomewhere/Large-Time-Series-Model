#!/bin/sh
#
# Native Timer (default PatchEmbedding + sinusoidal PE, --geometric_hpe 0): no curvature scaling.
# Default: finetune + test (run.py) then L×H encoder self-attention heatmaps.
#
# Only figure: RUN_FINETUNE=0 CKPT_PATH=.../checkpoint.pth bash ...
#
# Optional: USE_TORCHRUN=0 — single-GPU python. Default 1: torchrun DDP on NPROC_PER_NODE (default 3) GPUs.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
USE_TORCHRUN="${USE_TORCHRUN:-1}"

cd "$(dirname "$0")/../.." || exit 1

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
pretrain_ckpt="${PRETRAIN_CKPT:-checkpoints/Timer_forecast_1.0.ckpt}"
data="${DATA_NAME:-weather}"
root_path="${ROOT:-./datasets}"
data_csv="${DATA_CSV:-${data}.csv}"

RUN_FINETUNE="${RUN_FINETUNE:-1}"
CKPT_PATH="${CKPT_PATH:-$pretrain_ckpt}"
SEED="${SEED:-1}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PATIENCE="${PATIENCE:-15}"
GPU="${GPU:-0}"

OUT_DIR="${OUT_DIR:-./figure_attention_timer_layers_heads}"
OUT_NAME="${OUT_NAME:-attention_layers_heads_native.png}"
BATCH_INDEX="${BATCH_INDEX:-0}"
BATCH_EFF_INDEX="${BATCH_EFF_INDEX:-0}"
METRICS_MAX_BATCHES="${METRICS_MAX_BATCHES:-0}"

model_id="${MODEL_ID:-etth1_timer_native_attn}"

set -e

if [ "$RUN_FINETUNE" = "1" ]; then
  if [ ! -f "$pretrain_ckpt" ]; then
    echo "Missing pretrained checkpoint: $pretrain_ckpt" >&2
    exit 1
  fi
  echo "========== [1/2] Finetune then test (native Timer, geometric_hpe=0) =========="
  common_ft="--task_name forecast \
    --is_finetuning 1 \
    --is_training 1 \
    --seed $SEED \
    --ckpt_path $pretrain_ckpt \
    --root_path $root_path \
    --data_path $data_csv \
    --data $data \
    --model_id $model_id \
    --model $model_name \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --output_len $output_len \
    --e_layers 8 \
    --factor 3 \
    --des NativeAttn \
    --d_model 1024 \
    --d_ff 2048 \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_workers $NUM_WORKERS \
    --patch_len $patch_len \
    --finetune_epochs $FINETUNE_EPOCHS \
    --patience $PATIENCE \
    --train_test 0 \
    --subset_rand_ratio 1 \
    --itr 1 \
    --gpu $GPU \
    --use_ims \
    --geometric_hpe 0"

  if [ "$USE_TORCHRUN" = "1" ]; then
    # shellcheck disable=SC2086
    torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" run.py $common_ft --use_multi_gpu
  else
    # shellcheck disable=SC2086
    python run.py $common_ft
  fi

  d=$(ls -td checkpoints/forecast_"${model_id}"_Timer_"${data}"_* 2>/dev/null | head -1)
  if [ -n "$d" ] && [ -f "$d/checkpoint.pth" ]; then
    CKPT_PATH="$d/checkpoint.pth"
  fi
fi

if [ ! -f "$CKPT_PATH" ]; then
  echo "Checkpoint not found: $CKPT_PATH" >&2
  echo "Set CKPT_PATH or run with RUN_FINETUNE=1 (needs $pretrain_ckpt)." >&2
  exit 1
fi

if [ "$RUN_FINETUNE" = "1" ]; then
  echo "========== [2/2] Attention figure (L×H heatmaps, native Timer) =========="
else
  echo "========== Attention figure only (RUN_FINETUNE=0) =========="
fi
echo "CKPT_PATH=$CKPT_PATH"

python figure_timer_attention_layers_heads.py \
  --ckpt_path "$CKPT_PATH" \
  --root_path "$root_path" \
  --data_path "$data_csv" \
  --data "$data" \
  --seq_len "$seq_len" \
  --label_len "$label_len" \
  --pred_len "$pred_len" \
  --output_len "$output_len" \
  --patch_len "$patch_len" \
  --e_layers 8 \
  --n_heads 8 \
  --factor 3 \
  --d_model 1024 \
  --d_ff 2048 \
  --subset_rand_ratio 1 \
  --seed "$SEED" \
  --num_workers "$NUM_WORKERS" \
  --batch_index "$BATCH_INDEX" \
  --batch_eff_index "$BATCH_EFF_INDEX" \
  --metrics_max_batches "$METRICS_MAX_BATCHES" \
  --output_dir "$OUT_DIR" \
  --output_name "$OUT_NAME" \
  --gpu "$GPU"

echo "Done. Figure: $OUT_DIR/$OUT_NAME"
