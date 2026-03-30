#!/bin/sh
# Finetune from a Timer checkpoint on another dataset (full trainable weights + optional periodic branch).
# Repo root: bash scripts/forecast/resonance_ckpt_full_finetune.sh
#
set -e

CKPT_PATH=checkpoints/Timer_forecast_1.0.ckpt
DATA=ETTh1
ROOT_PATH=./datasets/ETT-small/
DATA_PATH="${DATA}.csv"
FREQ=h

export CUDA_VISIBLE_DEVICES=3,4
NPROC=2
SEED=1
NUM_WORKERS=4
BATCH_SIZE=1024

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
e_layers=8
factor=3
d_model=1024
d_ff=2048

LEARNING_RATE=3e-5
LOSS_FFT_ALPHA=0.0
MODEL_ID_PREFIX=periodic_full_ft

FINETUNE_EPOCHS=

if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
  echo "ERROR: set CKPT_PATH to an existing .pth/.ckpt" >&2
  exit 1
fi

if [ "$NPROC" -lt 1 ]; then NPROC=1; fi

_extra_epochs=
if [ -n "$FINETUNE_EPOCHS" ]; then
  _extra_epochs="--finetune_epochs $FINETUNE_EPOCHS"
fi

for subset_rand_ratio in 1
do
  torchrun --nnodes=1 --nproc_per_node="$NPROC" run.py \
    --task_name forecast \
    --is_finetuning 1 \
    --is_training 1 \
    --seed "$SEED" \
    --ckpt_path "$CKPT_PATH" \
    --root_path "$ROOT_PATH" \
    --data_path "$DATA_PATH" \
    --data "$DATA" \
    --freq "$FREQ" \
    --model_id "${MODEL_ID_PREFIX}_${DATA}_sr_${subset_rand_ratio}" \
    --model "$model_name" \
    --features M \
    --seq_len "$seq_len" \
    --label_len "$label_len" \
    --pred_len "$pred_len" \
    --output_len "$output_len" \
    --e_layers "$e_layers" \
    --factor "$factor" \
    --des 'Exp' \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_workers "$NUM_WORKERS" \
    --patch_len "$patch_len" \
    --train_test 0 \
    --subset_rand_ratio "$subset_rand_ratio" \
    --itr 1 \
    --gpu 0 \
    --use_ims \
    --use_multi_gpu \
    --periodic_embedding_branch 1 \
    --finetune_trainable full \
    --loss_fft_alpha "$LOSS_FFT_ALPHA" \
    $_extra_epochs
done
