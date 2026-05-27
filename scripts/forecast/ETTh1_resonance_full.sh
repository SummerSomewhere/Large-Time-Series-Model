#!/bin/sh
# Full finetune (all parameters) with periodic embedding residual enabled — for ablation vs periodic_emb_proj-only.
# Repo root: bash scripts/forecast/ETTh1_resonance_full.sh
#
set -e

export CUDA_VISIBLE_DEVICES=3,4
NPROC=2

model_name=Timer
data=ETTh2
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt

seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96

LEARNING_RATE=3e-5
LOSS_FFT_ALPHA=0.5
MODEL_ID_PREFIX=etth1_periodic_full

FINETUNE_EPOCHS=

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
    --seed 1 \
    --ckpt_path "$ckpt_path" \
    --root_path ./datasets/ETT-small/ \
    --data_path "${data}.csv" \
    --data "$data" \
    --freq h \
    --model_id "${MODEL_ID_PREFIX}_sr_${subset_rand_ratio}" \
    --model "$model_name" \
    --features M \
    --seq_len "$seq_len" \
    --label_len "$label_len" \
    --pred_len "$pred_len" \
    --output_len "$output_len" \
    --e_layers 8 \
    --factor 3 \
    --des 'Exp' \
    --d_model 1024 \
    --d_ff 2048 \
    --batch_size 1024 \
    --learning_rate "$LEARNING_RATE" \
    --num_workers 4 \
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
