#!/bin/sh
# ETTh1-style long-horizon forecast: Timer + periodic embedding residual (hour/day) + strict backbone freeze.
# Run from repo root: bash scripts/forecast/ETTh1_periodic_emb.sh
#
# Trains only hour_embed, day_embed, periodic_gamma, and proj; backbone frozen (~67M).
# Loss: (1-α)*MSE + α*MAE(rFFT) via --loss_fft_alpha.
#
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4}"

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
data=ETTh1

NPROC="${NPROC:-2}"
if [ "$NPROC" -lt 1 ]; then NPROC=1; fi

LOSS_FFT_ALPHA="${LOSS_FFT_ALPHA:-0.5}"

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
    --model_id "etth1_periodic_sr_${subset_rand_ratio}" \
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
    --learning_rate 1e-4 \
    --num_workers 4 \
    --patch_len "$patch_len" \
    --train_test 0 \
    --subset_rand_ratio "$subset_rand_ratio" \
    --itr 1 \
    --gpu 0 \
    --use_ims \
    --use_multi_gpu \
    --periodic_embedding_branch 1 \
    --finetune_trainable periodic_emb_proj \
    --loss_fft_alpha "$LOSS_FFT_ALPHA"
done
