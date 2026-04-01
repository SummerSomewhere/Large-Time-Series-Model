#!/bin/sh
#
# ETTh1 finetune with Geometric-HPE (curvature-scaled patch embed + harmonic phase encoding).
# Physical motivation: treat series as a geometric dynamical system — curvature highlights
# turning points; learnable harmonics (e.g. daily/weekly) with curvature-guided phase
# lock waveform structure without extra statistics.
#
# Requires: Timer backbone (--model Timer). Embedding replaces absolute sinusoidal PE when
# --geometric_hpe 1 (see layers/Embed.py GeometricHPEPatchEmbedding).

export CUDA_VISIBLE_DEVICES=3,4

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
data=ETTh2

for subset_rand_ratio in 1
do
torchrun --nnodes=1 --nproc_per_node=2 run.py \
  --task_name forecast \
  --is_finetuning 1 \
  --is_training 1 \
  --seed 1 \
  --ckpt_path "$ckpt_path" \
  --root_path ./datasets/ETT-small/ \
  --data_path $data.csv \
  --data $data \
  --model_id etth1_geohpe_${subset_rand_ratio} \
  --model $model_name \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --e_layers 8 \
  --factor 3 \
  --des 'GeoHPE' \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 1024 \
  --learning_rate 3e-5 \
  --num_workers 4 \
  --patch_len $patch_len \
  --geometric_hpe 1 \
  --geometric_hpe_periods 24,168 \
  --geometric_hpe_curv_phase_scale 1.0 \
  --train_test 0 \
  --subset_rand_ratio $subset_rand_ratio \
  --itr 1 \
  --gpu 0 \
  --use_ims \
  --use_multi_gpu
done
