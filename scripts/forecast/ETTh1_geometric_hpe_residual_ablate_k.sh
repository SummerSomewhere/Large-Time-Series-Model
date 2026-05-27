#!/bin/sh
#
# Ablation (strong): same training setup as ETTh1_geometric_hpe_residual.sh, but embed uses only
# trainable Linear layers (zero-init W,b); ω, φ, curv_to_phase are fixed zero buffers; k and
# res_lambda are fixed 0 in forward (see --geometric_hpe_only_linear_trainable).
#
# --geometric_hpe_pe_proj_ones_input 1 → pe_proj(1) so projection W receives gradients.
#
# Older behavior (separate ablate_k / ablate_omega / ablate_phi flags) is subsumed by
# only_linear_trainable=1 for harmonic/k; keep geometric_hpe_curv_residual=1 so v is not scaled by 1+σ(k).
#
# torchrun --nproc_per_node=2: logical cuda:0,1 on the first two IDs in CUDA_VISIBLE_DEVICES.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path="${CKPT_PATH:-checkpoints/Timer_forecast_1.0.ckpt}"
data="${DATA:-ETTh1}"

for subset_rand_ratio in 1
do
torchrun --nnodes=1 --nproc_per_node=2 run.py \
  --task_name forecast \
  --is_finetuning 1 \
  --is_training 1 \
  --seed 1 \
  --ckpt_path "$ckpt_path" \
  --root_path ./datasets/ \
  --data_path "$data.csv" \
  --data "$data" \
  --model_id "etth1_geohpe_res_k0_w0_phi0_${subset_rand_ratio}" \
  --model "$model_name" \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --e_layers 8 \
  --factor 3 \
  --des 'GeoHPE-Res-k0w0p0' \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 1024 \
  --learning_rate 3e-5 \
  --num_workers 4 \
  --patch_len $patch_len \
  --geometric_hpe 1 \
  --geometric_hpe_periods 24,168 \
  --geometric_hpe_curv_phase_scale 1.0 \
  --geometric_hpe_curv_residual 1 \
  --geometric_hpe_only_linear_trainable 1 \
  --geometric_hpe_pe_proj_ones_input 1 \
  --train_test 0 \
  --subset_rand_ratio $subset_rand_ratio \
  --itr 1 \
  --gpu 0 \
  --use_ims \
  --use_multi_gpu
done
