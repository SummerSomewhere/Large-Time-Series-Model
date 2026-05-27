#!/bin/sh
#
# ETTh1 finetune with Geometric-HPE (curvature-scaled patch embed + harmonic phase encoding).
# Physical motivation: treat series as a geometric dynamical system — curvature highlights
# turning points; learnable harmonics (e.g. daily/weekly) with curvature-guided phase
# lock waveform structure without extra statistics.
#
# Requires: Timer backbone (--model Timer). Embedding replaces absolute sinusoidal PE when
# --geometric_hpe 1 (see layers/Embed.py GeometricHPEPatchEmbedding).
#
# Patch: v' = v * (1 + a*k_norm), a learnable (linear_k_patch_scale=1, ab_fixed=0).
# PE:   pe' = pe * (1 + b*k_norm), b learnable (geometric_hpe_pe_curv_weighted=1, ab_fixed=0).
# k_norm = k / max_patch(k) per row; k = mean|Δ²x| on raw patch values only (Embed.py) — real curvature, not a loss term.
# --geometric_hpe_curv_residual 0 and --geometric_hpe_res_lambda 0: no residual λ path (other scripts default).
#
# Grid search over (a_init, b_init) and log MSE: scripts/forecast/ETTh1_geometric_hpe_grid_ab.sh
#
# Finetune loss: MSE + λ × normalized curvature error MSE(Δ²) (per-batch scale; 非权重正则). CURVATURE_* overrides below.

# Two processes (nproc_per_node=2): logical cuda:0,1 -> first two IDs here (override: CUDA_VISIBLE_DEVICES=... bash ...)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

CURVATURE_REG_ENABLE="${CURVATURE_REG_ENABLE:-1}"
# Learnable λ (default): init only; fixed λ used only if CURVATURE_LAMBDA_LEARNABLE=0.
CURVATURE_LAMBDA_LEARNABLE="${CURVATURE_LAMBDA_LEARNABLE:-1}"
CURVATURE_REG_LAMBDA_INIT="${CURVATURE_REG_LAMBDA_INIT:-11.8}"
CURVATURE_REG_LAMBDA="${CURVATURE_REG_LAMBDA:-1.0}"
CURVATURE_REG_SMOOTH="${CURVATURE_REG_SMOOTH:-0}"
model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
data=ETTh1

# Embed forward: pe' = pe * (1 + b*k_norm) requires pe_curv_weighted=1; b trainable requires ab_fixed=0 (see below).

for subset_rand_ratio in 1
do
torchrun --nnodes=1 --nproc_per_node=2 run.py \
  --task_name forecast \
  --is_finetuning 1 \
  --is_training 1 \
  --seed 1 \
  --ckpt_path "$ckpt_path" \
  --root_path ./datasets/ \
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
  --geometric_hpe_curv_residual 0 \
  --geometric_hpe_res_lambda 0 \
  --geometric_hpe_linear_k_patch_scale 1 \
  --geometric_hpe_linear_k_a_init 0.1 \
  --geometric_hpe_pe_curv_weighted 1 \
  --geometric_hpe_pe_curv_b_init 0.1 \
  --geometric_hpe_ab_fixed 0 \
  --curvature_reg_enable "$CURVATURE_REG_ENABLE" \
  --curvature_lambda_learnable "$CURVATURE_LAMBDA_LEARNABLE" \
  --curvature_reg_lambda_init "$CURVATURE_REG_LAMBDA_INIT" \
  --curvature_reg_lambda "$CURVATURE_REG_LAMBDA" \
  --curvature_reg_smooth "$CURVATURE_REG_SMOOTH" \
  --train_test 0 \
  --subset_rand_ratio $subset_rand_ratio \
  --itr 1 \
  --gpu 0 \
  --use_ims \
  --use_multi_gpu
done
