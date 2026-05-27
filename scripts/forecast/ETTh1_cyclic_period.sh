#!/bin/sh
#
# ETTh1 finetune (同 ETTh1.sh) + 注意力循环周期惩罚（本脚本内直接写死超参）:
#   attn_logits -= alpha * min(|i-j| mod P, P - (|i-j| mod P))  （patch 下标 i,j）
# P：此处 alpha=0.05, attn_p=0, attn_fft=1 → 每步用 rFFT 主周期映射到 patch 维度的 P；
#   若改固定周期：例如 --cyclic_period_attn_p 7 --cyclic_period_attn_fft 0
#
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=24
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
data=ETTh1

for subset_rand_ratio in 1
do
torchrun --nnodes=1 --nproc_per_node=2 run.py \
  --task_name forecast \
  --is_finetuning 1 \
  --is_training 1 \
  --seed 1 \
  --ckpt_path "$ckpt_path" \
  --root_path ./datasets \
  --data_path $data.csv \
  --data $data \
  --model_id etth1_cycP_$subset_rand_ratio \
  --model $model_name \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --e_layers 8 \
  --factor 3 \
  --des 'CycP' \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 1024 \
  --learning_rate 3e-5 \
  --num_workers 4 \
  --patch_len $patch_len \
  --cyclic_period_attn_alpha 0.05 \
  --cyclic_period_attn_p 0 \
  --cyclic_period_attn_fft 1 \
  --cyclic_period_fft_min_p 2 \
  --cyclic_period_fft_max_p 256 \
  --train_test 0 \
  --subset_rand_ratio $subset_rand_ratio \
  --itr 1 \
  --gpu 0 \
  --use_ims \
  --use_multi_gpu
done
