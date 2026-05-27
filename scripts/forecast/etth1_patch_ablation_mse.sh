#!/bin/sh
#
# Patch 消融：高 HSIC vs 随机 patch 置零，对比 IMS 预测窗 MSE/MAE。
# 四卡：CUDA_VISIBLE_DEVICES=4,5,6,7 + torchrun --nproc_per_node=4 + --use_multi_gpu（测试集按卡切分，指标 all_reduce）。
#

_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$_ROOT" || exit 1

export CUDA_VISIBLE_DEVICES=4,5,6,7

# ----- 与 ETTh1.sh 相同 -----
model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
data=ETTh1
root_path=./datasets/
num_workers=4
e_layers=8
factor=3
d_model=1024
d_ff=2048
n_heads=8
features=M
batch_size_mi=64
# --------------------------

subset_rand_ratio=1
ablate_k=4
ablate_mode=zero
seed=42

torchrun --nnodes=1 --nproc_per_node=4 experiments/etth1_patch_ablation_mse.py \
  --ckpt_path $ckpt_path \
  --root_path $root_path \
  --data_path ${data}.csv \
  --data $data \
  --features $features \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --patch_len $patch_len \
  --e_layers $e_layers \
  --factor $factor \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --use_ims \
  --use_multi_gpu \
  --batch_size $batch_size_mi \
  --num_workers $num_workers \
  --subset_rand_ratio $subset_rand_ratio \
  --ablate_k $ablate_k \
  --ablate_mode $ablate_mode \
  --seed $seed
