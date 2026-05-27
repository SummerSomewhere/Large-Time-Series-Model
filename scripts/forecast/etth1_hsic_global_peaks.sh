#!/bin/sh
#
# HSIC Global Peaks analysis (Timer + ETTh1)。
# 输出: global_mi_peaks_{model_id}.json，带 hsic_curve1 字段。
#

_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$_ROOT" || exit 1

# 四卡：物理 GPU 0–4（进程内为 cuda:0–4）。改卡号只改本行即可。
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# ----- 与 ETTh1.sh 相同 -----
model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
data=ETTm2
root_path=./datasets/
num_workers=6
e_layers=8
factor=3
d_model=1024
d_ff=2048
n_heads=8
features=M
batch_size_mi=1024
# --------------------------

out_dir=./results/mi_hsic_etth1_global
seed=42

# 用于命名输出文件的模型标识符（会生成 global_mi_peaks_{model_id}.json）
model_id=etth1
torchrun --nnodes=1 --nproc_per_node=5 experiments/etth1_hsic_global_peaks.py \
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
  --out_dir $out_dir \
  --model_id $model_id \
  --seed $seed
