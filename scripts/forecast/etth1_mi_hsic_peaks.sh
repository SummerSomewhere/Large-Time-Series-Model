#!/bin/sh
#
# MI / HSIC（Timer + ETTh1）。脚本内固定：CUDA_VISIBLE_DEVICES=4,5,6,7 + torchrun --nproc_per_node=4。
# 必须用 torchrun 启动（会设置 WORLD_SIZE），测试集才会按卡切分；不要用 4 个终端各跑 python。
# **输出直接写在 out_dir 下**，不追加 rank_* 子目录。
#
# 输出：
#   random_samples/        — 随机采样的逐样本逐层 HSIC 曲线图
#   mi_mean_over_batches/ — 全局平均 HSIC 曲线（decoder 前 8 个 attn block，layer_00…07）
#

_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$_ROOT" || exit 1

# 四卡：物理 GPU 4–7（进程内为 cuda:0–3）。改卡号只改本行即可。
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

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
num_workers=6
e_layers=8
factor=3
d_model=1024
d_ff=2048
n_heads=8
features=M
batch_size_mi=1024
# --------------------------

# y_decoder_mode: decoder (default) | embedding | raw
#   decoder    — h_y goes through full decoder layers (current baseline)
#   embedding  — h_y only through patch embedding, no decoder
#   raw        — h_y is raw future window mean, no model forward
y_decoder_mode=decoder

subset_rand_ratio=1
out_dir=./results/mi_hsic_etth1_${y_decoder_mode}
num_random_plot_samples=20
plot_seed=42
max_samples=0
bw_max_samples=2000
seed=42

# 用于命名输出文件的模型标识符（会生成 global_mi_peaks_{model_id}.json）
model_id=etth1
torchrun --nnodes=1 --nproc_per_node=5 experiments/etth1_mi_hsic_peaks.py \
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
  --num_random_plot_samples $num_random_plot_samples \
  --plot_seed $plot_seed \
  --subset_rand_ratio $subset_rand_ratio \
  --model_id $model_id \
  --max_samples $max_samples \
  --bw_max_samples $bw_max_samples \
  --seed $seed \
  --y_decoder_mode $y_decoder_mode

