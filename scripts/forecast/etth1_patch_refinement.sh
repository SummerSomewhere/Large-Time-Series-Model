#!/bin/bash
#
# 两阶段 Patch Refinement 对比实验启动脚本
#
# 实验设计：
#   1. Baseline：标准 Timer 推理（无 Refinement）
#   2. High-MI 实验：从预计算 peaks 文件中取该层 MI > Q3 的 patch，进行替换
#   3. Low-MI 实验：从预计算 peaks 文件中取该层 MI <= Q3 的 patch，进行替换
#
# 核心逻辑：
#   - 高/低 patch 索引来自 etth1_mi_hsic_peaks.py 生成的全局 JSON 文件
#   - 所有样本共用同一套全局 patch 索引，不再运行时计算
#   - topk_patches / bottomk_patches 控制从对应集合中取多少个 patch
#
# 使用说明（三种方式）：
#   # 跑全部：baseline + 高MI + 低MI（默认）
#   bash scripts/forecast/etth1_patch_refinement.sh
#
#   # 仅跑高MI实验
#   bash scripts/forecast/etth1_patch_refinement.sh --skip_baseline --bottomk_patches 0
#
#   # 仅跑低MI实验
#   bash scripts/forecast/etth1_patch_refinement.sh --skip_baseline --topk_patches 0
#

_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$_ROOT" || exit 1

# ── 模型与数据参数 ────────────────────────────────────────────────────────────
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
batch_size=64
subset_rand_ratio=1

# ── 输出目录 ───────────────────────────────────────────────────────────────────
out_dir=./results/refinement_exp

# ── Refinement 核心参数 ────────────────────────────────────────────────────────
# 提取层（负数表示从后数，如 -2 表示倒数第二层）
use_layer_output=-1

# 执行替换的层（在层输入处替换，然后继续往后跑）
# 同层替换：use_layer_output=-1, replace_layer_idx=-1
# 跨层替换：use_layer_output=-2, replace_layer_idx=-1（从倒数第二层提取，在最后一层替换）
replace_layer_idx=-8

# 从高/低 MI patch 集合中各取多少个进行替换
# topk_patches > 0  → 启用高MI实验（从 peaks 文件的 high_mi_patches 取前 topk_patches 个）
# bottomk_patches > 0 → 启用低MI实验（从 peaks 文件的 low_mi_patches 取前 bottomk_patches 个）
topk_patches=2
bottomk_patches=2

# ── 预计算 MI 文件（由 etth1_mi_hsic_peaks.py 生成）───────────────────────────
# peaks 文件包含每层的全局 HSIC 曲线及高/低 MI patch 索引
# 来自全局统计，所有样本共用，不再运行时计算 MI
mi_peaks_file=./results/mi_hsic_etth1/global_mi_peaks_etth1.json

# ── 多卡配置 ─────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nnodes=1 --nproc_per_node=4 experiments/etth1_patch_refinement.py \
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
  --batch_size $batch_size \
  --num_workers $num_workers \
  --output_dir $out_dir \
  --use_layer_output $use_layer_output \
  --replace_layer_idx $replace_layer_idx \
  --topk_patches $topk_patches \
  --bottomk_patches $bottomk_patches \
  --mi_peaks_file $mi_peaks_file \
  "$@"
