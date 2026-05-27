#!/bin/bash
#
# 语义锚点注入对比实验脚本 (ETTh1)
#
# 功能：
#   1. Baseline — 标准 Timer 推理
#   2. Guide-Injection — 两遍推理
#      第一遍：收集最后一层的最后一个 token 作为语义锚点
#      第二遍：将锚点注入 inject_layers 指定的层（每个样本独立）
#
# 使用方式：
#   bash scripts/forecast/etth1_layer_guide_injection.sh
#
# 自定义注入层：
#   inject_layers="1 2 3" bash scripts/forecast/etth1_layer_guide_injection.sh
#
# 多卡运行（指定 GPU）：
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/forecast/etth1_layer_guide_injection.sh
#   或修改脚本顶部的 CUDA_VISIBLE_DEVICES 变量
#

set -e

_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$_ROOT" || exit 1

# ─────────────────────────────────────────────────────────────────────────────
# 实验参数配置
# ─────────────────────────────────────────────────────────────────────────────

# GPU 配置（修改这里指定使用的 GPU）
export CUDA_VISIBLE_DEVICES=4,5,6
N_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)  # 自动计算 GPU 数量

# 模型参数（与 ETTh1.sh 保持一致）
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

# ── 引导 Token 注入参数 ───────────────────────────────────────────────────────
# 机制：inject_layers 中的层注入 anchor_layer 指定的层的 MeanPool 作为语义锚点
# 每个样本的锚点独立，不跨样本共享
use_layer_guide=true
anchor_layer=7   # 从第几层提取锚点（0-indexed，-1=最后一层）
inject_layers="0"   # 注入层列表（0-indexed），默认注入全部8层
guide_mode=last_token
truncate_guide=0   # 0=不截断（默认，guide token 持续存在）；1=截断（对齐 MOMENT，每层注入每层消耗）
batch_size=2048

# ── 输出配置 ──────────────────────────────────────────────────────────────────
output_dir=./results/layer_guide_injection

# ─────────────────────────────────────────────────────────────────────────────
mkdir -p "$output_dir"

# ─────────────────────────────────────────────────────────────────────────────
# 运行实验
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "逐层引导 Token 注入对比实验"
echo "============================================================"
echo "GPU           : $CUDA_VISIBLE_DEVICES"
echo "n_gpu         : $N_GPU"
echo "anchor_layer  : $anchor_layer"
echo "inject_layers : $inject_layers"
echo "output_dir    : $output_dir"
echo "============================================================"
echo ""

BASE_ARGS=(
    --ckpt_path "$ckpt_path"
    --root_path "$root_path"
    --data_path ${data}.csv
    --data "$data"
    --features "$features"
    --seq_len "$seq_len"
    --label_len "$label_len"
    --pred_len "$pred_len"
    --output_len "$output_len"
    --patch_len "$patch_len"
    --e_layers "$e_layers"
    --factor "$factor"
    --d_model "$d_model"
    --d_ff "$d_ff"
    --n_heads "$n_heads"
    --batch_size "$batch_size"
    --num_workers "$num_workers"
    --output_dir "$output_dir"
    --guide_mode "$guide_mode"
    --anchor_layer "$anchor_layer"
    --truncate_guide "$truncate_guide"
)

if [ "$use_layer_guide" = true ]; then
    BASE_ARGS+=(--use_layer_guide)
    # inject_layers 为逗号分隔的字符串，转换为空格分隔的独立参数
    IFS=',' read -ra LAYERS <<< "$inject_layers"
    BASE_ARGS+=(--inject_layers "${LAYERS[@]}")
fi

if [ "${plot_samples:-false}" = true ]; then
    BASE_ARGS+=(--plot_samples)
fi

if [ "$N_GPU" -gt 1 ]; then
    echo "[INFO] 使用多卡模式 (nproc_per_node=$N_GPU)"
    torchrun --nnodes=1 --nproc_per_node=$N_GPU \
        experiments/etth1_layer_guide_injection.py \
        "${BASE_ARGS[@]}"
else
    echo "[INFO] 使用单卡模式"
    python experiments/etth1_layer_guide_injection.py \
        "${BASE_ARGS[@]}"
fi

echo ""
echo "============================================================"
echo "逐层引导 Token 注入实验完成"
echo "============================================================"
echo "实验结果: $output_dir"
echo "============================================================"
