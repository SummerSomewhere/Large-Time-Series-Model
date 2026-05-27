#!/bin/bash
#
# Prototype Prompting experiment script (ETTh1)
#
# 功能：
#   1. 原型提取——指定层 HSIC 最高的 top-k patch 平均（High-MI 原型）
#   2. 原型提取——指定层 HSIC 最低的 top-k patch 平均（Low-MI 原型）
#   3. 原型注入推理对比（Baseline / High-MI Prototype / Low-MI Prototype）
#
# 使用方式：
#   bash scripts/forecast/etth1_prototype_injection.sh
#
# 多卡运行（必须用 torchrun）：
#   torchrun --nnodes=1 --nproc_per_node=4 scripts/forecast/etth1_prototype_injection.sh
#
# 可选参数（在脚本顶部修改 target_layer 变量）：
#   target_layer=-1  (默认: 最后一层)
#   target_layer=0   (第一层)
#   target_layer=3   (第四层，以此类推)
#

set -e

_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$_ROOT" || exit 1

# ─────────────────────────────────────────────────────────────────────────────
# 实验参数配置
# ─────────────────────────────────────────────────────────────────────────────

# GPU 配置
export CUDA_VISIBLE_DEVICES=4,5,6,7

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

# 原型提取参数
top_k=1                          # 选取 HSIC 最高/最低的 top-k patch
target_layer=-1                  # 目标层（-1=最后一层，0=第一层，以此类推）
proto_extract_out=./results/prototype_extraction
batch_size_proto=64

# 原型注入实验参数
proto_injection_out=./results/prototype_injection_exp
batch_size_exp=2048

# ─────────────────────────────────────────────────────────────────────────────
# 公共变量
# ─────────────────────────────────────────────────────────────────────────────

if [ "$target_layer" -eq -1 ]; then
    last_layer=$((e_layers - 1))
else
    last_layer=$target_layer
fi
proto_file="$proto_extract_out/h_proto_high_top${top_k}_layer${last_layer}.pt"
low_proto_file="$proto_extract_out/h_proto_low_top${top_k}_layer${last_layer}.pt"

mkdir -p "$proto_extract_out"

# ─────────────────────────────────────────────────────────────────────────────
# 阶段 1：High-MI 原型提取（HSIC 最高的 top-k patch 平均）
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "阶段 1: High-MI 原型提取"
echo "============================================================"
echo "Target layer: $target_layer (-1=last layer, resolved to layer $last_layer)"
echo "Top-K patches: $top_k"
echo "输出文件: $proto_file"
echo "============================================================"
echo ""

if [ -f "$proto_file" ]; then
    echo "[INFO] High-MI 原型文件已存在: $proto_file"
    echo "[INFO] 跳过 High-MI 原型提取步骤，如需重新提取请删除该文件"
else
    echo "[INFO] 运行 High-MI 原型提取脚本..."

    if command -v torchrun &> /dev/null && [ -n "$WORLD_SIZE" ]; then
        echo "[INFO] 检测到 torchrun，使用多卡模式"
        torchrun --nnodes=1 --nproc_per_node=4 experiments/prototype_extraction.py \
            --ckpt_path "$ckpt_path" \
            --root_path "$root_path" \
            --data_path ${data}.csv \
            --data "$data" \
            --features "$features" \
            --seq_len "$seq_len" \
            --label_len "$label_len" \
            --pred_len "$pred_len" \
            --output_len "$output_len" \
            --patch_len "$patch_len" \
            --e_layers "$e_layers" \
            --factor "$factor" \
            --d_model "$d_model" \
            --d_ff "$d_ff" \
            --n_heads "$n_heads" \
            --batch_size "$batch_size_proto" \
            --num_workers "$num_workers" \
            --out_dir "$proto_extract_out" \
            --top_k "$top_k" \
            --target_layer "$target_layer" \
            --mi_mode high \
            --use_multi_gpu
    else
        echo "[INFO] 使用单卡模式"
        python experiments/prototype_extraction.py \
            --ckpt_path "$ckpt_path" \
            --root_path "$root_path" \
            --data_path ${data}.csv \
            --data "$data" \
            --features "$features" \
            --seq_len "$seq_len" \
            --label_len "$label_len" \
            --pred_len "$pred_len" \
            --output_len "$output_len" \
            --patch_len "$patch_len" \
            --e_layers "$e_layers" \
            --factor "$factor" \
            --d_model "$d_model" \
            --d_ff "$d_ff" \
            --n_heads "$n_heads" \
            --batch_size "$batch_size_proto" \
            --num_workers "$num_workers" \
            --out_dir "$proto_extract_out" \
            --top_k "$top_k" \
            --target_layer "$target_layer" \
            --mi_mode high
    fi

    echo "[INFO] High-MI 原型提取完成"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 阶段 2：Low-MI 原型提取（HSIC 最低的 top-k patch 平均）
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "阶段 2: Low-MI 原型提取"
echo "============================================================"
echo "Target layer: $target_layer (-1=last layer, resolved to layer $last_layer)"
echo "Top-K patches: $top_k"
echo "输出文件: $low_proto_file"
echo "============================================================"
echo ""

if [ -f "$low_proto_file" ]; then
    echo "[INFO] Low-MI 原型文件已存在: $low_proto_file"
    echo "[INFO] 跳过 Low-MI 原型提取步骤，如需重新提取请删除该文件"
else
    echo "[INFO] 运行 Low-MI 原型提取脚本..."

    if command -v torchrun &> /dev/null && [ -n "$WORLD_SIZE" ]; then
        torchrun --nnodes=1 --nproc_per_node=4 experiments/prototype_extraction.py \
            --ckpt_path "$ckpt_path" \
            --root_path "$root_path" \
            --data_path ${data}.csv \
            --data "$data" \
            --features "$features" \
            --seq_len "$seq_len" \
            --label_len "$label_len" \
            --pred_len "$pred_len" \
            --output_len "$output_len" \
            --patch_len "$patch_len" \
            --e_layers "$e_layers" \
            --factor "$factor" \
            --d_model "$d_model" \
            --d_ff "$d_ff" \
            --n_heads "$n_heads" \
            --batch_size "$batch_size_proto" \
            --num_workers "$num_workers" \
            --out_dir "$proto_extract_out" \
            --top_k "$top_k" \
            --target_layer "$target_layer" \
            --mi_mode low \
            --use_multi_gpu
    else
        python experiments/prototype_extraction.py \
            --ckpt_path "$ckpt_path" \
            --root_path "$root_path" \
            --data_path ${data}.csv \
            --data "$data" \
            --features "$features" \
            --seq_len "$seq_len" \
            --label_len "$label_len" \
            --pred_len "$pred_len" \
            --output_len "$output_len" \
            --patch_len "$patch_len" \
            --e_layers "$e_layers" \
            --factor "$factor" \
            --d_model "$d_model" \
            --d_ff "$d_ff" \
            --n_heads "$n_heads" \
            --batch_size "$batch_size_proto" \
            --num_workers "$num_workers" \
            --out_dir "$proto_extract_out" \
            --top_k "$top_k" \
            --target_layer "$target_layer" \
            --mi_mode low
    fi

    echo "[INFO] Low-MI 原型提取完成"
fi

# 检查原型文件是否存在
if [ ! -f "$proto_file" ]; then
    echo "[ERROR] High-MI 原型文件不存在: $proto_file"
    echo "[ERROR] 请先运行原型提取步骤"
    exit 1
fi

if [ ! -f "$low_proto_file" ]; then
    echo "[ERROR] Low-MI 原型文件不存在: $low_proto_file"
    echo "[ERROR] 请先运行原型提取步骤"
    exit 1
fi

mkdir -p "$proto_injection_out"

# ─────────────────────────────────────────────────────────────────────────────
# 阶段 3：原型注入推理对比实验
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "阶段 3: 原型注入推理对比实验"
echo "============================================================"
echo "High-MI 原型文件: $proto_file"
echo "Low-MI  原型文件: $low_proto_file"
echo "输出目录: $proto_injection_out"
echo "============================================================"
echo ""

if command -v torchrun &> /dev/null && [ -n "$WORLD_SIZE" ]; then
    echo "[INFO] 检测到 torchrun，使用多卡模式"
    torchrun --nnodes=1 --nproc_per_node=4 experiments/etth1_prototype_injection.py \
        --ckpt_path "$ckpt_path" \
        --prototype_path "$proto_file" \
        --low_proto_path "$low_proto_file" \
        --root_path "$root_path" \
        --data_path ${data}.csv \
        --data "$data" \
        --features "$features" \
        --seq_len "$seq_len" \
        --label_len "$label_len" \
        --pred_len "$pred_len" \
        --output_len "$output_len" \
        --patch_len "$patch_len" \
        --e_layers "$e_layers" \
        --factor "$factor" \
        --d_model "$d_model" \
        --d_ff "$d_ff" \
        --n_heads "$n_heads" \
        --batch_size "$batch_size_exp" \
        --num_workers "$num_workers" \
        --output_dir "$proto_injection_out" \
        --use_ims \
        --plot_samples \
        --use_multi_gpu
else
    echo "[INFO] 使用单卡模式"
    python experiments/etth1_prototype_injection.py \
        --ckpt_path "$ckpt_path" \
        --prototype_path "$proto_file" \
        --low_proto_path "$low_proto_file" \
        --root_path "$root_path" \
        --data_path ${data}.csv \
        --data "$data" \
        --features "$features" \
        --seq_len "$seq_len" \
        --label_len "$label_len" \
        --pred_len "$pred_len" \
        --output_len "$output_len" \
        --patch_len "$patch_len" \
        --e_layers "$e_layers" \
        --factor "$factor" \
        --d_model "$d_model" \
        --d_ff "$d_ff" \
        --n_heads "$n_heads" \
        --batch_size "$batch_size_exp" \
        --num_workers "$num_workers" \
        --output_dir "$proto_injection_out" \
        --use_ims \
        --plot_samples
fi

echo ""
echo "============================================================"
echo "Prototype Prompting 实验完成"
echo "============================================================"
echo "High-MI 原型文件: $proto_file"
echo "Low-MI  原型文件: $low_proto_file"
echo "实验结果: $proto_injection_out/comparison_results.txt"
echo "============================================================"
