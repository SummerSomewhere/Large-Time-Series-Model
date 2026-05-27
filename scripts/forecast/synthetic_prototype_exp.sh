#!/bin/bash
#
# 合成数据原型注入实验脚本
#
# 功能：
#   1. 用 sin 函数样本提取原型
#   2. 用 y = ax + b 线性函数测试泛化效果
#   3. 绘制引导前后预测对比图
#
# 使用方式：
#   # 默认 Q3 分组
#   bash scripts/forecast/synthetic_prototype_exp.sh
#
#   # 指定 top-k 分组（取 MI 最高的 3 个和最低的 3 个）
#   bash scripts/forecast/synthetic_prototype_exp.sh --top_k 3
#
#   # 分别指定高/低 MI 的 patch 数
#   bash scripts/forecast/synthetic_prototype_exp.sh --top_k 5 --k_low 3
#

set -e

_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$_ROOT" || exit 1

# ─────────────────────────────────────────────────────────────────────────────
# 实验参数配置
# ─────────────────────────────────────────────────────────────────────────────

# GPU 配置
export CUDA_VISIBLE_DEVICES=0

# 模型参数
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt

# 数据参数
seq_len=672
pred_len=96
patch_len=96
batch_size=64

# 实验参数
sin_samples=500
test_samples=500

# 原型提取参数（分组模式）
# top_k=0: 使用 Q3 分组（默认）
# top_k=N: 取 MI 最高的 N 个和最低的 N 个 patch
top_k=1
k_low=1  # 低 MI 组 patch 数（0 = 等于 top_k）

# 输出目录
output_dir=./results/synthetic_exp

# ─────────────────────────────────────────────────────────────────────────────
# 解析命令行参数
# ─────────────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --top_k)
            top_k="$2"
            shift 2
            ;;
        --k_low)
            k_low="$2"
            shift 2
            ;;
        --sin_samples)
            sin_samples="$2"
            shift 2
            ;;
        --test_samples)
            test_samples="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# 运行实验
# ─────────────────────────────────────────────────────────────────────────────

group_desc="Q3 分组"
if [[ "$top_k" -gt 0 ]]; then
    if [[ "$k_low" -gt 0 && "$k_low" -ne "$top_k" ]]; then
        group_desc="Top-K 分组 (high=${top_k}, low=${k_low})"
    else
        group_desc="Top-K 分组 (k=${top_k})"
    fi
fi

echo ""
echo "============================================================"
echo "合成数据原型注入实验: sin → y=1 泛化"
echo "============================================================"
echo "分组模式: $group_desc"
echo "sin 样本数: $sin_samples"
echo "测试样本数: $test_samples"
echo "序列长度: $seq_len"
echo "预测长度: $pred_len"
echo "输出目录: $output_dir"
echo "============================================================"
echo ""

python3 experiments/synthetic_prototype_exp.py \
    --ckpt_path "$ckpt_path" \
    --device cuda \
    --sin_samples "$sin_samples" \
    --test_samples "$test_samples" \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --batch_size "$batch_size" \
    --output_dir "$output_dir" \
    --patch_len "$patch_len" \
    --stride "$patch_len" \
    --d_model 1024 \
    --e_layers 8 \
    --n_heads 8 \
    --factor 3 \
    --top_k "$top_k" \
    --k_low "$k_low" \
    --source_type sin

echo ""
echo "============================================================"
echo "实验完成"
echo "============================================================"
echo "输出目录: $output_dir"
echo "============================================================"
