#!/bin/bash
# ============================================================
# Timer MI 分析：计算 MI (HSIC)
# ============================================================
# 用法：
#   bash MI-Peaks/src/scripts/timer_calculate_mi.sh

# 设置路径
ROOT_DIR="/Users/summer/Large-Time-Series-Model"
cd "$ROOT_DIR" || exit 1

# 数据激活目录
DATA_DIR="MI-Peaks/acts/timer"

# 数据集配置
DATA="ETTh1"

# 模型标识
MODEL_TAG="timer_mi"

# MI 计算配置
# 要计算的层索引（None 表示全部）
LAYERS=""  # 空表示全部层

# 样本数量（-1 表示全部）
SAMPLE_NUM=-1

# 最小 batch 大小（HSIC 需要 B >= 4）
MIN_BATCH_SIZE=4

# 输出目录
SAVE_DIR="results/mi/timer"

echo "=========================================="
echo "Timer MI (HSIC) 计算"
echo "=========================================="
echo "数据集: $DATA"
echo "激活目录: $DATA_DIR"
echo "模型标识: $MODEL_TAG"
echo "最小 Batch: $MIN_BATCH_SIZE"
echo "输出目录: $SAVE_DIR"
echo ""

# 创建输出目录
mkdir -p "$SAVE_DIR"

# 运行 MI 计算
python MI-Peaks/src/timer_calculate_mi.py \
    --root_path "./datasets/" \
    --data "$DATA" \
    --data_path "${DATA}.csv" \
    --data_dir "$DATA_DIR" \
    --model_tag "$MODEL_TAG" \
    --gt_model_tag "$MODEL_TAG" \
    --sample_num $SAMPLE_NUM \
    --min_batch_size $MIN_BATCH_SIZE \
    --save_dir "$SAVE_DIR" \
    ${LAYERS:+"--layers"} $LAYERS

echo ""
echo "MI 计算完成！"
echo "输出文件: $SAVE_DIR/${DATA}_gt=${MODEL_TAG}_test=${MODEL_TAG}.pth"
