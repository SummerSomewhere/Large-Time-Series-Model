#!/bin/bash
# ============================================================
# Timer MI 分析：生成真值序列激活
# ============================================================
# 用法：
#   bash MI-Peaks/src/scripts/timer_generate_gt_activation.sh

# 设置路径
ROOT_DIR="/Users/summer/Large-Time-Series-Model"
cd "$ROOT_DIR" || exit 1

# 输出目录
OUTPUT_DIR="MI-Peaks/acts/timer"

# 模型 Checkpoint
CKPT_PATH="checkpoints/etth1_checkpoint.pth"

# 数据集配置
DATA="ETTh1"
DATA_PATH="ETTh1.csv"
SEQ_LEN=672
LABEL_LEN=576
PRED_LEN=96
PATCH_LEN=96
STRIDE=96

# 模型配置
D_MODEL=1024
D_FF=2048
E_LAYERS=8
N_HEADS=8
FACTOR=3
DROPOUT=0.1

# 激活收集配置
# 要收集的层索引（可设置只收集部分层以加速）
# LAYERS="0 1 2 3 4 5 6 7"  # 收集前 8 层
LAYERS=""  # 空表示收集全部层

# 样本数量（-1 表示全部）
SAMPLE_NUM=-1

# 设备
DEVICE="cuda"

# 是否使用 IMS 模式
USE_IMS=""  # 不使用 IMS

echo "=========================================="
echo "Timer 激活提取：真值序列"
echo "=========================================="
echo "数据集: $DATA"
echo "序列长度: $SEQ_LEN"
echo "预测长度: $PRED_LEN"
echo "Patch 长度: $PATCH_LEN"
echo "输出目录: $OUTPUT_DIR"
echo "Checkpoint: $CKPT_PATH"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR/gt"

# 运行激活提取脚本
python MI-Peaks/src/timer_generate_gt_activation.py \
    --root_path "./datasets/" \
    --data_path "$DATA_PATH" \
    --data "$DATA" \
    --features "M" \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --patch_len $PATCH_LEN \
    --stride $STRIDE \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --e_layers $E_LAYERS \
    --n_heads $N_HEADS \
    --factor $FACTOR \
    --dropout $DROPOUT \
    --output_dir "$OUTPUT_DIR" \
    --sample_num $SAMPLE_NUM \
    --batch_size 32 \
    --num_workers 4 \
    --device "$DEVICE" \
    --ckpt_path "$CKPT_PATH" \
    ${LAYERS:+"--layers"} $LAYERS \
    --pool_over_vars \
    ${USE_IMS:+"--use_ims"}

echo ""
echo "真值序列激活提取完成！"
echo "输出文件: $OUTPUT_DIR/gt/${DATA}_timer_mi.pth"
