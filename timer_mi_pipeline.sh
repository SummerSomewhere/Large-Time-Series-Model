#!/bin/bash
# ============================================================
# Timer MI 分析：一键运行脚本
# ============================================================
# 依次执行：
#   1. 生成输入序列激活
#   2. 生成真值序列激活
#   3. 计算 MI (HSIC)
#
# 用法：
#   bash MI-Peaks/src/scripts/timer_mi_pipeline.sh

# 设置路径
ROOT_DIR="/Users/summer/Large-Time-Series-Model"
cd "$ROOT_DIR" || exit 1

echo "=========================================="
echo "Timer MI 分析完整流程"
echo "=========================================="
echo ""

# ===========================================
# 步骤 1: 生成输入序列激活
# ===========================================
echo ">>> 步骤 1/3: 生成输入序列激活"
echo "----------------------------------------"
bash MI-Peaks/src/scripts/timer_generate_activation.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "[错误] 步骤 1 失败！"
    exit 1
fi

echo ""
echo ">>> 步骤 1 完成！"
echo ""

# ===========================================
# 步骤 2: 生成真值序列激活
# ===========================================
echo ">>> 步骤 2/3: 生成真值序列激活"
echo "----------------------------------------"
bash MI-Peaks/src/scripts/timer_generate_gt_activation.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "[错误] 步骤 2 失败！"
    exit 1
fi

echo ""
echo ">>> 步骤 2 完成！"
echo ""

# ===========================================
# 步骤 3: 计算 MI (HSIC)
# ===========================================
echo ">>> 步骤 3/3: 计算 MI (HSIC)"
echo "----------------------------------------"
bash MI-Peaks/src/scripts/timer_calculate_mi.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "[错误] 步骤 3 失败！"
    exit 1
fi

echo ""
echo ">>> 步骤 3 完成！"
echo ""

# ===========================================
# 完成
# ===========================================
echo "=========================================="
echo "Timer MI 分析完成！"
echo "=========================================="
echo ""
echo "输出文件："
echo "  - 输入激活: MI-Peaks/acts/timer/reasoning_evolve/ETTh1_timer_mi.pth"
echo "  - 真值激活: MI-Peaks/acts/timer/gt/ETTh1_timer_mi.pth"
echo "  - MI 结果: results/mi/timer/ETTh1_gt=timer_mi_test=timer_mi.pth"
echo ""
