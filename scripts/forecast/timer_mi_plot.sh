#!/bin/bash
# ============================================================
# Timer MI 绘图脚本
# 绘制每个样本的每一层的每个 patch 的 MI 图
# ============================================================

# 设置路径
ROOT_DIR="$HOME/Large-Time-Series-Model"
cd "$ROOT_DIR" || exit 1

# ── 参数配置 ─────────────────────────────────────────────────
data=ETTh1
model_id=timer_mi
mi_dir=results/mi/timer
save_dir=results/mi/timer/figures
plot_type=all

# ── GPU 配置 ─────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=4,5,6,7

# ── 绘制 MI 图 ─────────────────────────────────────────────────
python timer_plot_mi.py \
  --data $data \
  --model_id $model_id \
  --mi_dir $mi_dir \
  --save_dir $save_dir \
  --plot_type $plot_type

echo ""
echo "图片保存到: $save_dir/"
