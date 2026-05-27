#!/bin/bash
# 表征垄断测试 — Linear Probing for Information Monopoly
# 证明高 MI Patch 表征就是未来信息的"垄断"载体
#
# 依赖：
#   1. timer_mi_ksg_pca.py 输出 (--mi_result_dir 指向 global_mi_peaks_*.json)
#   2. 预训练 Timer checkpoint (--ckpt_path)
#
# 建议先跑:
#   python experiments/timer_mi_ksg_pca.py \
#       --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
#       --root_path ./datasets/ --data ETTh1 --data_path ETTh1.csv \
#       --out_dir ./results/timer_mi_ksg_pca/ --model_id etth1

set -e  # 任何命令失败则退出
set -u  # 使用未定义变量则报错

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these before running
# ═══════════════════════════════════════════════════════════════════════════════

# ── Model & Data ────────────────────────────────────────────────────────────
CKPT_PATH="${CKPT_PATH:-checkpoints/Timer_forecast_1.0.ckpt}"
MI_RESULT_DIR="${MI_RESULT_DIR:-./outputs/timer_mi_ksg_pca/Timer_MI_20260516_133108}"
ROOT_PATH="${ROOT_PATH:-./datasets}"

# ── Dataset ─────────────────────────────────────────────────────────────────
DATA_PATH="${DATA_PATH:-ETTh1.csv}"      # ETTh1, ETTh2, ETTm1, ETTm2
DATA_TYPE="${DATA_TYPE:-ETTh1}"
FREQ="${FREQ:-h}"

# ── Architecture ─────────────────────────────────────────────────────────────
SEQ_LEN="${SEQ_LEN:-672}"
PRED_LEN="${PRED_LEN:-96}"
PATCH_LEN="${PATCH_LEN:-96}"
D_MODEL="${D_MODEL:-1024}"
D_FF="${D_FF:-2048}"
E_LAYERS="${E_LAYERS:-8}"
N_HEADS="${N_HEADS:-8}"
DROPOUT="${DROPOUT:-0.1}"

# ── Probe ───────────────────────────────────────────────────────────────────
# 高 MI / 低 MI patch 的比例（默认各 15%，剩余 70% 中间 patches 不使用）
HIGH_MI_RATIO="${HIGH_MI_RATIO:-0.15}"
LOW_MI_RATIO="${LOW_MI_RATIO:-0.15}"
PROBE_EPOCHS="${PROBE_EPOCHS:-100}"
PROBE_LR="${PROBE_LR:-1e-3}"

# ── Experiment mode ──────────────────────────────────────────────────────────
# 目标层: -1=最后一层, 0=第一层, 7=第八层（最后一层），
# 或设置 LAYERS="0,1,2,3,7" 多层模式
TARGET_LAYER="${TARGET_LAYER:--1}"
LAYERS="${LAYERS:-}"       # e.g. "0,1,2,3,7", overrides TARGET_LAYER if set

# ── Output ───────────────────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-./outputs/timer_mi_monopoly_probe}"

# ── GPU ──────────────────────────────────────────────────────────────────────
GPU="${GPU:-0}"

# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

LAYERS_ARG=""
if [ -n "${LAYERS}" ]; then
    LAYERS_ARG="--layers ${LAYERS}"
fi

TARGET_LAYER_ARG="--target_layer ${TARGET_LAYER}"

python experiments/timer_mi_monopoly_probe.py \
    --mi_result_dir "${MI_RESULT_DIR}" \
    --root_path "${ROOT_PATH}" \
    --data_path "${DATA_PATH}" \
    --data_type "${DATA_TYPE}" \
    --seq_len "${SEQ_LEN}" \
    --pred_len "${PRED_LEN}" \
    --patch_len "${PATCH_LEN}" \
    --stride "${PATCH_LEN}" \
    --d_model "${D_MODEL}" \
    --d_ff "${D_FF}" \
    --e_layers "${E_LAYERS}" \
    --n_heads "${N_HEADS}" \
    --dropout "${DROPOUT}" \
    --ckpt_path "${CKPT_PATH}" \
    --probe_epochs "${PROBE_EPOCHS}" \
    --probe_lr "${PROBE_LR}" \
    --high_mi_ratio "${HIGH_MI_RATIO}" \
    --low_mi_ratio "${LOW_MI_RATIO}" \
    ${TARGET_LAYER_ARG} \
    ${LAYERS_ARG} \
    --gpu "${GPU}" \
    --freq "${FREQ}" \
    --seed 42 \
    --out_dir "${OUT_DIR}"

echo ""
echo "✓ 表征垄断测试完成"
echo "  结果目录: ${OUT_DIR}/run_*/"
echo ""
echo "  主要输出文件:"
echo "    monopoly_summary.png     — 多面板总览图（三曲线叠加 + 柱状图 + MI曲线）"
echo "    reconstruction_best.png — 最佳样本重建曲线"
echo "    reconstruction_median.png— 中位样本重建曲线"
echo "    reconstruction_worst.png — 最差样本重建曲线"
echo "    monopoly_bar_comparison.png— MSE/R² 柱状对比"
echo "    mi_curve_and_mse_dist.png — MI曲线 + 样本MSE分布"
echo "    probe_convergence.png    — 探针训练收敛曲线"
echo "    multilayer_monopoly_comparison.png — 多层对比"
echo "    monopoly_results.pt      — 完整结果（含所有预测）"
echo "    monopoly_summary.json    — 结果摘要"
