#!/bin/bash
#
# 合成数据 Layer-wise Probe 实验：11 种 data_mode
#
# Step 1: 特征提取 — 生成合成数据，通过 Timer 提取各层 hidden states
# Step 2: Linear Probe — Ridge + MLP 探测每层对各成分的解释力
# Step 3: 跨 Mode 对比可视化
#
# 11 种数据模式：
#   trend, periodic, noise, ar1, level_shift, random_walk,
#   spectral, time_warp, variance, trend_periodic, all
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXP_DIR="$PROJECT_ROOT/experiments"

# ── 特征提取参数 ─────────────────────────────────────────────────────────
CKPT_PATH=${CKPT_PATH:-checkpoints/Timer_forecast_1.0.ckpt}
N_SAMPLES=${N_SAMPLES:-20000}
SEQ_LEN=${SEQ_LEN:-672}
PRED_LEN=${PRED_LEN:-96}
PATCH_LEN=${PATCH_LEN:-96}
FREQ_HZ=${FREQ_HZ:-0.042}
E_LAYERS=${E_LAYERS:-8}
D_MODEL=${D_MODEL:-1024}
GPU_IDS=${GPU_IDS:-0}

FEATURE_OUT_DIR=${FEATURE_OUT_DIR:-$PROJECT_ROOT/results/synthetic_layerwise_probe_features}

# ── Probe 参数 ──────────────────────────────────────────────────────────
PROBE_OUT_DIR=${PROBE_OUT_DIR:-$PROJECT_ROOT/results/synthetic_layerwise_probe}
TRAIN_RATIO=${TRAIN_RATIO:-0.7}
ALPHA=${ALPHA:-1.0}

# ── 11 种 Mode 列表 ────────────────────────────────────────────────────
declare -a MODES=(
    "trend"
    "periodic"
    "noise"
    "ar1"
    "level_shift"
    "random_walk"
    "spectral"
    "time_warp"
    "variance"
    "trend_periodic"
    "all"
)

# ═══════════════════════════════════════════════════════════════════════════
echo "============================================================"
echo "  合成数据 Layer-wise Probe（freq_hz=0.042，周期≈24）"
echo "  11 种数据模式"
echo "============================================================"
echo "  CKPT:       $CKPT_PATH"
echo "  N_SAMPLES:  $N_SAMPLES"
echo "  SEQ_LEN:    $SEQ_LEN, PRED_LEN: $PRED_LEN"
echo "  FREQ_HZ:    $FREQ_HZ"
echo "  E_LAYERS:   $E_LAYERS"
echo "  MODES:      ${MODES[*]}"
echo "============================================================"

for MODE in "${MODES[@]}"; do
    FEATURE_NAME="synth_${N_SAMPLES}_f${FREQ_HZ}_${MODE}"
    FEATURE_PT="$FEATURE_OUT_DIR/${FEATURE_NAME}.pt"

    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo "  [mode=$MODE]"
    echo "══════════════════════════════════════════════════════════"

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1: 特征提取
    # ═══════════════════════════════════════════════════════════════════════
    if [ -f "$FEATURE_PT" ]; then
        echo "[Step 1] 跳过特征提取（已有: $FEATURE_PT）"
    else
        echo "[Step 1] 特征提取..."
        mkdir -p "$FEATURE_OUT_DIR"
        python "$EXP_DIR/synthetic_layerwise_probe_features.py" \
            --ckpt_path "$CKPT_PATH" \
            --n_samples $N_SAMPLES \
            --seq_len $SEQ_LEN \
            --pred_len $PRED_LEN \
            --patch_len $PATCH_LEN \
            --freq_hz $FREQ_HZ \
            --batch_size 64 \
            --e_layers $E_LAYERS \
            --d_model $D_MODEL \
            --d_ff $((D_MODEL * 2)) \
            --n_heads 8 \
            --dropout 0.1 \
            --activation gelu \
            --factor 3 \
            --seed 42 \
            --train_ratio $TRAIN_RATIO \
            --data_mode "$MODE" \
            --out_dir "$FEATURE_OUT_DIR" \
            --out_name "$FEATURE_NAME" \
            --gpu_ids "$GPU_IDS"
    fi

    # ═══════════════════════════════════════════════════════════════════════
    # Step 2: Ridge + MLP Probe
    # ═══════════════════════════════════════════════════════════════════════
    MODE_PROBE_DIR="$PROBE_OUT_DIR/mode_${MODE}"
    echo "[Step 2] Layer-wise Probe..."
    mkdir -p "$MODE_PROBE_DIR"
    python "$EXP_DIR/synthetic_layerwise_probe.py" \
        --features_pt_path "$FEATURE_PT" \
        --n_samples 0 \
        --alpha $ALPHA \
        --probe_type both \
        --mlp_hidden_dims 256,128 \
        --mlp_epochs 100 \
        --mlp_lr 1e-3 \
        --mlp_dropout 0.2 \
        --mlp_batch_size 256 \
        --train_ratio $TRAIN_RATIO \
        --out_dir "$MODE_PROBE_DIR"

    echo "  [完成] $MODE -> $MODE_PROBE_DIR"
done

# ── Step 3: 跨 Mode 对比图 ───────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  [Step 3] 跨 Mode 对比可视化"
echo "══════════════════════════════════════════════════════════"
python "$EXP_DIR/synthetic_layerwise_probe.py" \
    --compare_modes \
    --out_dir "$PROBE_OUT_DIR" \
    --modes "trend,periodic,noise,ar1,level_shift,random_walk,spectral,time_warp,variance,trend_periodic,all" \
    --compare_probe_type both

echo ""
echo "============================================================"
echo "  全部完成！"
echo "  特征:  $FEATURE_OUT_DIR/synth_${N_SAMPLES}_f${FREQ_HZ}_*.pt"
echo "  结果:  $PROBE_OUT_DIR/mode_*"
echo "  对比图: $PROBE_OUT_DIR/*_cross_mode_*.png"
echo "============================================================"
