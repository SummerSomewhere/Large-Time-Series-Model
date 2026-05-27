#!/bin/bash
#
# 合成数据 MI-HSIC 分析：freq_hz=0.042（周期≈24），四种 source 模式
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXP_DIR="$PROJECT_ROOT/experiments"

CKPT_PATH=${CKPT_PATH:-checkpoints/Timer_forecast_1.0.ckpt}
SEQ_LEN=${SEQ_LEN:-672}
PRED_LEN=${PRED_LEN:-96}
PATCH_LEN=${PATCH_LEN:-96}
E_LAYERS=${E_LAYERS:-8}
D_MODEL=${D_MODEL:-1024}
GPU_IDS=${GPU_IDS:-0}
N_SAMPLES=${N_SAMPLES:-20000}
TRAIN_RATIO=${TRAIN_RATIO:-0.7}

FREQ_HZ=${FREQ_HZ:-0.042}
declare -a SOURCES=("all" "trend" "periodic" "noise")

# 默认参数（etth1 保留）
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_BATCHES=${MAX_BATCHES:-0}
NUM_PLOT_SAMPLES=${NUM_PLOT_SAMPLES:-20}

echo "============================================================"
echo "  合成数据 MI-HSIC 分析（freq_hz=0.042，即周期≈24）"
echo "============================================================"
echo "  CKPT:       $CKPT_PATH"
echo "  SEQ_LEN:    $SEQ_LEN, PRED_LEN: $PRED_LEN"
echo "  N_SAMPLES:  $N_SAMPLES"
echo "  TRAIN_RATIO:$TRAIN_RATIO"
echo "  FREQ_HZ:    $FREQ_HZ"
echo "  SOURCES:    ${SOURCES[*]}"
echo "  GPU:        $GPU_IDS"
echo "============================================================"

for SOURCE in "${SOURCES[@]}"; do
    OUT_DIR="$PROJECT_ROOT/results/mi_hsic_synthetic_f${FREQ_HZ}_${SOURCE}"
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo "  [source=$SOURCE, freq_hz=$FREQ_HZ]"
    echo "══════════════════════════════════════════════════════════"

    python "$EXP_DIR/etth1_mi_hsic_peaks.py" \
        --data_type synthetic \
        --ckpt_path "$CKPT_PATH" \
        --n_samples $N_SAMPLES \
        --seq_len $SEQ_LEN \
        --pred_len $PRED_LEN \
        --label_len $((SEQ_LEN - PRED_LEN)) \
        --patch_len $PATCH_LEN \
        --stride 1 \
        --freq_hz "$FREQ_HZ" \
        --train_ratio $TRAIN_RATIO \
        --source "$SOURCE" \
        --batch_size $BATCH_SIZE \
        --max_batches $MAX_BATCHES \
        --num_random_plot_samples $NUM_PLOT_SAMPLES \
        --plot_seed 42 \
        --e_layers $E_LAYERS \
        --d_model $D_MODEL \
        --d_ff $((D_MODEL * 2)) \
        --n_heads 8 \
        --dropout 0.1 \
        --activation gelu \
        --factor 3 \
        --model_id "synthetic_f${FREQ_HZ}_${SOURCE}" \
        --out_dir "$OUT_DIR" \
        --no_save_peaks \
        --device cuda \
        --gpu_ids "$GPU_IDS"

    echo "  [完成] MI-HSIC 结果: $OUT_DIR"
done

echo ""
echo "============================================================"
echo "  全部完成！"
echo "  输出目录: $PROJECT_ROOT/results/mi_hsic_synthetic_f*"
echo "============================================================"
