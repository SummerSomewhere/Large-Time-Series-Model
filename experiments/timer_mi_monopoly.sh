#!/bin/bash
#===============================================================================
# 表征垄断测试：线性探测 (Linear Probing for Information Monopoly)
#
# 用法:
#   bash experiments/timer_mi_monopoly.sh --etth1           # ETTh1
#   bash experiments/timer_mi_monopoly.sh --etth2           # ETTh2
#   bash experiments/timer_mi_monopoly.sh --ettm1           # ETTm1
#   bash experiments/timer_mi_monopoly.sh --etth1_96        # ETTh1 pred_len=96
#   bash experiments/timer_mi_monopoly.sh --etth1_192       # ETTh1 pred_len=192
#   bash experiments/timer_mi_monopoly.sh --etth1_336       # ETTh1 pred_len=336
#   bash experiments/timer_mi_monopoly.sh --etth1_720       # ETTh1 pred_len=720 (复用已有MI文件)
#   bash experiments/timer_mi_monopoly.sh --random          # 随机初始化模型
#
# 可用 preset:
#   --debug      : 快速调试（小模型，少样本）
#   --random     : 使用随机初始化模型（无需下载预训练权重）
#   --etth1      : ETTh1, pred_len=96
#   --etth2      : ETTh2, pred_len=96
#   --ettm1      : ETTm1, pred_len=96
#   --etth1_192 : ETTh1, pred_len=192
#   --etth1_336 : ETTh1, pred_len=336
#   --etth1_720 : ETTh1, pred_len=720
#===============================================================================

set -e

# ── 默认参数 ──────────────────────────────────────────────────────────────────
CKPT_PATH="${CKPT_PATH:-checkpoints/Timer_forecast_1.0.ckpt}"
DATA="${DATA:-ETTh1}"
ROOT_PATH="${ROOT_PATH:-./datasets/}"
DATA_PATH="${DATA_PATH:-ETTh1.csv}"
SEQ_LEN="${SEQ_LEN:-672}"
PRED_LEN="${PRED_LEN:-96}"
PATCH_LEN="${PATCH_LEN:-96}"
E_LAYERS="${E_LAYERS:-8}"
D_MODEL="${D_MODEL:-1024}"
D_FF="${D_FF:-2048}"
N_HEADS="${N_HEADS:-8}"
DROPOUT="${DROPOUT:-0.1}"
N_SAMPLES="${N_SAMPLES:-2048}"
PROBE_EPOCHS="${PROBE_EPOCHS:-200}"
PROBE_LR="${PROBE_LR:-0.01}"
HIGH_MI_RATIO="${HIGH_MI_RATIO:-0.20}"
LOW_MI_RATIO="${LOW_MI_RATIO:-0.20}"
MI_K="${MI_K:-5}"
MI_FILE="${MI_FILE:-}"
EXTRACT_LAYER="${EXTRACT_LAYER:--1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
OUT_DIR="${OUT_DIR:-./outputs/timer_mi_monopoly}"

# ── Preset 选择 ───────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --debug)
            echo "[Preset] Debug mode: small model, few samples"
            CKPT_PATH="random"
            SEQ_LEN=192
            PRED_LEN=48
            PATCH_LEN=48
            E_LAYERS=4
            D_MODEL=256
            D_FF=512
            N_HEADS=4
            N_SAMPLES=512
            PROBE_EPOCHS=50
            EXTRACT_LAYER=-1
            OUT_DIR="./outputs/timer_mi_monopoly_debug"
            ;;
        --random)
            echo "[Preset] Random model (no pretrained weights)"
            CKPT_PATH="random"
            ;;
        --etth1)
            echo "[Preset] ETTh1 dataset"
            DATA="ETTh1"
            DATA_PATH="ETTh1.csv"
            ROOT_PATH="./datasets/"
            SEQ_LEN=672
            PRED_LEN=96
            PATCH_LEN=96
            E_LAYERS=8
            D_MODEL=1024
            D_FF=2048
            N_HEADS=8
            N_SAMPLES=2048
            ;;
        --etth2)
            echo "[Preset] ETTh2 dataset"
            DATA="ETTh2"
            DATA_PATH="ETTh2.csv"
            ROOT_PATH="./datasets/"
            SEQ_LEN=672
            PRED_LEN=96
            PATCH_LEN=96
            E_LAYERS=8
            D_MODEL=1024
            D_FF=2048
            N_HEADS=8
            N_SAMPLES=2048
            ;;
        --ettm1)
            echo "[Preset] ETTm1 dataset"
            DATA="ETTm1"
            DATA_PATH="ETTm1.csv"
            ROOT_PATH="./datasets/"
            SEQ_LEN=672
            PRED_LEN=96
            PATCH_LEN=96
            E_LAYERS=8
            D_MODEL=1024
            D_FF=2048
            N_HEADS=8
            N_SAMPLES=2048
            ;;
        --etth1_192)
            echo "[Preset] ETTh1, pred_len=192"
            DATA="ETTh1"
            DATA_PATH="ETTh1.csv"
            ROOT_PATH="./datasets/"
            SEQ_LEN=672
            PRED_LEN=192
            PATCH_LEN=96
            E_LAYERS=8
            D_MODEL=1024
            D_FF=2048
            N_HEADS=8
            N_SAMPLES=2048
            ;;
        --etth1_336)
            echo "[Preset] ETTh1, pred_len=336"
            DATA="ETTh1"
            DATA_PATH="ETTh1.csv"
            ROOT_PATH="./datasets/"
            SEQ_LEN=672
            PRED_LEN=336
            PATCH_LEN=96
            E_LAYERS=8
            D_MODEL=1024
            D_FF=2048
            N_HEADS=8
            N_SAMPLES=2048
            ;;
        --etth1_720)
            echo "[Preset] ETTh1, pred_len=720"
            DATA="ETTh1"
            DATA_PATH="ETTh1.csv"
            ROOT_PATH="./datasets/"
            SEQ_LEN=672
            PRED_LEN=720
            PATCH_LEN=96
            E_LAYERS=8
            D_MODEL=1024
            D_FF=2048
            N_HEADS=8
            N_SAMPLES=2048
            # 复用已有的 MI 文件（可选）
            MI_FILE="${MI_FILE:-global_mi_peaks_etth1.json}"
            ;;
        --etth1_mi_file)
            echo "[Preset] ETTh1 with precomputed MI file"
            DATA="ETTh1"
            DATA_PATH="ETTh1.csv"
            ROOT_PATH="./datasets/"
            SEQ_LEN=672
            PRED_LEN=96
            PATCH_LEN=96
            E_LAYERS=8
            D_MODEL=1024
            D_FF=2048
            N_HEADS=8
            N_SAMPLES=2048
            # 指定已有的 MI 文件路径（优先使用）
            MI_FILE="${MI_FILE:-global_mi_peaks_etth1.json}"
            ;;
        --gpu)
            # handled below
            ;;
        --*)
            echo "[Preset] Unknown: $arg, using defaults"
            ;;
    esac
done

# ── GPU 指定 ─────────────────────────────────────────────────────────────────
for arg in "$@"; do
    if [[ "$arg" == "--gpu" ]]; then
        continue
    fi
    if [[ "$arg" =~ ^--gpu= ]]; then
        GPU="${arg#--gpu=}"
    fi
done

# ── 日志 ──────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  表征垄断测试：线性探测 (Linear Probing)"
echo "  $(date)"
echo "============================================================"
echo "  CKPT_PATH      : $CKPT_PATH"
echo "  DATA           : $DATA"
echo "  SEQ_LEN        : $SEQ_LEN"
echo "  PRED_LEN       : $PRED_LEN"
echo "  PATCH_LEN      : $PATCH_LEN"
echo "  E_LAYERS       : $E_LAYERS"
echo "  D_MODEL        : $D_MODEL"
echo "  N_SAMPLES      : $N_SAMPLES"
echo "  HIGH_MI_RATIO  : $HIGH_MI_RATIO"
echo "  LOW_MI_RATIO   : $LOW_MI_RATIO"
echo "  PROBE_EPOCHS   : $PROBE_EPOCHS"
echo "  GPU            : $GPU"
echo "  OUT_DIR        : $OUT_DIR"
echo "============================================================"

# ── 检查 checkpoint ───────────────────────────────────────────────────────────
if [[ "$CKPT_PATH" != "random" && ! -f "$CKPT_PATH" ]]; then
    echo "[WARN] Checkpoint not found: $CKPT_PATH"
    echo "  请下载预训练模型或使用 --random 模式"
    echo "  下载地址: https://github.com/thuml/Large-Time-Series-Model"
    exit 1
fi

# ── 运行实验 ──────────────────────────────────────────────────────────────────
echo ""
echo ">>> 开始实验 (日志: $LOG_FILE)"
echo ""

CUDA_VISIBLE_DEVICES="$GPU" python experiments/timer_mi_monopoly.py \
    --ckpt_path "$CKPT_PATH" \
    --data "$DATA" \
    --root_path "$ROOT_PATH" \
    --data_path "$DATA_PATH" \
    --seq_len "$SEQ_LEN" \
    --pred_len "$PRED_LEN" \
    --patch_len "$PATCH_LEN" \
    --e_layers "$E_LAYERS" \
    --d_model "$D_MODEL" \
    --d_ff "$D_FF" \
    --n_heads "$N_HEADS" \
    --dropout "$DROPOUT" \
    --n_samples "$N_SAMPLES" \
    --probe_epochs "$PROBE_EPOCHS" \
    --probe_lr "$PROBE_LR" \
    --weight_decay 1e-4 \
    --high_mi_ratio "$HIGH_MI_RATIO" \
    --low_mi_ratio "$LOW_MI_RATIO" \
    --mi_k "$MI_K" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --gpu "$GPU" \
    --out_dir "$OUT_DIR" \
    ${MI_FILE:+--mi_file "$MI_FILE"} \
    --extract_layer "$EXTRACT_LAYER" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo ">>> 实验成功完成！"
    echo ">>> 结果目录: $OUT_DIR"
    echo ">>> 日志文件: $LOG_FILE"
else
    echo ">>> 实验失败 (exit code: $EXIT_CODE)"
    echo ">>> 查看日志: $LOG_FILE"
fi
