#!/bin/bash
# =============================================================================
# Timer 层间 MI + Linear Probing 实验启动脚本
#
# Pipeline:
#   1. 生成 7 种合成时间序列概念数据集 (AR1, LevelShift, RandomWalk, Spectral,
#      TimeWarp, Trend, VarianceShift)
#   2. 加载 Timer checkpoint，提取每层 Token 表示
#   3. 计算 per-token SPI / I(Hx, hY) / I(ex, Hx)
#   4. Group 1: Baseline 线性探针（层内均值池化 + Ridge Regression）
#   5. Group 2: Anchor vs Noise 对比实验（Top/Bottom 10% Token，不均化）
#   6. 绘制结果图（类似论文 Figure 2，双纵轴）
#
# Features:
#   - 支持选择 SPI、I(H,Y) 或 I(X,H) 作为 Token 筛选指标
#   - 支持可配置的概念数量（1-7）、采样比例、锚点比例
#   - 支持 --ckpt_path random 进行随机初始化模型的快速调试
#
# Usage:
#   # 完整实验（需 GPU + Timer checkpoint）:
#   bash scripts/forecast/timer_mi_probe.sh
#
#   # 快速测试（随机模型，较小规模）:
#   bash scripts/forecast/timer_mi_probe.sh --debug
#
#   # 自定义参数（选择不同筛选指标）:
#   bash scripts/forecast/timer_mi_probe.sh \
#       --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
#       --n_samples 4096 --e_layers 8 --select_metric spi
#   # 可选指标: spi (默认), ihy (I(H,Y)), ixh (I(X,H))
# =============================================================================

set -e

# ── Default Hyperparameters ──────────────────────────────────────────────────
CKPT_PATH=${CKPT_PATH:-checkpoints/Timer_forecast_1.0.ckpt}
SEQ_LEN=${SEQ_LEN:-672}
PRED_LEN=${PRED_LEN:-96}
PATCH_LEN=${PATCH_LEN:-96}
D_MODEL=${D_MODEL:-1024}
D_FF=${D_FF:-2048}
E_LAYERS=${E_LAYERS:-8}
N_HEADS=${N_HEADS:-8}
DROPOUT=${DROPOUT:-0.1}
BATCH_SIZE=${BATCH_SIZE:-64}
N_SAMPLES=${N_SAMPLES:-1000}
N_CONCEPTS=${N_CONCEPTS:-7}
K_NEIGHBORS=${K_NEIGHBORS:-3}
PCA_DIM=${PCA_DIM:-32}
SAMPLE_RATIO=${SAMPLE_RATIO:-1.0}
PROBE_ALPHA=${PROBE_ALPHA:-1.0}
ANCHOR_RATIO=${ANCHOR_RATIO:-0.20}
SELECT_METRIC=${SELECT_METRIC:-spi}
MODEL_ID=${MODEL_ID:-probe}
GPU=${GPU:-0}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/timer_mi_probe}

# Debug mode: uses random model, smaller scale
DEBUG_MODE=${DEBUG_MODE:-0}

# ── GPU setup ────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=$GPU

# ── Navigate to project root ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Parse command-line overrides ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG_MODE=1
            shift
            ;;
        --ckpt_path|--n_samples|--e_layers|--select_metric|--gpu|--out_dir)
            eval "${1#--}=$2"
            shift 2
            ;;
        -*)
            eval "${1#--}=$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# ── Debug mode overrides ────────────────────────────────────────────────────
if [[ "$DEBUG_MODE" == "1" ]]; then
    echo "  [DEBUG MODE] 使用随机初始化模型"
    CKPT_PATH="random"
    E_LAYERS=4
    SEQ_LEN=192
    PRED_LEN=48
    PATCH_LEN=48
    N_SAMPLES=256
    N_CONCEPTS=3
    SAMPLE_RATIO=1.0
    OUTPUT_DIR="${OUTPUT_DIR}_debug"
fi

echo "=========================================================="
echo "  Timer MI + Linear Probing 实验"
echo "=========================================================="
echo "  ckpt_path   : $CKPT_PATH"
echo "  seq_len     : $SEQ_LEN"
echo "  pred_len    : $PRED_LEN"
echo "  patch_len   : $PATCH_LEN"
echo "  e_layers    : $E_LAYERS"
echo "  n_samples   : $N_SAMPLES"
echo "  n_concepts  : $N_CONCEPTS"
echo "  select_metric: $SELECT_METRIC"
echo "  anchor_ratio: $ANCHOR_RATIO"
echo "  sample_ratio: $SAMPLE_RATIO"
echo "  pca_dim     : $PCA_DIM"
echo "  k_neighbors : $K_NEIGHBORS"
echo "  probe_alpha : $PROBE_ALPHA"
echo "  gpu         : $GPU"
echo "  out_dir     : $OUTPUT_DIR"
echo "=========================================================="

python experiments/timer_mi_probe.py \
    --ckpt_path "$CKPT_PATH" \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --e_layers $E_LAYERS \
    --n_heads $N_HEADS \
    --dropout $DROPOUT \
    --seq_len $SEQ_LEN \
    --pred_len $PRED_LEN \
    --patch_len $PATCH_LEN \
    --batch_size $BATCH_SIZE \
    --n_samples $N_SAMPLES \
    --n_concepts $N_CONCEPTS \
    --k_neighbors $K_NEIGHBORS \
    --pca_dim $PCA_DIM \
    --sample_ratio $SAMPLE_RATIO \
    --probe_alpha $PROBE_ALPHA \
    --select_metric $SELECT_METRIC \
    --anchor_ratio $ANCHOR_RATIO \
    --gpu $GPU \
    --out_dir "$OUTPUT_DIR" \
    --run_group2

echo ""
echo "=========================================================="
echo "  实验完成！结果保存于: $OUTPUT_DIR"
echo "=========================================================="
