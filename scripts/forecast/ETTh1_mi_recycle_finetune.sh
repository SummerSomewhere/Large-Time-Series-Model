#!/bin/sh
# ETTh1: 微调 + 测试，结构与 ETTh1_mi_recycle_ablation.sh 类似，但走 run.py 的 finetune 流程。
#
# 两次独立运行（checkpoint 分目录）：
#   1) Baseline：关闭 RR，从预训练 ckpt 微调并在测试集评估。
#   2) With RR：第 0/7 层、patch 3/6、两轮 alpha 0.4/0.2（可调），微调 + 测试。
#
# 超参对齐 ETTh1.sh 风格：sl672 ll576 pl96 IMS patch96；默认单卡 python run.py。
# 多卡可设 USE_TORCHRUN=1（需 2 张 GPU，见下方）。
#
# Usage:
#   bash scripts/forecast/ETTh1_mi_recycle_finetune.sh
#   CKPT=checkpoints/Timer_forecast_1.0.ckpt FINETUNE_EPOCHS=5 bash scripts/forecast/ETTh1_mi_recycle_finetune.sh
#
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"

CKPT="${CKPT:-checkpoints/Timer_forecast_1.0.ckpt}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/mi_recycle_finetune}"
RECYCLE_LAYERS="${RECYCLE_LAYERS:-0,7}"
RECYCLE_PATCHES="${RECYCLE_PATCHES:-3,6}"
RECYCLE_ROUND_ALPHAS="${RECYCLE_ROUND_ALPHAS:-0.4,0.2}"
HSIC_NPY="${HSIC_NPY:-}"
PEAK_MODE="${PEAK_MODE:-tukey}"

FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
PATIENCE="${PATIENCE:-3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-1}"

mkdir -p "$OUT_DIR"

# Shared training/data args (match ablation geometry + ETTh1.sh IMS); one line for sh word-splitting.
COMMON="--task_name forecast --is_training 1 --is_finetuning 1 --model Timer --features M --seq_len 672 --label_len 576 --pred_len 96 --output_len 96 --e_layers 8 --factor 3 --d_model 1024 --d_ff 2048 --n_heads 8 --dropout 0.1 --embed timeF --freq h --activation gelu --patch_len 96 --use_ims --data ETTh1 --root_path ./datasets/ETT-small/ --data_path ETTh1.csv --num_workers 4 --itr 1 --seed $SEED --finetune_epochs $FINETUNE_EPOCHS --learning_rate $LEARNING_RATE --patience $PATIENCE --train_test 0 --subset_rand_ratio 1 --ckpt_path $CKPT"

_run_one() {
  if [ "${USE_TORCHRUN:-0}" = "1" ]; then
    torchrun --nnodes=1 --nproc_per_node=2 run.py "$@" --use_multi_gpu --gpu 0
  else
    "$PYTHON" run.py "$@" --gpu 0
  fi
}

echo "=== (1/2) Finetune baseline (no recycle), batch=$BATCH_SIZE ==="
export FORECAST_TEST_METRICS_JSON="$OUT_DIR/baseline_test_metrics.json"
_run_one $COMMON \
  --model_id etth1_mi_ft_baseline \
  --des mi_recycle_ft_base \
  --batch_size "$BATCH_SIZE" \
  --recycle_encoder_layer -1 --recycle_encoder_layers "" \
  --recycle_patch_indices "" --recycle_round_alphas ""

echo "=== (2/2) Finetune with RR layers=$RECYCLE_LAYERS patches=$RECYCLE_PATCHES rounds=$RECYCLE_ROUND_ALPHAS ==="
export FORECAST_TEST_METRICS_JSON="$OUT_DIR/rr_test_metrics.json"
if [ -n "$HSIC_NPY" ] && [ -f "$HSIC_NPY" ]; then
  _run_one $COMMON \
    --model_id etth1_mi_ft_rr \
    --des mi_recycle_ft_rr \
    --batch_size "$BATCH_SIZE" \
    --recycle_encoder_layer -1 \
    --recycle_encoder_layers "$RECYCLE_LAYERS" \
    --recycle_hsic_mean_npy "$HSIC_NPY" \
    --recycle_peak_mode "$PEAK_MODE" \
    --recycle_round_alphas "$RECYCLE_ROUND_ALPHAS"
else
  _run_one $COMMON \
    --model_id etth1_mi_ft_rr \
    --des mi_recycle_ft_rr \
    --batch_size "$BATCH_SIZE" \
    --recycle_encoder_layer -1 \
    --recycle_encoder_layers "$RECYCLE_LAYERS" \
    --recycle_patch_indices "$RECYCLE_PATCHES" \
    --recycle_round_alphas "$RECYCLE_ROUND_ALPHAS"
fi

echo "=== Test metrics (if JSON written) ==="
"$PYTHON" - "$OUT_DIR" << 'PY'
import json
import pathlib
import sys

d = pathlib.Path(sys.argv[1])
for label, name in [("baseline", "baseline_test_metrics.json"), ("rr", "rr_test_metrics.json")]:
    p = d / name
    if not p.is_file():
        print(f"{label}: (no file) {p}")
        continue
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        print(f"{label}: empty json")
        continue
    row = data[0]
    print(f"{label}: MSE={row.get('mse')} MAE={row.get('mae')} output_len={row.get('output_len')}")
PY

echo "Checkpoints under ./checkpoints/ (setting names contain model_id + timestamp)."
echo "Metrics JSON: $OUT_DIR/baseline_test_metrics.json , $OUT_DIR/rr_test_metrics.json"
