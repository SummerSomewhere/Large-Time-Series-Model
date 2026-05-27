#!/bin/sh
# ETTh1: 对比两次独立微调 + 测试（与 mi_recycle_finetune 相同数据/几何）：
#   1) Baseline：无 VIB。默认 BASELINE_FINETUNE=proj_only（冻结 backbone，只训 patch 线性头 proj），
#      与 (2) 公平对照（仅差 VIB 两条支路 + proj）。若设 BASELINE_FINETUNE=full 则为全参微调。
#   2) VIB：--timer_vib 1，默认仅训 VIB + 预测头（VIB_FINETUNE=vib_proj：mu/logvar/proj），
#      backbone 冻结，不做全量微调。若坚持全参：VIB_FINETUNE=full bash ...
# 说明：超参（lr/batch/epoch 等）两次共用 COMMON。
# Loss：MSE（+ 可选 loss_fft_alpha）+ beta*KL；beta 前 vib_warmup_ratio 步为 0，再线性升至 vib_beta_max。
#
# Usage:
#   bash scripts/forecast/ETTh1_timer_vib_compare.sh
#   VIB_BETA_MAX=5e-5 FINETUNE_EPOCHS=10 bash scripts/forecast/ETTh1_timer_vib_compare.sh
#
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"

CKPT="${CKPT:-checkpoints/Timer_forecast_1.0.ckpt}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/timer_vib_compare}"

FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
PATIENCE="${PATIENCE:-3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-1}"
VIB_BETA_MAX="${VIB_BETA_MAX:-1e-4}"
VIB_WARMUP_RATIO="${VIB_WARMUP_RATIO:-0.1}"
# proj_only = 只训 proj；full = 全参微调
BASELINE_FINETUNE="${BASELINE_FINETUNE:-proj_only}"
# 第二次（VIB）：默认 vib_proj = 仅 mu/logvar/proj；full = 全参（一般不用）
VIB_FINETUNE="${VIB_FINETUNE:-vib_proj}"

mkdir -p "$OUT_DIR"
# 避免沿用上一次中断/仅测试 run 留下的 JSON，导致「baseline 像没微调」的错觉
rm -f "$OUT_DIR/baseline_test_metrics.json" "$OUT_DIR/vib_test_metrics.json"

COMMON="--task_name forecast --is_training 1 --is_finetuning 1 --model Timer --features M --seq_len 672 --label_len 576 --pred_len 96 --output_len 96 --e_layers 8 --factor 3 --d_model 1024 --d_ff 2048 --n_heads 8 --dropout 0.1 --embed timeF --freq h --activation gelu --patch_len 96 --use_ims --data ETTh1 --root_path ./datasets/ETT-small/ --data_path ETTh1.csv --num_workers 4 --itr 1 --seed $SEED --finetune_epochs $FINETUNE_EPOCHS --learning_rate $LEARNING_RATE --patience $PATIENCE --train_test 0 --subset_rand_ratio 1 --ckpt_path $CKPT"

_run_one() {
  if [ "${USE_TORCHRUN:-0}" = "1" ]; then
    torchrun --nnodes=1 --nproc_per_node=2 run.py "$@" --use_multi_gpu --gpu 0
  else
    "$PYTHON" run.py "$@" --gpu 0
  fi
}

echo "=== (1/2) Baseline: no VIB, finetune_trainable=$BASELINE_FINETUNE, batch=$BATCH_SIZE ==="
export FORECAST_TEST_METRICS_JSON="$OUT_DIR/baseline_test_metrics.json"
_run_one $COMMON \
  --model_id etth1_vib_baseline \
  --des timer_vib_base \
  --batch_size "$BATCH_SIZE" \
  --finetune_trainable "$BASELINE_FINETUNE" \
  --timer_vib 0 \
  --recycle_encoder_layer -1 \
  --recycle_encoder_layers "" \
  --recycle_patch_indices "" \
  --recycle_round_alphas ""
echo "=== Done (1/2) baseline: metrics -> $OUT_DIR/baseline_test_metrics.json ==="

echo "=== (2/2) VIB: timer_vib=1, finetune_trainable=$VIB_FINETUNE (default vib_proj=VIB+proj only), beta_max=$VIB_BETA_MAX warmup=$VIB_WARMUP_RATIO ==="
export FORECAST_TEST_METRICS_JSON="$OUT_DIR/vib_test_metrics.json"
_run_one $COMMON \
  --model_id etth1_vib_adapter \
  --des timer_vib_adapter \
  --batch_size "$BATCH_SIZE" \
  --finetune_trainable "$VIB_FINETUNE" \
  --timer_vib 1 \
  --vib_beta_max "$VIB_BETA_MAX" \
  --vib_warmup_ratio "$VIB_WARMUP_RATIO" \
  --recycle_encoder_layer -1 \
  --recycle_encoder_layers "" \
  --recycle_patch_indices "" \
  --recycle_round_alphas ""
echo "=== Done (2/2) VIB: metrics -> $OUT_DIR/vib_test_metrics.json ==="

echo "=== Test metrics ==="
"$PYTHON" - "$OUT_DIR" << 'PY'
import json
import pathlib
import sys

d = pathlib.Path(sys.argv[1])
for label, name in [("baseline", "baseline_test_metrics.json"), ("vib", "vib_test_metrics.json")]:
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

echo "JSON: $OUT_DIR/baseline_test_metrics.json , $OUT_DIR/vib_test_metrics.json"
