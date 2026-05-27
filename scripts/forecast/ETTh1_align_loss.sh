#!/bin/bash

# =============================================================================
# ETTh1 Fine-tuning: Four-Way Alignment Loss Comparison (with Region Pooling)
#
# Exp 1/4 — Baseline:        MSE only, no alignment loss
# Exp 2/4 — Align (KL) W=2:  L = MSE + beta * KL(AvgPool(MI,W=2) || AvgPool(A,W=2))
# Exp 3/4 — Align (LMSE) W=2: L = MSE + beta * MSE(AvgPool(logits,W=2), AvgPool(MI,W=2))
# Exp 4/4 — Align (SoftDTW) W=2: L = MSE + beta * SoftDTW(AvgPool(MI,W=2), AvgPool(attn,W=2))
#
# Region-wise pooling (W=2): 4-hour windows on hourly ETTh1 data → 2-token regions
#                             Allows temporal shifts within each region to go unpenalised.
#
# Usage:
#   bash scripts/forecast/ETTh1_align_loss.sh
# =============================================================================

set -e

# ── Hyperparameters ──────────────────────────────────────────────────────────
model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
data=ETTh1
mi_json="outputs/timer_mi_ksg_pca/Timer_MI_20260516_133108/global_mi_peaks_etth1.json"
RESULT_FILE="result_long_term_forecast.txt"

# ── GPU setup ────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=4

# ── Navigate to project root ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

# ── Activate conda environment ─────────────────────────────────────────────
for conda_sh in ~/anaconda3/etc/profile.d/conda.sh ~/miniconda3/etc/profile.d/conda.sh /opt/anaconda3/etc/profile.d/conda.sh; do
    if [ -f "$conda_sh" ]; then
        source "$conda_sh"
        break
    fi
done

conda activate lstm 2>/dev/null || true

# ── Shared torchrun arguments ────────────────────────────────────────────────
TORCHRUN="torchrun --nproc_per_node=1 --master_port 29501"
COMMON_ARGS=(
  --task_name forecast
  --is_training 1
  --seed 1
  --ckpt_path $ckpt_path
  --root_path ./datasets/
  --data_path $data.csv
  --data $data
  --features M
  --seq_len $seq_len
  --label_len $label_len
  --pred_len $pred_len
  --output_len $output_len
  --e_layers 8
  --factor 3
  --des 'Exp'
  --d_model 1024
  --d_ff 2048
  --batch_size 2048
  --learning_rate 3e-5
  --num_workers 4
  --patch_len $patch_len
  --train_test 1
  --subset_rand_ratio 1
  --itr 1
  --use_ims
  --finetune_epochs 10
)

# ── Helpers ──────────────────────────────────────────────────────────────────
extract_metrics() {
  local start=$1 end=$2
  local mse=$(sed -n "${start},${end}p" "$RESULT_FILE" | grep 'mse:' | tail -1 | sed 's/.*mse:\([0-9.]*\).*/\1/')
  local mae=$(sed -n "${start},${end}p" "$RESULT_FILE" | grep 'mae:' | tail -1 | sed 's/.*mae:\([0-9.]*\).*/\1/')
  echo "$mse $mae"
}

table_line() { printf "| %-11s | %-8s | %-8s |\n" "$1" "$2" "$3"; }
table_rule() { echo "|:-----------|:--------|:--------|"; }


: <<'comment'
# =============================================================================
# Experiment 1/3 — Baseline: MSE only
# =============================================================================
BASE_BEFORE=$(wc -l < "$RESULT_FILE" 2>/dev/null || echo 0)

echo ""
echo "============================================================"
echo "  Experiment 1/3 — Baseline (MSE only, no alignment loss)"
echo "============================================================"

$TORCHRUN run.py \
  "${COMMON_ARGS[@]}" \
  --model_id etth1_baseline \--model $model_name

BASE_AFTER=$(wc -l < "$RESULT_FILE" 2>/dev/null || echo 0)
BASE_START=$((BASE_BEFORE + 1))
BASE_END=$BASE_AFTER

read BASE_MSE BASE_MAE <<< "$(extract_metrics $BASE_START $BASE_END)"
echo "  Baseline → MSE=$BASE_MSE  MAE=$BASE_MAE"

comment

# =============================================================================
# Experiment 2/3 — Alignment Loss (KL mode)
#   L_align = (1/sum(M)) * sum_i M_i * KL(softmax(MI/tau) || A_{i,:})
# =============================================================================
ALIGN_KL_BEFORE=$(wc -l < "$RESULT_FILE" 2>/dev/null || echo 0)

echo ""
echo "============================================================"
echo "  Experiment 2/4 — Alignment Loss (KL mode, W=2 region pooling)"
echo "  L_align = KL(AvgPool(MI,W=2) || AvgPool(A,W=2))"
echo "============================================================"

$TORCHRUN run.py \
  "${COMMON_ARGS[@]}" \
  --model_id etth1_align_kl \
  --model $model_name \ --use_align_loss \
  --align_loss_file $mi_json \
  --align_loss_layers 6,7 \
  --align_loss_weight 0.5 \
  --align_loss_tau 0.5 \
  --align_loss_mode kl \
  --align_pool_size 2

ALIGN_KL_AFTER=$(wc -l < "$RESULT_FILE" 2>/dev/null || echo 0)
ALIGN_KL_START=$((ALIGN_KL_BEFORE + 1))
ALIGN_KL_END=$ALIGN_KL_AFTER

read ALIGN_KL_MSE ALIGN_KL_MAE <<< "$(extract_metrics $ALIGN_KL_START $ALIGN_KL_END)"
echo "  Align-KL  → MSE=$ALIGN_KL_MSE  MAE=$ALIGN_KL_MAE"






# =============================================================================
# Experiment 3/4 — Alignment Loss (Logit MSE mode, W=2 region pooling)
#   L_align = (1/sum(M)) * sum_i M_i * MSE(attn_logit, gamma * P_MI)
# =============================================================================
ALIGN_LMSE_BEFORE=$(wc -l < "$RESULT_FILE" 2>/dev/null || echo 0)

echo ""
echo "============================================================"
echo "  Experiment 3/4 — Alignment Loss (Logit MSE mode, W=2 region pooling)"
echo "  L_align = MSE(AvgPool(logits,W=2), AvgPool(MI,W=2))"
echo "============================================================"

$TORCHRUN run.py \
  "${COMMON_ARGS[@]}" \
  --model_id etth1_align_lmse \
  --model $model_name \
  --use_align_loss \
  --align_loss_file $mi_json \
  --align_loss_layers 0,1,2,3,4,5,6,7 \
  --align_loss_weight 1 \
  --align_loss_tau 0.01 \
  --align_loss_mode logit_mse \
  --align_gamma 0.1 \
  --align_pool_size 2

ALIGN_LMSE_AFTER=$(wc -l < "$RESULT_FILE" 2>/dev/null || echo 0)
ALIGN_LMSE_START=$((ALIGN_LMSE_BEFORE + 1))
ALIGN_LMSE_END=$ALIGN_LMSE_AFTER

read ALIGN_LMSE_MSE ALIGN_LMSE_MAE <<< "$(extract_metrics $ALIGN_LMSE_START $ALIGN_LMSE_END)"
echo "  Align-LMSE→ MSE=$ALIGN_LMSE_MSE  MAE=$ALIGN_LMSE_MAE"


# =============================================================================
# Experiment 4/4 — Alignment Loss (Soft-DTW mode)
#   L_align = (1/sum(M)) * sum_i M_i * SoftDTW(P_MI, mean_h(attn_i))
#   Soft-DTW: elastic temporal alignment via differentiable dynamic programming
#   gamma=0.1: moderately soft alignment (between hard-DTW and MSE)
# =============================================================================
ALIGN_SDTW_BEFORE=$(wc -l < "$RESULT_FILE" 2>/dev/null || echo 0)

echo ""
echo "============================================================"
echo "  Experiment 4/4 — Alignment Loss (Soft-DTW mode, W=2 region pooling)"
echo "  L_align = SoftDTW(AvgPool(MI,W=2), AvgPool(attn,W=2))  gamma=0.1"
echo "============================================================"

$TORCHRUN run.py \
  "${COMMON_ARGS[@]}" \
  --model_id etth1_align_softdtw \
  --model $model_name \
  --use_align_loss \
  --align_loss_file $mi_json \
  --align_loss_layers 0,1,2,3,4,5,6,7 \
  --align_loss_weight 1 \
  --align_loss_mode softdtw \
  --align_dtw_gamma 0.1 \
  --align_dtw_bw -1

ALIGN_SDTW_AFTER=$(wc -l < "$RESULT_FILE" 2>/dev/null || echo 0)
ALIGN_SDTW_START=$((ALIGN_SDTW_BEFORE + 1))
ALIGN_SDTW_END=$ALIGN_SDTW_AFTER

read ALIGN_SDTW_MSE ALIGN_SDTW_MAE <<< "$(extract_metrics $ALIGN_SDTW_START $ALIGN_SDTW_END)"
echo "  Align-SoftDTW → MSE=$ALIGN_SDTW_MSE  MAE=$ALIGN_SDTW_MAE"


# =============================================================================
# Summary Table
# =============================================================================
echo ""
echo "============================================================"
echo "  Four-Way Comparison Summary"
echo "============================================================"
echo "| Model         | MSE      | MAE      |"
table_rule
table_line "Baseline"       "$BASE_MSE"       "$BASE_MAE"
table_line "Align-KL"       "$ALIGN_KL_MSE"   "$ALIGN_KL_MAE"
table_line "Align-LMSE"     "$ALIGN_LMSE_MSE" "$ALIGN_LMSE_MAE"
table_line "Align-SoftDTW"  "$ALIGN_SDTW_MSE" "$ALIGN_SDTW_MAE"

echo ""
printf "%-14s %10s %10s %10s %10s\n" "" "MSE" "MAE" "dMSE" "dMAE"
echo "  -----------------------------------------------------------------"
printf "%-14s %10s %10s %10s %10s\n" "Baseline" "$BASE_MSE" "$BASE_MAE" "—" "—"

for name mse mae in "Align-KL"      "$ALIGN_KL_MSE"   "$ALIGN_KL_MAE" \
                    "Align-LMSE"    "$ALIGN_LMSE_MSE" "$ALIGN_LMSE_MAE" \
                    "Align-SoftDTW" "$ALIGN_SDTW_MSE" "$ALIGN_SDTW_MAE"; do
  d_mse=$(python3 -c "print(round($mse - $BASE_MSE, 6))")
  d_mae=$(python3 -c "print(round($mae - $BASE_MAE, 6))")
  p_mse=$(python3 -c "print(round(($mse - $BASE_MSE) / $BASE_MSE * 100, 2))")
  p_mae=$(python3 -c "print(round(($mae - $BASE_MAE) / $BASE_MAE * 100, 2))")
  printf "%-14s %+10s (%+6s%%) %+10s (%+6s%%)\n" "$name" "$d_mse" "$p_mse%" "$d_mae" "$p_mae%"
done
echo "============================================================"
