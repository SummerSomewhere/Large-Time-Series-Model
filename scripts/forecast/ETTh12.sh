#!/bin/sh
# Run from repo root: bash scripts/forecast/ETTh12.sh

# Physical GPUs 0,2,3,4,5 — processes see them as cuda:0..cuda:4 (five ranks).
export CUDA_VISIBLE_DEVICES=0,2,3,4,5

# Finetune writes under checkpoints/<setting>/: checkpoint.pth (flat state_dict) and
# checkpoint.ckpt ({"state_dict": ...}) for loaders that expect Lightning-style .ckpt.
#
# After testing, optional attention figures (skip with SKIP_ATTN_VIZ=1):
#   - Per layer: attention_maps_etth12/<run_id>/layer_XX/head_YY.png (each head heatmap)
#   - Per layer grid: layer_XX_all_heads.png
#   - Summary 8x8: all_layers_all_heads_summary.png (MSE/MAE 来自本次 torchrun test 写入的 JSON，与热力图同源 test 集)
# Override: ATTN_OUT=./my_maps ATTN_VIZ_GPU=2 ./scripts/forecast/ETTh12.sh  (physical GPU id for viz only)

model_name=Timer
seq_len=96
label_len=48
pred_len=48
output_len=48
patch_len=1
ckpt_path=checkpoints/forecast_etth1_sr_1_Timer_ETTh1_ftM_sl96_ll48_pl48_pl1_dm1024_nh8_el8_dl1_df2048_fc3_ebtimeF_dtTrue_Exp26-03-27_11-39-13/checkpoint.pth
data=ETTh1

for subset_rand_ratio in  1
do
  ATTN_OUT="${ATTN_OUT:-./attention_maps_etth12}"
  mkdir -p "$ATTN_OUT"
  # Rank 0 writes test MSE/MAE here after exp.test; visualize reads the same file for figure captions.
  TEST_METRICS_JSON="${TEST_METRICS_JSON:-$ATTN_OUT/forecast_test_metrics.json}"
  export FORECAST_TEST_METRICS_JSON="$TEST_METRICS_JSON"

torchrun --nnodes=1 --nproc_per_node=5 run.py \
  --task_name forecast \
  --is_finetuning 0 \
  --is_training 0 \
  --seed 1 \
  --ckpt_path "$ckpt_path" \
  --root_path ./datasets/ETT-small/ \
  --data_path "${data}.csv" \
  --data "$data" \
  --model_id "etth1_sr_${subset_rand_ratio}" \
  --model $model_name \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --e_layers 8 \
  --factor 3 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 1024 \
  --learning_rate 3e-5 \
  --num_workers 4 \
  --patch_len $patch_len \
  --train_test 0 \
  --subset_rand_ratio $subset_rand_ratio \
  --itr 1 \
  --gpu 0 \
  --use_ims \
  --use_multi_gpu

  # Same ETTh1 test split / IMS / hparams as run.py; batch_index=0 = first batch of test DataLoader (shuffle off).
  if [ -z "$SKIP_ATTN_VIZ" ]; then
    _def_rid="etth12_sr_${subset_rand_ratio}"
    ATTN_RUN_ID="${ATTN_RUN_ID:-$_def_rid}"
    echo "=== Attention heatmaps -> ${ATTN_OUT}/${ATTN_RUN_ID}/ (per-head PNGs + summary) ==="
    _viz_dev="${CUDA_VISIBLE_DEVICES%%,*}"
    if [ -z "$_viz_dev" ]; then _viz_dev=0; fi
    CUDA_VISIBLE_DEVICES="${ATTN_VIZ_GPU:-$_viz_dev}" python visualize_timer_attention.py \
      --ckpt_path "$ckpt_path" \
      --root_path ./datasets/ETT-small/ \
      --data_path "${data}.csv" \
      --data "$data" \
      --seq_len "$seq_len" \
      --label_len "$label_len" \
      --pred_len "$pred_len" \
      --output_len "$output_len" \
      --patch_len "$patch_len" \
      --d_model 1024 \
      --d_ff 2048 \
      --e_layers 8 \
      --n_heads 8 \
      --factor 3 \
      --output_dir "$ATTN_OUT" \
      --run_id "$ATTN_RUN_ID" \
      --gpu 0 \
      --batch_index "${ATTN_BATCH_INDEX:-0}" \
      --test_metrics_json "$TEST_METRICS_JSON"
  fi
done
