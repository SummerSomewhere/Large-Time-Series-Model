#!/bin/sh
# Run Timer attention visualization (visualize_timer_attention.py at repo root).
# Usage: ./visualize_timer_attention.sh
# Default: visible physical GPUs 4 and 5 (--gpu 0 -> first of those, i.e. GPU 4).
# Override devices: CUDA_VISIBLE_DEVICES=0,1 ./visualize_timer_attention.sh
# Default input: --synthetic_sin → batch_x/batch_y built with torch.sin (multivariate); optional noise via --sin_noise_std.
# Real ETTh1 CSV: USE_REAL_DATA=1 ./visualize_timer_attention.sh
# CKPT: absolute or repo-relative path to checkpoint.pth / .ckpt (must exist on disk).
# DDP-trained weights (module.* / backbone.*) are stripped automatically in models/checkpoint_utils.py.
# Override output base: OUT=... ./visualize_timer_attention.sh
# All figures go under OUT/<subfolder>/ only (never mixed into OUT root unless --flat_output).
# Auto subfolder: run_YYYYMMDD_HHMMSS_microseconds. Same RUN_ID twice -> exp1, then exp1_002, ...
# Example: RUN_ID=exp_lr1e4 ./visualize_timer_attention.sh
# Overwrite same folder: add --reuse_exact_run_id
# Flat layout (no subfolder): ./visualize_timer_attention.sh --flat_output
# Extra Python args: ./visualize_timer_attention.sh --batch_index 1 --gpu 0
# Attention: default linear_quantile color stretch (faint stripes visible). Paper-style [0,1]: --attn_scale linear01

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Default physical GPUs 4,5 (script uses single process; cuda:0 maps to the first listed).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

CKPT="${CKPT:-checkpoints/forecast_etth1_sr_1_Timer_ETTh1_ftM_sl96_ll48_pl48_pl1_dm1024_nh8_el8_dl1_df2048_fc3_ebtimeF_dtTrue_Exp26-03-27_11-39-13/checkpoint.pth}"
OUT="${OUT:-./attention_maps_timer}"
DATA_ROOT="${DATA_ROOT:-./datasets/ETT-small/}"

EXTRA_PY=()
if [ -n "$RUN_ID" ]; then
  EXTRA_PY=(--run_id "$RUN_ID")
fi
if [ -z "$USE_REAL_DATA" ]; then
  EXTRA_PY+=(--synthetic_sin)
fi

exec python visualize_timer_attention.py \
  --ckpt_path "$CKPT" \
  --root_path "$DATA_ROOT" \
  --data_path ETTh1.csv \
  --data ETTh1 \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 6 \
  --output_len 6 \
  --patch_len 1 \
  --sin_n_vars 1 \
  --d_model 1024 \
  --d_ff 2048 \
  --e_layers 8 \
  --n_heads 8 \
  --factor 3 \
  --output_dir "$OUT" \
  --gpu 0 \
  "${EXTRA_PY[@]}" \
  "$@"
