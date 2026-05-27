#!/bin/bash

# ============================================================
# ETTh1 特征注入（Feature Injection）实验启动脚本
#
# 功能：对比标准 Timer 与 启用特征注入的 Timer 预测性能
# 特征注入：将高 MI patch 的特征向量注入低 MI patch
# ============================================================

# --- 默认参数 ---
ROOT_PATH="./datasets/"
DATA_PATH="ETTm1.csv"
DATA="ETTm1"
CKPT_PATH="checkpoints/Timer_forecast_1.0.ckpt"
SEQ_LEN=672
LABEL_LEN=576
PRED_LEN=96
PATCH_LEN=96
STRIDE=96
OUTPUT_DIR="./results/feature_injection_exp/"

# --- GPU 配置 ---
# 使用 GPU 4, 5, 6, 7（共 4 张卡）
GPU_IDS="4,5,6,7"
GPUS=4

# --- 用户可修改参数 ---
# INJECTION_ALPHA: 特征注入系数，默认 0.5
INJECTION_ALPHA=${INJECTION_ALPHA:-0.5}
# HSIC_FILE: etth1_mi_hsic_peaks 输出的 JSON 文件路径
HSIC_FILE=${HSIC_FILE:-"results/etth1_mi_hsic_peaks/hsic_mi_curves.json"}
# HSIC_LAYER: 使用哪一层的 HSIC MI（-1=最后一层）
HSIC_LAYER=${HSIC_LAYER:-"-1"}

# --- 实验类型选择 ---
#   0: 运行全部（基线 + 特征注入）
#   1: 仅基线
#   2: 仅特征注入
EXP_MODE=${EXP_MODE:-0}

# ============================================================
# 辅助函数
# ============================================================

run_baseline() {
    echo ""
    echo ">>> [1/2] 运行基线实验（标准 Timer）"
    echo ">>>"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" torchrun \
        --nnodes=1 --nproc_per_node=$GPUS \
        experiments/etth1_feature_injection.py \
        --ckpt_path "$CKPT_PATH" \
        --root_path "$ROOT_PATH" \
        --data_path "$DATA_PATH" \
        --data "$DATA" \
        --seq_len "$SEQ_LEN" \
        --label_len "$LABEL_LEN" \
        --pred_len "$PRED_LEN" \
        --patch_len "$PATCH_LEN" \
        --stride "$STRIDE" \
        --output_dir "$OUTPUT_DIR" \
        --use_ims \
        --skip_injection \
        --use_multi_gpu
}

run_injection() {
    echo ""
    echo ">>> [2/2] 运行特征注入实验（alpha=$INJECTION_ALPHA）"
    echo ">>>"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" torchrun \
        --nnodes=1 --nproc_per_node=$GPUS \
        experiments/etth1_feature_injection.py \
        --ckpt_path "$CKPT_PATH" \
        --root_path "$ROOT_PATH" \
        --data_path "$DATA_PATH" \
        --data "$DATA" \
        --seq_len "$SEQ_LEN" \
        --label_len "$LABEL_LEN" \
        --pred_len "$PRED_LEN" \
        --patch_len "$PATCH_LEN" \
        --stride "$STRIDE" \
        --output_dir "$OUTPUT_DIR" \
        --use_ims \
        --skip_baseline \
        --alpha "$INJECTION_ALPHA" \
        --hsic_file "$HSIC_FILE" \
        --hsic_layer "$HSIC_LAYER" \
        --use_multi_gpu
}

# ============================================================
# 主逻辑
# ============================================================

# 临时文件存储各实验结果
BASELINE_LOG="$OUTPUT_DIR/baseline_run.log"
INJECTION_LOG="$OUTPUT_DIR/injection_run.log"

# 清空旧日志
> "$BASELINE_LOG"
> "$INJECTION_LOG"

mkdir -p "$OUTPUT_DIR"

case "$EXP_MODE" in
    0)
        echo ">>> [1/2] 运行基线实验（标准 Timer）"
        CUDA_VISIBLE_DEVICES="$GPU_IDS" torchrun \
            --nnodes=1 --nproc_per_node=$GPUS \
            experiments/etth1_feature_injection.py \
            --ckpt_path "$CKPT_PATH" \
            --root_path "$ROOT_PATH" \
            --data_path "$DATA_PATH" \
            --data "$DATA" \
            --seq_len "$SEQ_LEN" \
            --label_len "$LABEL_LEN" \
            --pred_len "$PRED_LEN" \
            --patch_len "$PATCH_LEN" \
            --stride "$STRIDE" \
            --output_dir "$OUTPUT_DIR" \
            --skip_injection \
            --use_multi_gpu 2>&1 | tee "$BASELINE_LOG"

        echo ""
        echo ">>> [2/2] 运行特征注入实验（alpha=$INJECTION_ALPHA）"
        CUDA_VISIBLE_DEVICES="$GPU_IDS" torchrun \
            --nnodes=1 --nproc_per_node=$GPUS \
            experiments/etth1_feature_injection.py \
            --ckpt_path "$CKPT_PATH" \
            --root_path "$ROOT_PATH" \
            --data_path "$DATA_PATH" \
            --data "$DATA" \
            --seq_len "$SEQ_LEN" \
            --label_len "$LABEL_LEN" \
            --pred_len "$PRED_LEN" \
            --patch_len "$PATCH_LEN" \
            --stride "$STRIDE" \
            --skip_baseline \
            --alpha "$INJECTION_ALPHA" \
            --hsic_file "$HSIC_FILE" \
            --hsic_layer "$HSIC_LAYER" \
            --use_multi_gpu 2>&1 | tee "$INJECTION_LOG"

        # 从日志中提取指标
        extract_metric() {
            local log_file=$1
            local metric_name=$2
            grep "^  ${metric_name}:" "$log_file" | awk '{print $2}' | head -1
        }

        # 读取 baseline 指标
        BL_MSE=$(extract_metric "$BASELINE_LOG" "MSE")
        BL_MAE=$(extract_metric "$BASELINE_LOG" "MAE")
        BL_RMSE=$(extract_metric "$BASELINE_LOG" "RMSE")

        # 读取 injection 指标
        INJ_MSE=$(extract_metric "$INJECTION_LOG" "MSE")
        INJ_MAE=$(extract_metric "$INJECTION_LOG" "MAE")
        INJ_RMSE=$(extract_metric "$INJECTION_LOG" "RMSE")

        # 计算差值和改进率
        calc_delta_pct() {
            local base=$1
            local inj=$2
            echo "$base $inj" | awk '{
                delta = $2 - $1
                pct = (delta / $1) * 100
                printf "%.6f %.6f %+.6f %+.2f", $1, $2, delta, pct
            }'
        }

        read -r MSE_B MSE_I MSE_D MSE_P <<< "$(calc_delta_pct "$BL_MSE" "$INJ_MSE")"
        read -r MAE_B MAE_I MAE_D MAE_P <<< "$(calc_delta_pct "$BL_MAE" "$INJ_MAE")"
        read -r RMSE_B RMSE_I RMSE_D RMSE_P <<< "$(calc_delta_pct "$BL_RMSE" "$INJ_RMSE")"

        echo ""
        echo "================================================================================"
        echo "│                     Baseline vs Feature Injection 对比分析                    │"
        echo "================================================================================"
        printf "%-10s %15s %18s %15s %12s\n" "指标" "Baseline" "Feature Injection" "差值 (Δ)" "改进率"
        echo "--------------------------------------------------------------------------------"
        printf "%-10s %15.6f %18.6f %15.6f %11.2f%% ↓\n" "MSE" "$MSE_B" "$MSE_I" "$MSE_D" "$MSE_P"
        printf "%-10s %15.6f %18.6f %15.6f %11.2f%% ↓\n" "MAE" "$MAE_B" "$MAE_I" "$MAE_D" "$MAE_P"
        printf "%-10s %15.6f %18.6f %15.6f %11.2f%% ↓\n" "RMSE" "$RMSE_B" "$RMSE_I" "$RMSE_D" "$RMSE_P"
        echo "================================================================================"

        echo ""
        echo "📊 实验总结:"
        # 使用 awk 替代 bc 进行浮点数比较（macOS 兼容性）
        if awk "BEGIN {exit !($MSE_P < 0)}"; then
            echo "  ✅ MSE 降低 $(awk "BEGIN {printf \"%.2f\", -($MSE_P)}")%（预测精度提升）"
        else
            echo "  ⚠️  MSE 上升 $(awk "BEGIN {printf \"%.2f\", $MSE_P}")%（预测精度下降）"
        fi
        if awk "BEGIN {exit !($MAE_P < 0)}"; then
            echo "  ✅ MAE 降低 $(awk "BEGIN {printf \"%.2f\", -($MAE_P)}")%（预测误差减少）"
        else
            echo "  ⚠️  MAE 上升 $(awk "BEGIN {printf \"%.2f\", $MAE_P}")%（预测误差增加）"
        fi

        # 保存结果到文件
        cat > "$OUTPUT_DIR/comparison_results.txt" << EOF
================================================================================
        Feature Injection 对比实验结果 (Baseline vs Feature Injection)
================================================================================

【各实验详细指标】

实验: baseline
  MSE: $BL_MSE
  MAE: $BL_MAE
  RMSE: $BL_RMSE

实验: injection
  MSE: $INJ_MSE
  MAE: $INJ_MAE
  RMSE: $INJ_RMSE

================================================================================
【差值分析 (Delta = Injection - Baseline)】
================================================================================
指标              Baseline   Feature Injection       差值 (Δ)        改进率
--------------------------------------------------------------------------------
MSE              $MSE_B          $MSE_I         $MSE_D $MSE_P%
MAE              $MAE_B          $MAE_I         $MAE_D $MAE_P%
RMSE             $RMSE_B          $RMSE_I         $RMSE_D $RMSE_P%
================================================================================

总结: MSE 改进率 = $MSE_P%, MAE 改进率 = $MAE_P%
EOF


        if awk "BEGIN {exit !($MSE_P < 0 && $MAE_P < 0)}"; then
            echo "结论: Feature Injection 有效提升了预测精度" >> "$OUTPUT_DIR/comparison_results.txt"
        elif awk "BEGIN {exit !($MSE_P > 0 || $MAE_P > 0)}"; then
            echo "结论: Feature Injection 导致预测精度下降" >> "$OUTPUT_DIR/comparison_results.txt"
        else
            echo "结论: Feature Injection 对预测精度无明显影响" >> "$OUTPUT_DIR/comparison_results.txt"
        fi
        ;;
    1)
        run_baseline
        ;;
    2)
        run_injection
        ;;
    *)
        echo "[Error] EXP_MODE 必须是 0、1 或 2"
        exit 1
        ;;
esac

echo ""
echo "📁 详细结果已保存到: $OUTPUT_DIR/comparison_results.txt"
echo ""
