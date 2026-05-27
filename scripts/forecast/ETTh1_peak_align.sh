#!/bin/sh

# ============================================================================
# ETTh1 实验脚本：Baseline vs 层级对齐损失改进对比
# ============================================================================

# 公共参数
model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
data=ETTh1
e_layers=8
d_model=1024
d_ff=2048
batch_size=2048
learning_rate=3e-5

# MI分布文件路径（根据实际生成的路径修改）
MI_DIST_FILE="./results/mi_hsic_etth1_decoder/global_mi_peaks_etth1.json"

# 应用对齐损失的层（Encoder全部8层）
ATTN_MI_LAYERS="0,1,2,3,4,5,6,7"

# MI权重幂次：控制高MI token的关注强度
# power=1.0: 线性权重（与p_mi成正比）
# power=2.0: 二次权重（高MI token权重平方，增强关注）
# power越大，高MI的token越被强调
ATTN_MI_WEIGHT_POWER="1.0"

# 结果文件路径
RESULTS_FILE="./experiments/comparison_results.txt"

# 确保目录存在
mkdir -p "$(dirname "$RESULTS_FILE")"

# 记录可学习参数值
FINAL_ATTN_WEIGHT=""

for subset_rand_ratio in 1
do
    # ========================================================================
    # 实验1: Baseline（无层级对齐损失）
    # ========================================================================
    echo "================================================================"
    echo "实验1: Baseline - 无层级对齐损失"
    echo "================================================================"

    export CUDA_VISIBLE_DEVICES=4,5,6,7

    # 运行训练并捕获输出
    BASELINE_OUTPUT=$(torchrun --nnodes=1 --nproc_per_node=4 run.py \
      --task_name forecast \
      --is_training 0 \
      --seed 1 \
      --ckpt_path $ckpt_path \
      --root_path ./datasets/ \
      --data_path $data.csv \
      --data $data \
      --model_id etth1_baseline_sr_$subset_rand_ratio \
      --model $model_name \
      --features M \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --output_len $output_len \
      --e_layers $e_layers \
      --factor 3 \
      --des 'Baseline' \
      --d_model $d_model \
      --d_ff $d_ff \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --num_workers 4 \
      --patch_len $patch_len \
      --train_test 0 \
      --subset_rand_ratio $subset_rand_ratio \
      --itr 1 \
      --use_ims \
      --use_multi_gpu 2>&1)

    # 提取 Baseline MSE
    BASELINE_MSE=$(echo "$BASELINE_OUTPUT" | grep -oP 'mse:\K[0-9.]+' | tail -1)

    echo "Baseline MSE: $BASELINE_MSE"

    # ========================================================================
    # 实验2: 层级对齐损失改进（可学习权重 lambda=1）
    # ========================================================================
    echo "================================================================"
    echo "实验2: 层级对齐损失改进 - 可学习权重"
    echo "================================================================"

    export CUDA_VISIBLE_DEVICES=4,5,6,7

    # 运行训练并捕获输出
    PEAKALIGN_OUTPUT=$(torchrun --nnodes=1 --nproc_per_node=4 run.py \
      --task_name forecast \
      --is_training 0 \
      --seed 1 \
      --ckpt_path $ckpt_path \
      --root_path ./datasets/ \
      --data_path $data.csv \
      --data $data \
      --model_id etth1_peak_align_sr_$subset_rand_ratio \
      --model $model_name \
      --features M \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --output_len $output_len \
      --e_layers $e_layers \
      --factor 3 \
      --des 'PeakAlign' \
      --d_model $d_model \
      --d_ff $d_ff \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --num_workers 4 \
      --patch_len $patch_len \
      --train_test 0 \
      --subset_rand_ratio $subset_rand_ratio \
      --itr 1 \
      --use_ims \
      --use_multi_gpu \
      --use_attn_mi_loss \
      --use_learnable_attn_mi_weight \
      --attn_mi_loss_weight 1.0 \
      --attn_mi_loss_layers $ATTN_MI_LAYERS \
      --attn_mi_file $MI_DIST_FILE \
      --attn_mi_weight_power $ATTN_MI_WEIGHT_POWER 2>&1)

    # 提取 PeakAlign MSE
    PEAKALIGN_MSE=$(echo "$PEAKALIGN_OUTPUT" | grep -oP 'mse:\K[0-9.]+' | tail -1)

    # 提取最终可学习权重值
    FINAL_ATTN_WEIGHT=$(echo "$PEAKALIGN_OUTPUT" | grep -oP '\[Final\]\s*Learnable attn_mi_weight = \K[0-9.]+' | tail -1)

    echo "PeakAlign MSE: $PEAKALIGN_MSE"

    # ========================================================================
    # 打印对比结果
    # ========================================================================
    echo ""
    echo "================================================================"
    echo "                         实验对比结果"
    echo "================================================================"
    echo "Baseline MSE:      $BASELINE_MSE"
    echo "PeakAlign MSE:    $PEAKALIGN_MSE"

    # 计算差值
    if [ -n "$BASELINE_MSE" ] && [ -n "$PEAKALIGN_MSE" ]; then
        DELTA=$(echo "$BASELINE_MSE - $PEAKALIGN_MSE" | bc -l)
        IMPROVEMENT=$(echo "scale=4; ($DELTA / $BASELINE_MSE) * 100" | bc -l)
        echo "MSE 差值:         $DELTA"
        echo "相对提升:          ${IMPROVEMENT}%"
    fi

    if [ -n "$FINAL_ATTN_WEIGHT" ]; then
        echo "最终可学习权重 λ:  $FINAL_ATTN_WEIGHT"
    fi
    echo "================================================================"

    # 保存结果到文件
    echo "========================================" >> "$RESULTS_FILE"
    echo "Date: $(date)" >> "$RESULTS_FILE"
    echo "Baseline MSE: $BASELINE_MSE" >> "$RESULTS_FILE"
    echo "PeakAlign MSE: $PEAKALIGN_MSE" >> "$RESULTS_FILE"
    if [ -n "$DELTA" ]; then
        echo "MSE 差值: $DELTA" >> "$RESULTS_FILE"
        echo "相对提升: ${IMPROVEMENT}%" >> "$RESULTS_FILE"
    fi
    if [ -n "$FINAL_ATTN_WEIGHT" ]; then
        echo "最终可学习权重 λ: $FINAL_ATTN_WEIGHT" >> "$RESULTS_FILE"
    fi
    echo "" >> "$RESULTS_FILE"

done
