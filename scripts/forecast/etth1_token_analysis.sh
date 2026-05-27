#!/bin/bash

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
data=ETTh1

# Token 分析参数
top_k=2       # 高MI patch数量
bottom_k=2    # 低MI patch数量
num_samples=100  # 采样的样本数量（用于可视化）
max_batches=100 # 最多处理多少个batch用于统计

# 清空结果文件
> result_token_analysis.txt

echo "=============================================="
echo "ETTh1 Token 分析实验"
echo "对比维度: 高 MI Patch vs 低 MI Patch"
echo "=============================================="
torchrun --nnodes=1 --nproc_per_node=1 experiments/etth1_token_analysis.py \
  --task_name forecast \
  --is_training 0 \
  --is_finetuning 0 \
  --seed 1 \
  --ckpt_path $ckpt_path \
  --root_path ./datasets/ \
  --data_path $data.csv \
  --data $data \
  --model_id etth1_token_analysis \
  --model $model_name \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --e_layers 8 \
  --factor 3 \
  --des 'Token_Analysis' \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 2048 \
  --num_workers 4 \
  --patch_len $patch_len \
  --train_test 0 \
  --subset_rand_ratio 1 \
  --itr 1 \
  --gpu 0 \
  --use_ims \
  --top_k $top_k \
  --bottom_k $bottom_k \
  --num_samples $num_samples \
  --max_batches $max_batches

echo ""
echo "分析完成，结果保存在 experiments/output/token_analysis/ 目录"
