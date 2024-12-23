#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用的GPU编号
export MASTER_PORT=29500  # master端口
export MASTER_ADDR=localhost  # master地址
export WORLD_SIZE=1  # 总进程数（GPU数量）
export NODE_RANK=0  # 当前节点的rank
export LOCAL_RANK=0  # 本地rank
#export WANDB_DISABLED=TRUE  # 是否使用wandb
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_ENABLED=0

# 启动分布式训练
PYTHONPATH=$(pwd) deepspeed --num_gpus=1 \
    --master_port=$MASTER_PORT \
    src/train/finetune.py \
    --deepspeed ds_config.json \
    --output_dir 'tmp/output' \
    --model_name_or_path '/Users/luxun/workspace/ai/ms/models/Qwen2.5-0.5B-Instruct'  \
    --train_data_path  '/Users/luxun/workspace/ai/ms/datasets/code_all' \
    --use_lora True \
    --train_data_format  'arrow' \
    --num_train_epochs  1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps  2 \
    --learning_rate '1e-5' \
    --warmup_steps 100 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --report_to tensorboard --report_to wandb  \
    --bf16 True \
    --logging_dir 'tmp/log' \
    --log_level 'debug' \
    --logging_steps  100 \
    --metric_for_best_model 'loss' \
    --max_seq_length  512 \
    --deepspeed script/ds_config.json
