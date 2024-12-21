python  -m src.train.finetune \
  --output_dir 'tmp/output' \
  --model_name_or_path '/Users/luxun/workspace/ai/ms/models/Qwen2.5-0.5B-Instruct'  \
  --use_lora True \
  --train_data_path  '/Users/luxun/workspace/ai/ms/datasets/code_all' \
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
# --resume_from_checkpoint '/Users/luxun/workspace/ai/mine/copilot4Coder/src/train/tmp/output/checkpoint-6'

