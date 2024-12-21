python  -m src.train.finetune \
  --output_dir 'tmp/output' \
  --model_name_or_path '/Users/luxun/workspace/ai/hf/models/Qwen1.5-0.5B'  \
  --use_lora True \
  --train_data_path  '/Users/luxun/workspace/ai/ms/datasets/code_all' \
  --train_data_format  'arrow' \
  --num_train_epochs  1 \
  --per_device_train_batch_size 16 \
  --bf16 True