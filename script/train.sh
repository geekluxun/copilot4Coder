python  -m src.train.finetune \
  --output_dir 'tmp/output' \
  --model_name_or_path '/mnt/workspace/model/Qwen2.5-0.5B-Instruct'  \
  --use_lora True \
  --train_data_path  '/mnt/workspace/data/code_all' \
  --train_data_format  'arrow' \
  --num_train_epochs  1 \
  --per_device_train_batch_size 16 \
  --bf16 True