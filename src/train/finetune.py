import os
import sys
from typing import Any, Dict

import torch
import transformers
from peft import LoraConfig, get_peft_model
from torch.multiprocessing import freeze_support
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForSeq2Seq, Trainer
from trl import SFTTrainer

from src.monitor.monitor import init_monitor
from src.train.arguments import ModelArguments, DataArguments, print_args, MyTrainingArguments
from src.data.data_load import load_train_data
from src.util.device_util import get_train_device

# 环境设置
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DIR"] = "tmp/wandb_log"
os.environ["WANDB_CACHE_DIR"] = "tmp/wandb_log"


# local


# 可以预处理logits，部分逻辑可以compute_metrics函数中移到这里
def preprocess_logits_for_metrics(logits, labels):
    pass


def train_model():
    model_args, data_args, training_args = get_args()
    device = get_train_device()
    init_monitor(training_args)
    mode_kwargs = _get_mode_kwargs(model_args, training_args)
    # 加载模型
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **mode_kwargs,
    )
    model.to(device)

    if model_args.use_lora:
        # 定义 Lora 配置
        lora_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # 加载tokenizer和数据集
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    def preprocess_data_function(examples):
        prompts = []
        for instruction, completion in zip(examples["prompt"], examples["completion"]):
            prompt = f"### Instruction: {instruction}\n### Completion: {completion}"
            prompts.append(prompt)

        # 进行分词，并确保返回的是 PyTorch 张量
        model_inputs = tokenizer(
            prompts,
            max_length=training_args.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 克隆 input_ids 以创建 labels
        labels = model_inputs["input_ids"].clone()

        # 将 attention_mask 为 0 的位置（填充部分）设置为 -100
        labels[model_inputs["attention_mask"] == 0] = -100

        # 如果你只希望计算 Completion 部分的损失，而忽略 Instruction 部分，
        # 你需要进一步将 Instruction 部分的 labels 设置为 -100。
        # 假设 "### Completion:" 的开始位置相同，你可以找到该位置并进行掩码。

        # 定义 Completion 的前缀
        # completion_prefix = "### Completion:"
        # completion_token_ids = tokenizer(completion_prefix, add_special_tokens=False)["input_ids"]
        #
        # # 转换为列表以便搜索
        # input_ids_list = labels.tolist()
        # completion_length = len(completion_token_ids)
        #
        # for i, tokens in enumerate(input_ids_list):
        #     try:
        #         # 查找 Completion 前缀的起始位置
        #         prefix_start = tokens.index(completion_token_ids[0])
        #         for j in range(1, completion_length):
        #             if tokens[prefix_start + j] != completion_token_ids[j]:
        #                 break
        #         else:
        #             # Completion 内容的实际开始位置
        #             completion_start = prefix_start + completion_length
        #             # 将 Instruction 部分的 labels 设置为 -100
        #             labels[i, :completion_start] = -100
        #     except ValueError:
        #         # 如果未找到 Completion 前缀，则全部设置为 -100
        #         labels[i] = -100

        # 将处理后的 labels 添加到 model_inputs
        model_inputs["labels"] = labels

        return model_inputs
    # 加载数据集
    dataset = load_train_data(data_args)
    train_dataset = dataset["train"].map(preprocess_data_function, batched=True)
    val_dataset = dataset["test"].map(preprocess_data_function, batched=True)
    print(f"train_dataset size: {len(train_dataset)}, val_dataset size: {len(val_dataset)}")

    # 创建DataCollator
    # collator = DataCollatorForCompletionOnlyLM(response_template=SFT_RESPONSE_TEMPLATE, tokenizer=tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    # 开始训练
    print("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


def _get_mode_kwargs(model_args: "ModelArguments", training_args: "TrainingArguments") -> Dict[str, Any]:
    init_kwargs = {}
    if training_args.bf16:
        init_kwargs["torch_dtype"] = torch.bfloat16
    elif training_args.fp16:
        init_kwargs["torch_dtype"] = torch.float16
    init_kwargs["trust_remote_code"] = True
    return init_kwargs


def get_args():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print_args(model_args, 'model arguments')
    print_args(data_args, 'data arguments')
    print_args(training_args, 'training arguments')
    return model_args, data_args, training_args


if __name__ == '__main__':
    freeze_support()
    # sys.argv = ['finetune.py',
    #             '--output_dir', 'tmp/output',
    #             '--model_name_or_path', '/Users/luxun/workspace/ai/ms/models/Qwen2.5-0.5B-Instruct',
    #             '--use_lora', 'True',
    #             '--train_data_path',
    #             '/Users/luxun/workspace/ai/ms/datasets/code_all',
    #             '--train_data_format', 'arrow',
    #             # '--max_steps', '55',
    #             '--num_train_epochs', '1',
    #             '--per_device_train_batch_size', '16',
    #             '--bf16', 'True',
    #             '--max_seq_length', '512'
    #             ]
    train_model()
