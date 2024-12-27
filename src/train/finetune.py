import argparse
import sys
from typing import Any, Dict

import transformers
from peft import LoraConfig, get_peft_model
from torch.multiprocessing import freeze_support
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer

from src.common.constant import HubOrigin
from src.data.data_templete import formatting_prompts_alpaca_style
from src.data.data_load import load_train_data
# from src.eval.eval import compute_metrics, EvaluateCallback
from src.monitor.monitor import init_wandb
from src.train.arguments import print_args, MyTrainingArguments, MyModelArguments, MyDataArguments
from src.train.hp_tune import hp_space
from src.util.device_util import get_train_device


# local


# 可以预处理logits，部分逻辑可以compute_metrics函数中移到这里
def preprocess_logits_for_metrics(logits, labels):
    pass


def train(model_args, data_args, training_args):
    # 加载tokenizer和数据集
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    def preprocess_data_function(examples):
        prompts = formatting_prompts_alpaca_style(examples)
        # for instruction, completion in zip(examples["prompt"], examples["completion"]):
        #     prompt = f"### Instruction: {instruction}\n### Completion: {completion}"
        #     prompts.append(prompt)

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

        # 将处理后的 labels 添加到 model_inputs
        model_inputs["labels"] = labels

        return model_inputs

    # 加载数据集
    dataset = load_train_data(data_args)
    train_dataset = dataset["train"].map(preprocess_data_function, batched=True)
    val_dataset = dataset["test"].map(preprocess_data_function, batched=True)
    data_args.num_train_samples = len(train_dataset)
    data_args.num_val_samples = len(val_dataset)

    # 创建DataCollator
    # collator = DataCollatorForCompletionOnlyLM(response_template=SFT_RESPONSE_TEMPLATE, tokenizer=tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    def model_init():
        mode_kwargs = _get_mode_kwargs(model_args)
        device = get_train_device()
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
        return model

    # 默认使用eval loss作为指标，如果需要其他指标，可以使用compute_objective参数
    if training_args.use_hp_search:
        print("Starting hyperparameter_search training...")
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        best_run = trainer.hyperparameter_search(
            direction="minimize",
            hp_space=hp_space(backend=training_args.hp_search_backend),
            backend=training_args.hp_search_backend,
            n_trials=training_args.hp_search_trails,
        )
        # 输出最佳超参数
        print("Best hyperparameters:", best_run.hyperparameters)
    else:
        model = model_init()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            # compute_metrics=compute_metrics,
            # callbacks=[EvaluateCallback()]
        )
        # if training_args.eval_by_other_metric:
        #    trainer.compute_metrics = compute_metrics

        print("Starting  training...")
        _log_all_training_params(model, training_args, data_args)
        # trainer.evaluate()
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


def _optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-4),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [4, 8, 16, 24]),
    }


def _get_mode_kwargs(model_args: "MyModelArguments") -> Dict[str, Any]:
    init_kwargs = {}
    # if training_args.bf16:
    #     init_kwargs["torch_dtype"] = torch.bfloat16
    # elif training_args.fp16:
    #     init_kwargs["torch_dtype"] = torch.float16
    init_kwargs["trust_remote_code"] = True
    return init_kwargs


def _log_all_training_params(model, training_args: "MyTrainingArguments", data_args: "MyDataArguments"):
    init_wandb(training_args)
    print_args(model.config, '模型参数汇总')
    print_args(data_args, '数据参数汇总')
    print_args(training_args, '训练参数汇总')

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / total_params * 100
    # 计算每个epoch的步骤数
    steps_per_epoch = data_args.num_train_samples // training_args.per_device_train_batch_size
    # 计算每个epoch的有效步数（考虑梯度累积）
    effective_steps_per_epoch = steps_per_epoch // training_args.gradient_accumulation_steps
    # 计算总训练步数（考虑梯度累积）
    total_training_steps = effective_steps_per_epoch * training_args.num_train_epochs
    eval_count_per_epoch = effective_steps_per_epoch // training_args.eval_steps
    effective_train_samples_per_step = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    train_samples_per_eval = effective_train_samples_per_step * training_args.eval_steps

    print(f"总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    print(f"可训练参数占比: {trainable_ratio:.2f}%")
    print(f"训练样本总数: {data_args.num_train_samples}")
    print(f"验证样本总数: {data_args.num_val_samples}")
    print(f"每step有效batch大小:{effective_train_samples_per_step}")
    print(f"每个epoch的有效训练步数（考虑梯度累积）：{effective_steps_per_epoch}")
    print(f"评估步数：{training_args.eval_steps}")
    print(f"每个epoch评估次数： {eval_count_per_epoch}")
    print(f"每训练样本数评估一次:{train_samples_per_eval}")
    print(f"训练epoch数:{training_args.num_train_epochs}")
    print(f"总训练步数（考虑梯度累积）：{total_training_steps}")


# 训练环境
def _init_env(model_args, data_args, training_args):
    # 环境设置
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = "tmp/wandb_log"
    os.environ["WANDB_CACHE_DIR"] = "tmp/wandb_log"
    if training_args.hub_origin == HubOrigin.HF_MIRROR.value:
        os.environ["HF_ENDPOINT"] = "http://hf-mirror.com"


def _get_args():
    argsParser = argparse.ArgumentParser()
    # 添加命令行参数
    argsParser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model")
    argsParser.add_argument("--train_data_name_or_path", type=str, required=True, help="Path to dataset")
    argsParser.add_argument("--json_param_path", type=str, default='script/params.json', required=False, help="Path to json param ")
    args = argsParser.parse_args()

    parser = transformers.HfArgumentParser((MyModelArguments, MyDataArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=args.json_param_path)
    data_args.train_data_name_or_path = args.train_data_name_or_path
    model_args.model_name_or_path = args.model_name_or_path
    return model_args, data_args, training_args


if __name__ == '__main__':
    freeze_support()
    import os

    DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1')
    if DEBUG:
        print("debug mode...")
        sys.argv = ['finetune.py',
                    '--model_name_or_path', '/Users/luxun/workspace/ai/hf/models/Qwen1.5-0.5B',
                    '--train_data_name_or_path', 'sahil2801/CodeAlpaca-20k',
                    '--json_param_path', '/Users/luxun/workspace/ai/mine/copilot4Coder/script/params.json'
                    ]
    model_args, data_args, training_args = _get_args()
    _init_env(model_args, data_args, training_args)
    train(model_args, data_args, training_args)
