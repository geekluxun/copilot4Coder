import os
import shutil

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_path = "/Users/luxun/workspace/ai/ms/models/Qwen2.5-0.5B-Instruct"
lora_checkpoint_path = "/Users/luxun/workspace/ai/mine/copilot4Coder/log/checkpoint/checkpoint-32938"
output_dir = "Qwen2.5-0.5B-Instruct-ali-lora-32938"


def merge_lora_model():
    # 加载基座模型和 LoRA
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    lora_model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # 将 LoRA 参数融入到基座模型中
    merged_model = lora_model.merge_and_unload()

    # 检查并删除 output_dir 目录（如果存在）
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    # 保存合并后的模型
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    merge_lora_model()
