from datasets import load_dataset
from transformers import AutoTokenizer

tokenizers_path = "/Users/luxun/workspace/ai/ms/models/Qwen2.5-0.5B-Instruct"
origin_data_path = "/Users/luxun/workspace/ai/ms/datasets/code_all"
max_token = 128
filterd_data_path = "/Users/luxun/workspace/ai/ms/datasets/code_all_128_filtered"


def filter_dataset_by_token():
    """
        按照token数过滤数据集
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizers_path)
    dataset = load_dataset("arrow", data_dir=origin_data_path)

    def tokenize_function(examples):
        prompts = []
        for instruction, completion in zip(examples["prompt"], examples["completion"]):
            prompt = f"### Instruction: {instruction}\n### Completion: {completion}"
            prompts.append(prompt)
        return tokenizer(prompts, truncation=False, padding=False)

    dataset_tokenized = dataset.map(tokenize_function, batched=True, batch_size=10240, num_proc=10)

    # 只保留小于max_token的样本
    def filter_function(example):
        return len(example["input_ids"]) <= max_token

    # 5. 对整个数据集进行过滤
    dataset_filtered = dataset_tokenized.filter(filter_function)

    # 6. 保存过滤后的数据集到本地
    dataset_filtered.save_to_disk(filterd_data_path)


if __name__ == '__main__':
    filter_dataset_by_token()
