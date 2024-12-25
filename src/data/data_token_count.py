import random

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from matplotlib import font_manager
from tqdm import tqdm
from transformers import AutoTokenizer

model_path = "/Users/luxun/workspace/ai/ms/models/Qwen2.5-0.5B-Instruct"
data_path = "/Users/luxun/workspace/ai/ms/datasets/code_all"

sample_percentage = 1.0
batch_size = 100000


def main():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = load_dataset('arrow', data_dir=data_path)
    # 检查数据集的列名
    print("数据集列名:", dataset['train'].column_names)

    # 合并 'prompt' 和 'completion' 为一个完整的文本样本
    def merge_prompt_completion(example):
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")
        return {"text": prompt + " " + completion}

    training_dataset = dataset['train'].map(merge_prompt_completion, remove_columns=dataset['train'].column_names)

    # 提取合并后的文本字段，并过滤空文本
    training_data = training_dataset['text']
    training_data = [text for text in training_data if text.strip()]
    training_data = random.sample(training_data, int(len(training_data) * sample_percentage))
    print(f"处理后的样本数: {len(training_data)}")

    token_counts = []
    # 分批处理并统计token数
    for i in tqdm(range(0, len(training_data), batch_size), desc="批量统计token数量"):
        batch = training_data[i:i + batch_size]
        encodings = tokenizer(batch, add_special_tokens=False, return_length=True, truncation=False)
        counts = encodings['length']
        token_counts.extend(counts)

    total_tokens = sum(token_counts)
    print(f"数据集中所有token总数为: {total_tokens},{format_number(total_tokens)}")

    # 中文字体支持
    font_path = '/System/Library/Fonts/Hiragino Sans GB.ttc'
    # 创建字体属性
    font_prop = font_manager.FontProperties(fname=font_path)
    # 全局设置 Matplotlib 字体
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans GB']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 将 token_counts 转换为 DataFrame
    df = pd.DataFrame(token_counts, columns=['token_count'])

    # 均值、中位数、75%分位数、90%分位数、95%分位数和99%分位数
    token_stats = df['token_count'].describe(percentiles=[.25, .50, .75, .90, .95, .99])
    print("样本的Token分布情况：")
    print(token_stats)
    percentile_99_value = token_stats['99%']
    percentile_99_count = token_stats.quantile(0.99)
    print(f"percentile_99_value:{percentile_99_value},percentile_99_count:{percentile_99_count}")

    # 设置 Seaborn 风格后再绘图
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(token_counts, kde=True, stat="count", color="blue")

    plt.title("Token Count Distribution", fontsize=16)
    plt.xlabel("Token Count", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    # 限制限制的横坐标范围（根据样本的百分位值）
    plt.xlim(left=0, right=int(percentile_99_value))
    plt.show()


def format_number(num: int) -> str:
    """
    Convert a number to a human-readable format with statistical units (K, M, B, T).
    Args:
        num (int): The number to format.

    Returns:
        str: The formatted string with units.
    """
    units = ["", "K", "M", "B", "T"]
    for unit in units:
        if num < 1000:
            return f"{num:.2f}{unit}".rstrip("0").rstrip(".")
        num /= 1000
    return f"{num:.2f}P"  # 超过 T 的情况


if __name__ == "__main__":
    main()
