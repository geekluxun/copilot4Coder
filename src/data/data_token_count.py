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
    print(f"处理后的样本数: {len(training_data)}")

    # 确保 training_data 是一个列表
    if not isinstance(training_data, list):
        training_data = training_data.tolist()

    # 定义批量大小
    batch_size = 1000
    token_counts = []

    # 分批处理并统计token数
    for i in tqdm(range(0, len(training_data), batch_size), desc="批量统计token数量"):
        batch = training_data[i:i + batch_size]
        encodings = tokenizer(batch, add_special_tokens=False, return_length=True, truncation=False)
        counts = encodings['length']
        token_counts.extend(counts)

    print(f"统计的token数量: {len(token_counts)}")

    # 中文字体支持
    font_path = '/System/Library/Fonts/Hiragino Sans GB.ttc'
    # 创建字体属性
    font_prop = font_manager.FontProperties(fname=font_path)
    # 全局设置 Matplotlib 字体
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans GB']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    # 将 token_counts 转换为 DataFrame
    df = pd.DataFrame(token_counts, columns=['token_count'])

    # 计算统计指标
    stats = df['token_count'].describe()
    print("Token数的统计指标：")
    print(stats)

    # 设置 Seaborn 风格后再绘图
    sns.set(style="whitegrid")

    # 绘制直方图和密度图
    plt.figure(figsize=(10, 6))
    sns.histplot(df['token_count'], bins=50, kde=True, color='skyblue')
    plt.title('训练样本Token数分布', fontsize=16, fontproperties=font_prop)
    plt.xlabel('Token数量', fontsize=14, fontproperties=font_prop)
    plt.ylabel('样本数', fontsize=14, fontproperties=font_prop)
    plt.tight_layout()
    plt.show()

    # 绘制箱线图
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df['token_count'], color='lightgreen')
    plt.title('训练样本Token数箱线图', fontsize=16, fontproperties=font_prop)
    plt.xlabel('Token数量', fontsize=14, fontproperties=font_prop)
    plt.tight_layout()
    plt.show()

    # 绘制累积分布函数（CDF）
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(df['token_count'], color='purple')
    plt.title('训练样本Token数累积分布', fontsize=16, fontproperties=font_prop)
    plt.xlabel('Token数量', fontsize=14, fontproperties=font_prop)
    plt.ylabel('累积概率', fontsize=14, fontproperties=font_prop)
    plt.tight_layout()
    plt.show()

    # 计算95百分位数并过滤数据
    percentile_95 = df['token_count'].quantile(0.95)
    print(f"95百分位数的Token数量: {percentile_95}")

    filtered_data = [sample for sample, count in zip(training_data, token_counts) if count <= percentile_95]

    print(f"原始样本数: {len(training_data)}")
    print(f"过滤后样本数: {len(filtered_data)}")


if __name__ == "__main__":
    main()
