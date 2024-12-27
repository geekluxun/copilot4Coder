from transformers import AutoModelForCausalLM, AutoTokenizer

merged_model_path = "/Users/luxun/workspace/ai/mine/copilot4Coder/src/util/Qwen2.5-0.5B-Instruct-ali-lora-32938"


def generate():
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
    model = AutoModelForCausalLM.from_pretrained(merged_model_path)

    # 测试输入
    input_text = "Write a Python function that calculates the factorial of a number:"

    # 编码输入
    inputs = tokenizer(input_text, return_tensors="pt")

    # 获取 attention_mask
    attention_mask = inputs["attention_mask"]

    # 生成输出
    output = model.generate(
        inputs["input_ids"],
        attention_mask=attention_mask,
        max_length=150,  # 最大生成长度
        num_beams=5,  # Beam search
        temperature=0.7,  # 控制生成的随机性
        top_p=0.9,  # 核采样
        top_k=50,  # 限制采样的候选词
        no_repeat_ngram_size=2,  # 防止重复
        early_stopping=True  # 提前停止
    )

    # 解码输出
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated text:")
    print(generated_text)


if __name__ == "__main__":
    generate()
