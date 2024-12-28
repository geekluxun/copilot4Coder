SFT_RESPONSE_TEMPLATE = " ### Answer: "
SFT_INSTRUCTION_TEMPLATE = " ### Question: "


def formatting_prompts_func(sample):
    output_texts = []
    for i in range(len(sample['prompt'])):
        text = f"{SFT_INSTRUCTION_TEMPLATE} {sample['prompt'][i]}\n {SFT_RESPONSE_TEMPLATE} {sample['completion'][i]}"
        output_texts.append(text)
    return output_texts


def formatting_prompts_alpaca_style(samples, data_columns):
    # 检查 samples 是否包含 instruction 和 output 字段
    required_keys = [data_columns["instruction"], data_columns["output"]]
    for key in required_keys:
        if key not in samples:
            raise ValueError(f"Missing required column in samples: {key}")
    prompts = []
    for instruction, input, output in zip(
            samples.get(data_columns["instruction"], []),
            samples.get(data_columns["input"], [""] * len(samples.get(data_columns["instruction"], []))),
            samples.get(data_columns["output"], [])
    ):
        if input:
            prompt = f"### Instruction: {instruction}\n### Input: {input}\n### Output: {output}"
        else:
            prompt = f"### Instruction: {instruction}\n### Output: {output}"
        prompts.append(prompt)

    return prompts
