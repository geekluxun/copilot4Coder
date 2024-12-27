SFT_RESPONSE_TEMPLATE = " ### Answer: "
SFT_INSTRUCTION_TEMPLATE = " ### Question: "


def formatting_prompts_func(sample):
    output_texts = []
    for i in range(len(sample['prompt'])):
        text = f"{SFT_INSTRUCTION_TEMPLATE} {sample['prompt'][i]}\n {SFT_RESPONSE_TEMPLATE} {sample['completion'][i]}"
        output_texts.append(text)
    return output_texts


def formatting_prompts_alpaca_style(samples):
    prompts = []
    for instruction, input, output in zip(samples["instruction"], samples["input"], samples["output"]):
        if input:
            prompt = f"### Instruction: {instruction}\n### Input: {input}\n### Output: {output}"
        else:
            prompt = f"### Instruction: {instruction}\n### Output: {output}"
        prompts.append(prompt)
    return prompts
