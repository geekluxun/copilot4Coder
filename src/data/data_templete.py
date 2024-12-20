SFT_RESPONSE_TEMPLATE = " ### Answer: "
SFT_INSTRUCTION_TEMPLATE = " ### Question: "


def formatting_prompts_func(sample):
    output_texts = []
    for i in range(len(sample['prompt'])):
        text = f"{SFT_INSTRUCTION_TEMPLATE} {sample['prompt'][i]}\n {SFT_RESPONSE_TEMPLATE} {sample['completion'][i]}"
        output_texts.append(text)
    return output_texts
