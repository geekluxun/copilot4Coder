import evaluate
import numpy as np
import torch
from transformers import AutoTokenizer, TrainerCallback

model_path = "/Users/luxun/workspace/ai/hf/models/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 加载 ROUGE 库
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
accuracy = evaluate.load("accuracy")


# 通过回调方方式实现
class EvaluateCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            perplexity = torch.exp(torch.Tensor([eval_loss]))
            # metrics["eval_perplexity"] = perplexity


def compute_metrics(eval_pred):
    logits, labels, losses = eval_pred.predictions, eval_pred.label_ids, eval_pred.losses
    # 将 logits 和 labels 转换为 torch.Tensor
    logits = torch.tensor(logits, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)

    print(f"logits shape:{logits.size()}, labels shape:{labels.size()}")
    batch_size, sequence_length, vocab_size = logits.size()
    # loss1 = ForCausalLMLoss(logits, labels, vocab_size)

    loss2 = torch.tensor(losses).mean()
    # 计算困惑度
    perplexity = torch.exp(loss2)

    predictions = np.argmax(logits, axis=-1)  # 对于分类任务或生成任务的标签 (logits)
    # 解码预测的token ID为文本
    decoded_predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    decoded_labels = [tokenizer.decode(label[label != -100], skip_special_tokens=True) for label in labels]
    # 计算 ROUGE
    rouge_score = rouge.compute(predictions=decoded_predictions, references=decoded_labels)
    bleu_score = bleu.compute(predictions=decoded_predictions, references=decoded_labels)
    accuracy_metric = accuracy.compute(references=predictions.view(-1), predictions=labels.view(-1))
    metrics = {}
    metrics.update(rouge_score)
    metrics.update({"bleu": bleu_score["bleu"]})
    metrics.update({"perplexity": perplexity.item()})
    metrics.update({"accuracy": accuracy_metric["accuracy"]})

    return metrics


def ForCausalLMLoss(logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss


def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = torch.nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss
