from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."}
    )
    use_lora: bool = field(default=False, metadata={"help": "Enable Lora for faster training."})
    hidden_size: int = field(default=2048, metadata={"help": "The hidden size of the model."})
    num_layers: int = field(default=24, metadata={"help": "The number of layers of the model."})
    num_attention_heads: int = field(default=16, metadata={"help": "The number of attention heads of the model."})
    intermediate_size: int = field(default=8192, metadata={"help": "The intermediate size of the model."})
    max_position_embeddings: int = field(
        default=2048,
        metadata={"help": "The maximum sequence length that this model might ever be used with."}
    )
    vocab_size: int = field(default=50257, metadata={"help": "The vocabulary size of the model."})
    type_vocab_size: int = field(default=1, metadata={"help": "The vocabulary size of the model."})
    layer_norm_eps: float = field(
        default=1e-5,
        metadata={"help": "The epsilon used by the layer normalization layers of the model."}
    )
    moe_topk: int = field(default=4, metadata={"help": "The topk for MOE."})
    num_experts: int = field(default=8, metadata={"help": "The number of experts for MOE."})
    num_key_value_heads: int = field(default=16, metadata={"help": "The number of key-value heads in GQA."})
    use_cla: bool = field(default=False, metadata={"help": "Whether to use CLA."})
    cla_share_factor: int = field(default=2, metadata={"help": "The share factor for CLA."})
    use_mixed_mlp_moe: bool = field(
        default=False,
        metadata={"help": "Whether to use mixed MoE with shared expert."}
    )
    num_shared_expert: int = field(default=1, metadata={"help": "Number of shared experts."})
    use_qk_norm: bool = field(default=False, metadata={"help": "Whether to use qk norm."})
    tie_word_embeddings: bool = field(
        default=True,
        metadata={"help": "Whether to tie the word embeddings of the encoder and the decoder."}
    )
    lora_rank: int = field(default=64, metadata={"help": "The rank of lora."})
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    train_attention_params_only: bool = field(default=False, metadata={
        "help": "Whether to train attention parameters only."}
                                              )


@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    train_data_percentage: Optional[float] = field(
        default=0.01,
        metadata={"help": "The percentage of data to use. If < 1.0, will use a subset of the data."}
    )
    load_local_dataset: bool = field(default=True, metadata={"help": "Whether to load local dataset."})
    test_datasets_percentage: Optional[float] = field(default=0.1,
                                                      metadata={"help": "The percentage of test data ."})
    train_data_format: str = field(default='json', metadata={"help": "The format of the data."})


@dataclass
class MyTrainingArguments(TrainingArguments):
    max_seq_length: int = field(default=512, metadata={"help": "The maximum sequence length."})
    deepspeed: str = field(default=None, metadata={"help": "The deepspeed config file."})
    use_hp_search: bool = field(default=False, metadata={"help": "Enable  hyperparameter search"})
    hp_search_backend: str = field(default='optuna',
                                   metadata={"help": "The hyperparameter search backend."})
    hp_search_trails: int = field(default=10, metadata={"help": "The number of trails for hyperparameter search."})


def print_args(args, name='arguments'):
    """Print arguments."""
    # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
    print(f'------------------------ {name} ------------------------', flush=True)
    str_list = []
    for arg in vars(args):
        dots = '.' * (48 - len(arg))
        str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print(f'-------------------- end of {name} ---------------------', flush=True)
