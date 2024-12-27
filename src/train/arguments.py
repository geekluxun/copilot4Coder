from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

from src.common.constant import HubOrigin


@dataclass
class MyModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_lora: bool = field(default=False, metadata={"help": "Enable Lora for faster training."})
    lora_rank: int = field(default=64, metadata={"help": "The rank of lora."})
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})


@dataclass
class MyDataArguments:
    train_data_name_or_path: str = field(default=None, metadata={"help": "Path to the training data."})
    train_data_percentage: Optional[float] = field(
        default=1.0,
        metadata={"help": "The percentage of data to use. If < 1.0, will use a subset of the data."}
    )
    load_local_dataset: bool = field(default=True, metadata={"help": "Whether to load local dataset."})
    test_datasets_percentage: Optional[float] = field(default=0.1,
                                                      metadata={"help": "The percentage of test data ."})
    train_data_format: str = field(default='json', metadata={"help": "The format of the data."})
    num_train_samples: int = field(default=None, metadata={"help": "The number of training samples."})
    num_val_samples: int = field(default=None, metadata={"help": "The number of validation samples."})


@dataclass
class MyTrainingArguments(TrainingArguments):
    max_seq_length: int = field(default=512, metadata={"help": "The maximum sequence length."})
    deepspeed: str = field(default=None, metadata={"help": "The deepspeed config file."})
    use_hp_search: bool = field(default=False, metadata={"help": "Enable  hyperparameter search"})
    hp_search_backend: str = field(default='optuna',
                                   metadata={"help": "The hyperparameter search backend."})
    hp_search_trails: int = field(default=10, metadata={"help": "The number of trails for hyperparameter search."})
    run_name: str = field(default='copilot4Coder', metadata={"help": "The name of the run."})
    eval_by_other_metric: bool = field(default=False, metadata={"help": "Whether to evaluate by other  metric."})
    hub_origin: str = field(default=HubOrigin.HF.value, metadata={"help": "The organization of the hub."})


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
