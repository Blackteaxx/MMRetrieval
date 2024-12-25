from dataclasses import dataclass

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str


@dataclass
class DataArguments:
    data_dir: str
    img_dir: str
    cache_dir: str


@dataclass
class MMTrainArguments(TrainingArguments):
    temperature: float = 0.05
    loss_type: str = "triplet"
