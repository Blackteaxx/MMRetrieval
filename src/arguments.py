from dataclasses import dataclass
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str


@dataclass
class DataArguments:
    data_dir: str
    img_dir: Optional[str] = None
    cache_dir: Optional[str] = None

    read_mode: str = "all"  # Options are "text", "image", "all"


@dataclass
class MMTrainArguments(TrainingArguments):
    temperature: float = 0.05
    loss_type: str = "triplet"
    negatives_cross_batch: bool = False
