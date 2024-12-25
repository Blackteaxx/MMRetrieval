import logging

from transformers import CLIPProcessor, HfArgumentParser

from train.arguments import DataArguments, MMTrainArguments, ModelArguments
from train.dataset import ShopeeDataset, get_collate_fn
from train.modeling_clip import CLIPForFusion
from train.trainer import MMTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, MMTrainArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: MMTrainArguments

    processor = CLIPProcessor.from_pretrained(model_args.model_name_or_path)
    model = CLIPForFusion.from_pretrained(model_args.model_name_or_path)

    # frozen the clip model
    for param in model.clip_model.vision_model.parameters():
        param.requires_grad = True

    for param in model.clip_model.text_model.parameters():
        param.requires_grad = True

    dataset = ShopeeDataset(data_args)
    collate_fn = get_collate_fn(processor)

    trainer = MMTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset,
    )

    trainer.train()
