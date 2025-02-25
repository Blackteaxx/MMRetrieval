# %%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname("__file__"), ".."))
sys.path.append(project_root)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# %%
import torch
from src.arguments import DataArguments
from src.dataset.coupert import CoupertDataset
from src.modeling.modeling_clip import CLIPForEmbedding
from transformers import AutoProcessor
from utils.embedder import Embedder

model_dir = "../model/CLIP-ViT-L-14-laion2B-s32B-b82K"
embedding_path = "../embeddings/" + model_dir.split("/")[-1] + ".safetensors"
data_dir = "../data/coupert"

print(torch.cuda.device_count())

# %%
from torch.nn import DataParallel

processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
model = CLIPForEmbedding.from_pretrained(model_dir)
model = DataParallel(model)

train_config = DataArguments(data_dir=data_dir, read_mode="all")

eval_config = DataArguments(data_dir=data_dir, read_mode="all")
gallery_config = DataArguments(data_dir=data_dir, read_mode="all")

train_dataset = CoupertDataset(train_config, mode="train")
eval_dataset = CoupertDataset(eval_config, mode="eval")
gallery_dataset = CoupertDataset(gallery_config, mode="gallery")


# %%
query_instruction_for_retrieval = (
    "Represent this title of product for searching similar products. \n {}"
)


def get_collate_fn(processor):
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        texts = [item["title"] for item in batch]
        global_indices = [item["global_idx"] for item in batch]
        processor.image_processor.do_rescale = False
        processed = processor(
            text=texts,
            images=images,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        outputs = {}

        outputs["text"] = {
            "input_ids": processed["input_ids"],
            "attention_mask": processed["attention_mask"],
        }

        outputs["image"] = {
            "pixel_values": processed["pixel_values"],
        }

        outputs["global_indices"] = torch.tensor(global_indices, dtype=torch.long)
        return outputs

    return collate_fn


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    collate_fn=get_collate_fn(processor),
    num_workers=32,
    pin_memory=True,
)

eval_loader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=256,
    shuffle=False,
    collate_fn=get_collate_fn(processor),
    num_workers=16,
    pin_memory=True,
)

gallery_loader = torch.utils.data.DataLoader(
    gallery_dataset,
    batch_size=256,
    shuffle=False,
    collate_fn=get_collate_fn(processor),
    num_workers=16,
    pin_memory=True,
)


# %%
embedder = Embedder(model, processor=processor, tokenizer=None)
embedder.embed(eval_loader, "eval")
embedder.embed(gallery_loader, "gallery")
embedder.save_embeddings(embedding_path)
