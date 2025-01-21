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
from src.modeling.marqo_fashionSigLIP import MarqoFashionSigLIPForEmbedding
from transformers import AutoProcessor
from utils.embedder import Embedder

model_dir = "../model/Marqo/marqo-ecommerce-embeddings-L"
embedding_path = "../embeddings/" + model_dir.split("/")[-1] + ".safetensors"
data_dir = "../data/coupert"

print(torch.cuda.device_count())

# %%
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
model = MarqoFashionSigLIPForEmbedding.from_pretrained(model_dir)

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
        processed["global_indices"] = torch.tensor(global_indices, dtype=torch.long)
        return processed

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
    batch_size=128,
    shuffle=False,
    collate_fn=get_collate_fn(processor),
    num_workers=16,
    pin_memory=True,
)

gallery_loader = torch.utils.data.DataLoader(
    gallery_dataset,
    batch_size=128,
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
