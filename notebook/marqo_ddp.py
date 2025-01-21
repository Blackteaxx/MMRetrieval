import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname("__file__"), ".."))
sys.path.append(project_root)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
from safetensors.torch import save_file

# 你原先项目的 import
from src.arguments import DataArguments
from src.dataset.coupert import CoupertDataset
from src.modeling.marqo_fashionSigLIP import MarqoFashionSigLIPForEmbedding
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor
from utils.embedder import Embedder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, default="../model/Marqo/marqo-ecommerce-embeddings-L"
    )
    parser.add_argument("--data_dir", type=str, default="../data/coupert")
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="../embeddings/marqo-ecommerce-embeddings-L.safetensors",
    )
    parser.add_argument("--batch_size_train", type=int, default=512)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=24)
    return parser.parse_args()


def setup_distributed():
    """
    初始化分布式环境
    在单机多卡的场景下, 使用 torchrun --nproc_per_node=NUM_GPUS ddp_embed.py
    """
    dist.init_process_group(backend="nccl")


def cleanup_distributed():
    """销毁分布式环境"""
    dist.destroy_process_group()


def main():
    args = parse_args()
    setup_distributed()

    # local_rank = 当前进程在本机中的 GPU 编号
    # 如果使用 torchrun, PyTorch 会自动注入 LOCAL_RANK 这个环境变量
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 输出, 仅在 rank=0 时打印
    rank = dist.get_rank()
    if rank == 0:
        print(
            f"Running DDP on rank {rank}, local_rank {local_rank}, world_size={dist.get_world_size()}"
        )

    # 1. 加载模型与处理器
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = MarqoFashionSigLIPForEmbedding.from_pretrained(args.model_dir)
    model.to(device)

    # 2. 使用 DistributedDataParallel 包装
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 3. 准备数据集 & 分布式采样器
    train_config = DataArguments(data_dir=args.data_dir, read_mode="all")
    eval_config = DataArguments(data_dir=args.data_dir, read_mode="all")
    gallery_config = DataArguments(data_dir=args.data_dir, read_mode="all")

    train_dataset = CoupertDataset(train_config, mode="train")
    eval_dataset = CoupertDataset(eval_config, mode="eval")
    gallery_dataset = CoupertDataset(gallery_config, mode="gallery")

    # 分布式采样器: 保证每个进程只处理数据集的一部分, 避免重复
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    gallery_sampler = DistributedSampler(gallery_dataset, shuffle=False)

    # collate_fn
    def get_collate_fn(processor):
        def collate_fn(batch):
            images = [item["image"] for item in batch]
            texts = [item["title"] for item in batch]
            global_indices = [item["global_idx"] for item in batch]
            # 关闭 rescale (按你原先的需求)
            processor.image_processor.do_rescale = False
            processed = processor(
                text=texts,
                images=images,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            processed["global_indices"] = torch.tensor(global_indices)
            return processed

        return collate_fn

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size_train,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=get_collate_fn(processor),
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size_eval,
        sampler=eval_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=get_collate_fn(processor),
    )

    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=args.batch_size_eval,
        sampler=gallery_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=get_collate_fn(processor),
    )

    # 如果你只是想跑 "embedding" 推理, 不一定需要 train_loader.
    # 这里只是示例: 先加载/构造好 train_loader, 也可以做分布式训练.

    # 4. 推理或训练逻辑 (这里只示范 embedding)，并且保存到safetensors文件
    embedder = Embedder(model, processor=processor, tokenizer=None)

    world_size = dist.get_world_size()

    local_eval_embed, local_eval_ids = embedder.embed(eval_loader, "eval")
    local_gallery_embed, local_gallery_ids = embedder.embed(gallery_loader, "gallery")

    del model
    torch.cuda.empty_cache()

    gathered_embeddings = {
        "eval_embs": [
            torch.zeros_like(local_eval_embed, device="cuda") for _ in range(world_size)
        ],
        "eval_ids": [
            torch.zeros_like(local_eval_ids, device="cuda") for _ in range(world_size)
        ],
        "gallery_embs": [
            torch.zeros_like(local_gallery_embed, device="cuda")
            for _ in range(world_size)
        ],
        "gallery_ids": [
            torch.zeros_like(local_gallery_ids, device="cuda")
            for _ in range(world_size)
        ],
    }

    dist.all_gather(
        [gathered_embeddings["eval_embs"], gathered_embeddings["eval_ids"]],
        [local_eval_embed, local_eval_ids],
    )

    if rank == 0:
        # 保存到文件
        save_file(gathered_embeddings, args.embedding_path)

    cleanup_distributed()


if __name__ == "__main__":
    main()
