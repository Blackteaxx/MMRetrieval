from typing import List

import numpy as np
import torch
from tqdm import tqdm


def retrieval(
    embs: torch.Tensor,
    gallery_embs: torch.Tensor,
    gallery_ids: List[int],
    chunk_size: int = 512,
    gallery_chunk_size: int = 512,
    threshold: float = None,
    topK: int = None,
    device: str = "cuda",
) -> List[List[int]]:
    """Retrieval topK or above threshold gallery_ids for each query.
    Use chunks to reduce memory usage.

    Args:
        embs (torch.Tensor): query embeddings
        gallery_embs (torch.Tensor): gallery embeddings
        gallery_ids (List[int]): gallery ids
        chunk_size (int, optional): chunk size of query. Defaults to 512.
        gallery_chunk_size (int, optional): chunk size of gallery. Defaults to 512.
        threshold (float, optional): threshold for topP. Defaults to None.
        topK (int, optional): threshold for topK. Defaults to None.
        device (str, optional): device of embeddings during computing. Defaults to "cuda".

    Returns:
        List[List[int]]: _description_
    """

    assert threshold is not None or topK is not None, (
        "Either threshold or topK should be provided"
    )
    assert threshold is None or topK is None, (
        "Only one of threshold or topK should be provided"
    )

    embs_pt = embs
    gallery_embs_pt = gallery_embs

    # norm
    embs_pt = embs_pt / embs_pt.norm(p=2, dim=-1, keepdim=True)
    gallery_embs_pt = gallery_embs_pt / gallery_embs_pt.norm(p=2, dim=-1, keepdim=True)

    embs_pt = embs_pt.to(device)
    gallery_embs_pt = gallery_embs_pt.to(device)

    num_chunks = (embs_pt.shape[0] + chunk_size - 1) // chunk_size
    num_gallery_chunks = (
        gallery_embs_pt.shape[0] + gallery_chunk_size - 1
    ) // gallery_chunk_size
    topk_gallery_id = []

    print(f"Chunk Size: {chunk_size}, {num_chunks} chunks")
    print(f"Gallery Chunk Size: {gallery_chunk_size}, {num_gallery_chunks} chunks")

    for i in tqdm(range(num_chunks)):
        emb_start = i * chunk_size
        emb_end = min((i + 1) * chunk_size, embs_pt.shape[0])

        sim = torch.zeros((emb_end - emb_start, gallery_embs_pt.shape[0]))

        for j in range(num_gallery_chunks):
            gallery_start = j * gallery_chunk_size
            gallery_end = min((j + 1) * gallery_chunk_size, gallery_embs_pt.shape[0])

            print(f"Start: {gallery_start}, End: {gallery_end}")
            
            local_sim = (
                torch.mm(
                    embs_pt[emb_start:emb_end],
                    gallery_embs_pt[gallery_start:gallery_end].T,
                )
                .detach()
                .cpu()
            )

            sim[:, gallery_start:gallery_end] = local_sim

        if topK is not None:
            indices = torch.topk(sim, topK, dim=1).indices.cpu().numpy()
            topk_gallery_id.extend([[gallery_ids[j] for j in row] for row in indices])
        elif threshold is not None:
            mask = sim > threshold
            indices = [
                torch.nonzero(mask[j]).squeeze().cpu().numpy()
                for j in range(mask.shape[0])
            ]
            indices = [np.unique(i) for i in indices]
            sorted_indices = [
                indices[j][np.argsort(-sim[j, indices[j]].cpu().numpy())]
                for j in range(len(indices))
            ]
            topk_gallery_id.extend(
                [[gallery_ids[j] for j in row] for row in sorted_indices]
            )

    # clean up
    del embs_pt, gallery_embs_pt, sim
    torch.cuda.empty_cache()

    return topk_gallery_id
