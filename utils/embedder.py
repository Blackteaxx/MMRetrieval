import torch
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm


class Embedder:
    def __init__(self, model, tokenizer, processor, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device

        self.embeddings = {}

    def embed(self, loader: DataLoader, mode="eval"):
        self.model.eval()
        if self.device is not None and not isinstance(self.model, DDP):
            self.model.to(self.device)

        # Select the forward method based on the model type
        if self.model.__class__.__name__ == "DataParallel":
            name = self.model.module.__class__.__name__
        elif self.model.__class__.__name__ == "DistributedDataParallel":
            name = self.model.module.__class__.__name__
        else:
            name = self.model.__class__.__name__

        if name == "BertModel":
            forward = self._forward_bert
        if name == "CLIPForEmbedding":
            forward = self._forward_clip
        if "SigLIP" in name:
            forward = self._forward_siglip

        with torch.no_grad():
            with torch.amp.autocast(self.device, torch.float16):
                embeddings, ids = forward(loader=loader)

        embeddings /= torch.norm(embeddings, p=2, dim=1, keepdim=True)
        self.embeddings[f"{mode}_embs"] = embeddings
        self.embeddings[f"{mode}_ids"] = ids

        return embeddings, ids

    def save_embeddings(self, path):
        save_file(self.embeddings, path)

    def _forward_bert(self, loader: DataLoader):
        embeddings = []
        for inputs in tqdm(loader):
            inputs = self._batch_to_device(inputs, self.device)
            outputs = self.model(**inputs).last_hidden_state[:, -1].detach().cpu()
            embeddings.append(outputs)
        return torch.cat(embeddings)

    def _forward_clip(self, loader: DataLoader):
        embeddings = []
        for inputs in tqdm(loader):
            inputs = self._batch_to_device(inputs, self.device)
            outputs = self.model(inputs).detach().cpu()
            embeddings.append(outputs)
        return torch.cat(embeddings)

    def _forward_siglip(self, loader: DataLoader):
        embeddings = []
        sample_ids = []

        for inputs in tqdm(loader):
            inputs = self._batch_to_device(inputs, self.device)
            outputs = self.model(inputs).detach().cpu()
            sample_ids.append(inputs["global_indices"].detach().cpu())
            embeddings.append(outputs)
        return torch.cat(embeddings), torch.cat(sample_ids)

    def _batch_to_device(self, batch, device):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

            if isinstance(v, dict):
                self._batch_to_device(v, device)

        return batch
