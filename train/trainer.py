import os
from typing import Optional

import torch
from transformers import Trainer

from .criterion import InfoNCELoss, TripletLogitsLoss


class MMTrainer(Trainer):
    def compute_loss(self, model, inputs, **kwargs):
        query = inputs["query"]
        pos = inputs["pos"]
        neg = inputs["neg"]

        query_embeddings = model(query)
        pos_embeddings = model(pos)
        neg_embeddings = model(neg)

        # normalize embeddings
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
        pos_embeddings = torch.nn.functional.normalize(pos_embeddings, p=2, dim=-1)
        neg_embeddings = torch.nn.functional.normalize(neg_embeddings, p=2, dim=-1)

        if self.args.loss_type == "nce":
            criterion = InfoNCELoss(self.args)
        elif self.args.loss_type == "triplet":
            criterion = TripletLogitsLoss(self.args)

        loss = criterion(query_embeddings, pos_embeddings, neg_embeddings)

        return loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # save clip model
        self.model.clip_model.save_pretrained(output_dir)

        # save processor
        if self.model.processor is not None:
            self.model.processor.save_pretrained(output_dir)

        # save the self defined model, inheriting from torch.nn.Module
        state_dict = self.model.state_dict()
        # clip_state_dict = {k: v for k, v in state_dict.items() if k.startswith("clip_model")}
        # only save the fc layers
        fc_state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("clip_model")
        }

        torch.save(fc_state_dict, os.path.join(output_dir, "model.pt"))
