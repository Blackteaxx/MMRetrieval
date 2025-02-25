import logging
import os

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class CLIPForEmbedding(CLIPModel):
    def forward(
        self,
        inputs,
    ):
        """_summary_

        Args:
            inputs (Dict[str, Dict[str, torch.Tensor]]):
                {
                    "text": {
                        "input_ids": torch.Tensor,
                        "attention_mask": torch.Tensor,
                    },
                    "image": {
                        "pixel_values": torch.Tensor,
                    }
                }

        Returns:
            Dict[str, torch.Tensor]:
                {
                    "text": torch.Tensor,
                    "image": torch.Tensor
                }
        """

        text_inputs = inputs["text"]
        image_inputs = inputs["image"]

        text_embs = self.get_text_features(**text_inputs)
        image_embs = self.get_image_features(**image_inputs)

        # normalize
        text_norm = text_embs.detach().norm(p=2, dim=-1, keepdim=True)
        text_embs_normed = text_embs / text_norm

        image_norm = image_embs.detach().norm(p=2, dim=-1, keepdim=True)
        image_embs_normed = image_embs / image_norm

        return {"text": text_embs_normed, "image": image_embs_normed}


class CLIPForFusion(nn.Module):
    def __init__(self, clip_model=None, processor=None):
        super().__init__()
        self.clip_model = clip_model
        self.processor = processor

        self.txt_hidden_dim = clip_model.projection_dim
        self.img_hidden_dim = clip_model.projection_dim
        self.fusion_dim = 512

        self.text_fc = nn.Linear(self.txt_hidden_dim, self.fusion_dim)
        self.image_fc = nn.Linear(self.img_hidden_dim, self.fusion_dim)
        self.act_fn = nn.ReLU()

        self.fusion_fc = nn.Linear(self.fusion_dim * 2, self.fusion_dim)

        self.projector = nn.Linear(self.fusion_dim, self.fusion_dim)

    @classmethod
    def from_pretrained(cls, model_name_or_path):
        clip_model = CLIPForEmbedding.from_pretrained(model_name_or_path)
        processor = CLIPProcessor.from_pretrained(model_name_or_path)
        model = cls(clip_model, processor)
        model_path = os.path.join(model_name_or_path, "model.pt")

        if os.path.exists(model_path):
            logging.info(f"Load model.pt from {model_path}")
            state_dict = torch.load(model_path, weights_only=True)
            # only load the fc layers
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
            logging.info(f"missing keys: {missing_keys}")
            logging.info(f"unexpected keys: {unexpected_keys}")
        else:
            logging.warning(
                f"model.pt not found in {model_name_or_path}, initialize from scratch"
            )
        return model

    def forward(self, inputs):
        embed = self.clip_model(inputs)

        text_embs = embed["text"]
        image_embs = embed["image"]

        text_fusion = self.act_fn(self.text_fc(text_embs))
        image_fusion = self.act_fn(self.image_fc(image_embs))

        fusion = self.act_fn(
            self.fusion_fc(torch.cat([text_fusion, image_fusion], dim=-1))
        )

        proj = self.act_fn(self.projector(fusion))

        return proj

    def encode(self, inputs):
        embed = self.clip_model(inputs)
        text_embs = embed["text"]
        image_embs = embed["image"]

        text_fusion = self.act_fn(self.text_fc(text_embs))
        image_fusion = self.act_fn(self.image_fc(image_embs))

        fusion = self.act_fn(
            self.fusion_fc(torch.cat([text_fusion, image_fusion], dim=-1))
        )

        return fusion
