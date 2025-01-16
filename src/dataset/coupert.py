import logging
import os
import warnings

import albumentations
import pandas as pd
import torch
import torchvision
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from ..arguments import DataArguments  # type: ignore


def get_train_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(512, 512, always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.8),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(0.09, 0.6), p=0.5
            ),
            ToTensorV2(p=1.0),
        ]
    )


def get_valid_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(512, 512, always_apply=True),
            ToTensorV2(p=1.0),
        ]
    )


class CoupertDataset(Dataset):
    def __init__(self, data_args: DataArguments, mode: str = "train"):
        self.data_args = data_args
        self.mode = mode

        assert mode in ["train", "eval", "gallery"], f"Invalid mode: {mode}"
        assert data_args.read_mode in ["text", "image", "all"], (
            f"Invalid read mode: {data_args.read_mode}"
        )

        # Chech the data_path is a directory
        assert os.path.isdir(data_args.data_dir), (
            f"Invalid data path: {data_args.data_dir}"
        )

        if mode == "train":
            query_path = os.path.join(data_args.data_dir, "coupert_train.jsonl")
            query_df = pd.read_json(query_path, lines=True, orient="records")
        elif mode == "eval":
            query_path = os.path.join(data_args.data_dir, "coupert_eval.jsonl")
            query_df = pd.read_json(query_path, lines=True, orient="records")
        elif mode == "gallery":
            query_path = os.path.join(data_args.data_dir, "coupert_gallery.jsonl")
            query_df = pd.read_json(query_path, lines=True, orient="records")

        self.data = query_df

        self.transforms = (
            get_train_transforms() if mode == "train" else get_valid_transforms()
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        title = row["title"]
        image_path = row["img_path"]

        if self.data_args.read_mode == "text":
            return {"title": title}

        try:
            # image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = self.transforms(image=image)["image"]
            image = self._read_image(image_path)
        except OSError as e:
            logging.error(f"Error loading image: {image_path}")
            logging.error(e)
            # use a blank image
            image = torch.zeros((3, 512, 512), dtype=torch.int8)

        if self.data_args.read_mode == "image":
            return {"image": image}

        return {
            "title": title,
            "image": image,
        }

    def _read_image(self, image_path):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
                torchvision.transforms.ToTensor(),
            ]
        )
        # ignore the warnings
        warnings.filterwarnings("ignore")
        image = Image.open(image_path)
        image = transforms(image)
        return image
