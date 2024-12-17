import concurrent.futures
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from .arguments import DataArguments  # type: ignore


def get_collate_fn(processor):
    def collate_fn(batch):
        query = [x["query"] for x in batch]
        pos = [x["pos"] for x in batch]
        neg = [x["neg"] for x in batch]

        # aggregate text and image data
        query = {k: [x[k] for x in query] for k in query[0].keys()}
        pos = {k: [x[k] for x in pos] for k in pos[0].keys()}
        neg = {k: [x[k] for x in neg] for k in neg[0].keys()}

        # process text data
        query["text"] = processor(
            text=query["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        pos["text"] = processor(
            text=pos["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        neg["text"] = processor(
            text=neg["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

        # process image data
        query["image"] = processor(
            images=query["image"], return_tensors="pt", do_rescale=False
        )
        pos["image"] = processor(
            images=pos["image"], return_tensors="pt", do_rescale=False
        )
        neg["image"] = processor(
            images=neg["image"], return_tensors="pt", do_rescale=False
        )

        return {
            "query": query,
            "pos": pos,
            "neg": neg,
        }

    return collate_fn


class ShopeeDataset(Dataset):
    def __init__(self, args: DataArguments, split: str = "train"):
        self.args = args
        self.df = pd.read_json(args.data_dir, lines=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # convert to RGB
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.ToTensor(),
            ]
        )
        self.split = split
        self.len = len(self.df)
        # self.imgs = self._read_all_images()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.split == "train":
            query_text = row["query"]
            pos_text = row["pos_txt"][0]
            neg_text = row["neg_txt"][0]

            query_img_path = os.path.join(self.args.img_dir, row["image"])
            pos_img_path = os.path.join(self.args.img_dir, row["pos_img"][0])
            neg_img_path = os.path.join(self.args.img_dir, row["neg_img"][0])

            query_img = self._get_image(query_img_path)
            pos_img = self._get_image(pos_img_path)
            neg_img = self._get_image(neg_img_path)

            # query_img = self.imgs[query_img_path]
            # pos_img = self.imgs[pos_img_path]
            # neg_img = self.imgs[neg_img_path]

            return {
                "query": {
                    "text": query_text,
                    "image": query_img,
                },
                "pos": {
                    "text": pos_text,
                    "image": pos_img,
                },
                "neg": {
                    "text": neg_text,
                    "image": neg_img,
                },
            }

        elif self.split == "valid":
            title = row["title"]
            image_path = os.path.join(self.args.img_dir, row["image"])

            img = self._get_image(image_path)

            # img = self.imgs[image_path]

            return {
                "title": title,
                "image": img,
            }

    def _get_image(self, path):
        img = Image.open(path)
        img = self.transform(img)
        return img

    def _read_all_images(self):
        def load_image(idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.args.img_dir, row["image"])
            img = Image.open(img_path)
            img = self.transform(img)
            return img_path, img

        imgs = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(load_image, idx) for idx in range(self.len)]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=self.len,
                desc="Reading images",
            ):
                img_path, img = future.result()
                imgs[img_path] = img

        return imgs
