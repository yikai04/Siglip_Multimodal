import os
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from utils import read_caption_file


def build_transform(image_size: int = 224, train: bool = True, augment: str = "default", cj_strength: float = 0.2):
    if train:
        if augment == "randaugment":
            return T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandAugment(num_ops=2, magnitude=9),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        elif augment == "trivialaugment":
            return T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.TrivialAugmentWide(),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        else:
            return T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ColorJitter(brightness=cj_strength, contrast=cj_strength, saturation=cj_strength, hue=cj_strength * 0.5),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_siglip_transform(image_size: int = 224, train: bool = True):
    """Transform matching SigLIP ViT preprocessing: BICUBIC resize + SigLIP normalization."""
    SIGLIP_MEAN = [0.5, 0.5, 0.5]
    SIGLIP_STD = [0.5, 0.5, 0.5]
    if train:
        return T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
        ])
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
    ])


class Flickr8kDataset(Dataset):
    def __init__(self, image_root: str, captions_file: str, tokenizer, transform=None):
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transform or build_transform(train=True)
        self.rows = read_caption_file(captions_file)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = os.path.join(self.image_root, row["image_id"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        input_ids = torch.tensor(self.tokenizer.encode(row["caption"]), dtype=torch.long)
        return {
            "image": image,
            "input_ids": input_ids,
            "image_id": row["image_id"],
            "caption": row["caption"],
        }


class Flickr8kDistilBERTDataset(Dataset):
    """Flickr8k dataset variant using DistilBertTokenizerFast with attention_mask."""

    def __init__(self, image_root: str, captions_file: str, tokenizer, transform=None, max_len: int = 64):
        self.image_root = image_root
        self.tokenizer = tokenizer  # DistilBertTokenizerFast
        self.transform = transform or build_siglip_transform(train=True)
        self.max_len = max_len
        self.rows = read_caption_file(captions_file)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = os.path.join(self.image_root, row["image_id"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        encoded = self.tokenizer(
            row["caption"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "image_id": row["image_id"],
            "caption": row["caption"],
        }


class SyntheticPairDataset(Dataset):
    """Tiny deterministic dataset for code sanity checks without Flickr8k images."""

    def __init__(self, tokenizer, size: int = 256, image_size: int = 64, num_concepts: int = 32):
        self.tokenizer = tokenizer
        self.size = size
        self.image_size = image_size
        self.num_concepts = num_concepts

    @staticmethod
    def captions(num_concepts: int = 32):
        return [f"synthetic object {idx}" for idx in range(num_concepts)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        concept = idx % self.num_concepts
        generator = torch.Generator().manual_seed(concept)
        image = torch.rand((3, self.image_size, self.image_size), generator=generator)
        image[0].mul_((concept + 1) / self.num_concepts)
        image[1].mul_(1.0 - concept / self.num_concepts)
        caption = f"synthetic object {concept}"
        input_ids = torch.tensor(self.tokenizer.encode(caption), dtype=torch.long)
        return {
            "image": image,
            "input_ids": input_ids,
            "image_id": f"synthetic_{concept:03d}.jpg",
            "caption": caption,
        }
