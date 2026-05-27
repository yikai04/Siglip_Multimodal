"""Load Exp21 BF16 model and pre-compute all Flickr8k embeddings for fast retrieval."""
import os
import time
import logging

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .config import (
    CKPT_PATH, DEVICE, USE_BF16, MAX_LEN, IMAGE_SIZE, EMBED_DIM,
    NUM_UNFROZEN_VIT_LAYERS, NUM_UNFROZEN_DB_LAYERS, DISTILBERT_LOCAL, DATA_DIR,
)
from ..models.dual_pretrained_v2_siglip import SigLIPDualPretrainedV2Model

logger = logging.getLogger(__name__)


def _load_model(ckpt_path, device, use_bf16=False):
    """Load Exp21 model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    def _get(key, default=None):
        return args.get(key, default) if isinstance(args, dict) else getattr(args, key, default)

    model = SigLIPDualPretrainedV2Model(
        embed_dim=_get("embed_dim", EMBED_DIM),
        num_unfrozen_vit_layers=_get("num_unfrozen_vit_layers", NUM_UNFROZEN_VIT_LAYERS),
        num_unfrozen_db_layers=_get("num_unfrozen_db_layers", NUM_UNFROZEN_DB_LAYERS),
        text_proj_type=_get("text_proj_type", "linear"),
    )
    model.load_state_dict(ckpt["model"])

    if use_bf16:
        model = model.to(torch.bfloat16)
    model = model.to(device).eval()

    return model, ckpt


def _load_tokenizer():
    from transformers import DistilBertTokenizerFast
    if os.path.isdir(DISTILBERT_LOCAL):
        return DistilBertTokenizerFast.from_pretrained(DISTILBERT_LOCAL)
    return DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def _build_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def _load_flickr8k():
    img_dir = os.path.join(DATA_DIR, "Images")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(DATA_DIR, "images")

    cap_file = os.path.join(DATA_DIR, "test_captions.txt")
    if not os.path.isfile(cap_file):
        cap_file = os.path.join(DATA_DIR, "Flickr8k.token.txt")

    captions_map = {}
    if "test_captions" in cap_file:
        with open(cap_file, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("image"):
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                fname, cap = parts
                captions_map.setdefault(fname, []).append(cap.strip())
    else:
        with open(cap_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, cap = line.split("\t", 1)
                fname = key.split("#")[0]
                captions_map.setdefault(fname, []).append(cap)

    split_file = os.path.join(DATA_DIR, "Flickr_8k.testImages.txt")
    if os.path.isfile(split_file):
        with open(split_file, encoding="utf-8") as f:
            test_images = [ln.strip() for ln in f if ln.strip()]
    else:
        test_images = list(captions_map.keys())

    data = []
    for fname in test_images:
        img_path = os.path.join(img_dir, fname)
        if not os.path.isfile(img_path):
            continue
        caps = captions_map.get(fname, [""])
        data.append({
            "image_path": img_path,
            "filename": fname,
            "captions": caps,
            "primary_caption": caps[0],
        })

    logger.info(f"Loaded Flickr8k: {len(data)} images")
    return data


class InferenceEngine:
    def __init__(self, ckpt_path=CKPT_PATH, device=DEVICE, use_bf16=USE_BF16):
        logger.info(f"Loading Exp21 model (BF16={use_bf16})...")
        t0 = time.time()

        self.device = device
        self.use_bf16 = use_bf16
        self.model, self.ckpt = _load_model(ckpt_path, device, use_bf16=use_bf16)
        self.tokenizer = _load_tokenizer()
        self.transform = _build_transform()
        self.flickr_data = _load_flickr8k()

        self.model_size_mb = sum(
            p.nelement() * p.element_size() for p in self.model.parameters()
        ) / (1024 * 1024)

        logger.info(f"Model loaded in {time.time()-t0:.1f}s")
        self._precompute()

    def _encode_batch_images(self, img_paths, batch_size=64):
        all_embeds = []
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i:i+batch_size]
            tensors = torch.stack([
                self.transform(Image.open(p).convert("RGB"))
                for p in batch_paths
            ]).to(self.device)
            if self.use_bf16:
                tensors = tensors.to(torch.bfloat16)

            with torch.no_grad():
                img_emb = self.model.encode_image(tensors)
            all_embeds.append(img_emb.float())

        return torch.cat(all_embeds, dim=0)

    def _encode_batch_texts(self, texts, batch_size=128):
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded = self.tokenizer(
                batch_texts, return_tensors="pt", padding="max_length",
                truncation=True, max_length=MAX_LEN,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                txt_emb = self.model.encode_text(input_ids, attention_mask)
            all_embeds.append(txt_emb.float())

        return torch.cat(all_embeds, dim=0)

    def _precompute(self):
        logger.info("Pre-computing Flickr8k embeddings...")
        t0 = time.time()

        img_paths = [d["image_path"] for d in self.flickr_data]
        all_captions = [d["primary_caption"] for d in self.flickr_data]

        self.image_embeds = F.normalize(
            self._encode_batch_images(img_paths), dim=-1
        )
        self.text_embeds = F.normalize(
            self._encode_batch_texts(all_captions), dim=-1
        )

        logger.info(f"Pre-computation done in {time.time()-t0:.1f}s")

    def search_text_to_image(self, query_text, top_k=5):
        """Text → Image retrieval. Only runs text encoder."""
        encoded = self.tokenizer(
            query_text, return_tensors="pt", padding="max_length",
            truncation=True, max_length=MAX_LEN,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            txt_emb = self.model.encode_text(input_ids, attention_mask)
        query_emb = F.normalize(txt_emb.float(), dim=-1)

        scores = (query_emb @ self.image_embeds.T).squeeze(0)
        top_indices = scores.topk(min(top_k, len(self.flickr_data))).indices.tolist()

        return [(self.flickr_data[idx]["image_path"],
                 self.flickr_data[idx]["primary_caption"],
                 scores[idx].item()) for idx in top_indices]

    def search_image_to_text(self, query_image, top_k=5):
        """Image → Text retrieval. Only runs image encoder."""
        if query_image is None:
            return []
        tensor = self.transform(query_image.convert("RGB")).unsqueeze(0).to(self.device)
        if self.use_bf16:
            tensor = tensor.to(torch.bfloat16)

        with torch.no_grad():
            img_emb = self.model.encode_image(tensor)
        query_emb = F.normalize(img_emb.float(), dim=-1)

        scores = (query_emb @ self.text_embeds.T).squeeze(0)
        top_indices = scores.topk(min(top_k, len(self.flickr_data))).indices.tolist()

        return [(self.flickr_data[idx]["primary_caption"], scores[idx].item()) for idx in top_indices]