"""Shared utilities for benchmark and optimization experiments."""
import os
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast

from siglip.data import Flickr8kDistilBERTDataset, build_siglip_transform
from siglip.loss import SigLIPLoss
from siglip.models.dual_pretrained_v2_siglip import SigLIPDualPretrainedV2Model
from siglip.utils import AverageMeter


def load_model(ckpt_path, device, n_vit_drop=0, n_db_drop=0):
    """Load Exp21 model from checkpoint, optionally with layer dropping."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args_dict = ckpt.get("args", {})

    class _Args:
        pass
    args = _Args()
    for k, v in args_dict.items():
        setattr(args, k, v)

    model = SigLIPDualPretrainedV2Model(
        embed_dim=args.embed_dim,
        num_unfrozen_vit_layers=args.num_unfrozen_vit_layers,
        num_unfrozen_db_layers=args.num_unfrozen_db_layers,
        text_proj_type=args.text_proj_type,
    ).to(device)

    if n_vit_drop > 0 or n_db_drop > 0:
        # Drop bottom frozen layers and remap state dict
        state_dict = ckpt["model"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("vision_model.encoder.layers."):
                parts = key.split(".")
                layer_idx = int(parts[3])
                if layer_idx < n_vit_drop:
                    continue  # Skip dropped layer
                new_idx = layer_idx - n_vit_drop
                parts[3] = str(new_idx)
                new_key = ".".join(parts)
                new_state_dict[new_key] = value
            elif key.startswith("text_model.transformer.layer."):
                parts = key.split(".")
                layer_idx = int(parts[3])
                if layer_idx < n_db_drop:
                    continue
                new_idx = layer_idx - n_db_drop
                parts[3] = str(new_idx)
                new_key = ".".join(parts)
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        # Drop layers from model
        if n_vit_drop > 0:
            vit_layers = model.vision_model.encoder.layers
            new_layers = torch.nn.ModuleList(list(vit_layers)[n_vit_drop:])
            model.vision_model.encoder.layers = new_layers
            model.vit_freeze_until -= n_vit_drop
            # Re-freeze remaining bottom layers
            for i in range(model.vit_freeze_until):
                for p in model.vision_model.encoder.layers[i].parameters():
                    p.requires_grad = False

        if n_db_drop > 0:
            db_layers = model.text_model.transformer.layer
            new_layers = torch.nn.ModuleList(list(db_layers)[n_db_drop:])
            model.text_model.transformer.layer = new_layers
            model.db_freeze_until -= n_db_drop
            for i in range(model.db_freeze_until):
                for p in model.text_model.transformer.layer[i].parameters():
                    p.requires_grad = False

        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(ckpt["model"])

    model.eval()
    return model, args


def load_criterion(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    criterion = SigLIPLoss().to(device)
    criterion.load_state_dict(ckpt["criterion"])
    criterion.eval()
    return criterion


def load_tokenizer():
    _here = os.path.abspath(__file__)
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    local_path = os.path.join(_repo_root, "pretrained_distilbert")
    hf_id = "distilbert-base-uncased"
    tok_source = local_path if os.path.isdir(local_path) else hf_id
    return DistilBertTokenizerFast.from_pretrained(tok_source)


def build_test_loader(args, tokenizer, batch_size=128):
    img_dir = "Images" if os.path.isdir(os.path.join(args.data_dir, "Images")) else "images"
    image_root = os.path.join(args.data_dir, img_dir)
    test_dataset = Flickr8kDistilBERTDataset(
        image_root=image_root,
        captions_file=os.path.join(args.data_dir, "test_captions.txt"),
        tokenizer=tokenizer,
        transform=build_siglip_transform(args.image_size, train=False),
        max_len=args.max_len,
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


@torch.no_grad()
def evaluate_retrieval(model, loader, device, topk=(1, 5, 10)):
    model.eval()
    all_image_embeds, all_text_embeds = [], []
    all_image_ids, all_captions = [], []

    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        if device.type == "cuda" and images.dtype == torch.float16:
            images = images.float()
        image_embeds, text_embeds = model(images, input_ids, attention_mask)
        if image_embeds.dtype == torch.float16:
            image_embeds = image_embeds.float()
            text_embeds = text_embeds.float()
        all_image_embeds.append(image_embeds)
        all_text_embeds.append(text_embeds)
        all_image_ids.extend(batch["image_id"])
        all_captions.extend(batch["caption"])

    image_embeds = F.normalize(torch.cat(all_image_embeds), dim=-1)
    text_embeds = F.normalize(torch.cat(all_text_embeds), dim=-1)

    img2indices = defaultdict(list)
    img2text = defaultdict(list)
    for idx, img_id in enumerate(all_image_ids):
        img2indices[img_id].append(idx)
        img2text[img_id].append(idx)
    unique_images = list(img2indices.keys())
    img_idx_map = {img_id: i for i, img_id in enumerate(unique_images)}

    unique_image_embeds = []
    for img_id in unique_images:
        indices = img2indices[img_id]
        unique_image_embeds.append(image_embeds[indices].mean(dim=0))
    unique_image_embeds = F.normalize(torch.stack(unique_image_embeds), dim=-1)

    sim = unique_image_embeds @ text_embeds.T

    t2i_hits = {k: 0 for k in topk}
    for text_i in range(len(all_captions)):
        img_id = all_image_ids[text_i]
        target_img_idx = img_idx_map[img_id]
        scores = sim[:, text_i]
        ranked = scores.argsort(descending=True)
        for k in topk:
            if target_img_idx in ranked[:k].tolist():
                t2i_hits[k] += 1

    i2t_hits = {k: 0 for k in topk}
    for img_idx, img_id in enumerate(unique_images):
        gt_text_indices = img2text[img_id]
        scores = sim[img_idx]
        ranked = scores.argsort(descending=True)
        for k in topk:
            if any(t in ranked[:k].tolist() for t in gt_text_indices):
                i2t_hits[k] += 1

    n_text = len(all_captions)
    n_img = len(unique_images)
    metrics = {}
    for k in topk:
        metrics[f"t2i_R@{k}"] = t2i_hits[k] / n_text
        metrics[f"i2t_R@{k}"] = i2t_hits[k] / n_img
    return metrics


@torch.no_grad()
def evaluate_loss(model, criterion, loader, device):
    model.eval()
    criterion.eval()
    meter = AverageMeter()
    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        if device.type == "cuda" and images.dtype == torch.float16:
            images = images.float()
        image_embeds, text_embeds = model(images, input_ids, attention_mask)
        if image_embeds.dtype == torch.float16:
            image_embeds = image_embeds.float()
            text_embeds = text_embeds.float()
        loss = criterion(image_embeds, text_embeds)
        meter.update(loss.item(), images.size(0))
    return meter.avg


def measure_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_mb": total_bytes / (1024 * 1024),
    }


def measure_latency(model, loader, device, n_warmup=5, n_runs=20):
    """Measure inference latency using CUDA events (GPU) or perf_counter (CPU)."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_warmup:
                break
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            _ = model(images, input_ids, attention_mask)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Measure
    latencies = []
    batch_sizes = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_runs:
                break
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
                _ = model(images, input_ids, attention_mask)
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))
            else:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.perf_counter()
                _ = model(images, input_ids, attention_mask)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms

            batch_sizes.append(images.size(0))

    avg_ms = sum(latencies) / len(latencies)
    avg_batch = sum(batch_sizes) / len(batch_sizes)
    per_sample_ms = avg_ms / avg_batch

    return {
        "avg_batch_ms": avg_ms,
        "avg_per_sample_ms": per_sample_ms,
        "n_runs": len(latencies),
        "avg_batch_size": avg_batch,
    }


def print_results(name, metrics, latency, size):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Model size: {size['total_mb']:.1f} MB, params: {size['total_params']:,}")
    print(f"  Latency: {latency['avg_batch_ms']:.2f} ms/batch, {latency['avg_per_sample_ms']:.2f} ms/sample")
    for k, v in metrics.items():
        print(f"  {k}: {v*100:.2f}%")