"""Train SigLIP with dual pretrained model (Exp21): partial unfreezing on both sides.

ViT-B/16: bottom 9 layers + embeddings frozen, top 3 trainable (LR=1e-5)
DistilBERT: bottom 4 layers + embeddings frozen, top 2 trainable (LR=2e-5)
Projection + Loss: fully trainable (LR=3e-4)

Usage:
    python train_siglip_dual_pretrained_v2.py --data-dir Flickr8k --epochs 50 --output-dir outputs/exp21_dual_pretrained_v2
"""

import argparse
import math
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast
from tqdm import tqdm

from data_loader import Flickr8kDistilBERTDataset, build_siglip_transform
from loss import SigLIPLoss
from models.dual_pretrained_v2_siglip import SigLIPDualPretrainedV2Model
from utils import AverageMeter, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train dual pretrained SigLIP V2 (Exp21).")
    parser.add_argument("--data-dir", default="Flickr8k")
    parser.add_argument("--output-dir", default="outputs/exp21_dual_pretrained_v2")
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--num-unfrozen-vit-layers", type=int, default=3)
    parser.add_argument("--num-unfrozen-db-layers", type=int, default=2)
    parser.add_argument("--text-proj-type", default="linear", choices=["linear", "identity"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--vit-lr", type=float, default=1e-5)
    parser.add_argument("--db-lr", type=float, default=2e-5)
    parser.add_argument("--proj-lr", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--vit-wd", type=float, default=0.01)
    parser.add_argument("--db-wd", type=float, default=0.01)
    parser.add_argument("--proj-wd", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def build_dataloaders(args):
    local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pretrained_distilbert")
    hf_id = "distilbert-base-uncased"
    tok_source = local_path if os.path.isdir(local_path) else hf_id
    tokenizer = DistilBertTokenizerFast.from_pretrained(tok_source)

    img_dir = "Images" if os.path.isdir(os.path.join(args.data_dir, "Images")) else "images"
    image_root = os.path.join(args.data_dir, img_dir)

    train_dataset = Flickr8kDistilBERTDataset(
        image_root=image_root,
        captions_file=os.path.join(args.data_dir, "train_captions.txt"),
        tokenizer=tokenizer,
        transform=build_siglip_transform(args.image_size, train=True),
        max_len=args.max_len,
    )
    val_dataset = Flickr8kDistilBERTDataset(
        image_root=image_root,
        captions_file=os.path.join(args.data_dir, "val_captions.txt"),
        tokenizer=tokenizer,
        transform=build_siglip_transform(args.image_size, train=False),
        max_len=args.max_len,
    )
    test_dataset = Flickr8kDistilBERTDataset(
        image_root=image_root,
        captions_file=os.path.join(args.data_dir, "test_captions.txt"),
        tokenizer=tokenizer,
        transform=build_siglip_transform(args.image_size, train=False),
        max_len=args.max_len,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    return tokenizer, train_loader, val_loader, test_loader


def train_one_epoch(model, criterion, loader, optimizer, scheduler, device):
    model.train()
    criterion.train()
    meter = AverageMeter()
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        image_embeds, text_embeds = model(images, input_ids, attention_mask)
        loss = criterion(image_embeds, text_embeds)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        meter.update(loss.item(), images.size(0))
    return meter.avg


@torch.no_grad()
def evaluate_loss(model, criterion, loader, device):
    model.eval()
    criterion.eval()
    meter = AverageMeter()
    for batch in tqdm(loader, desc="eval-loss", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        image_embeds, text_embeds = model(images, input_ids, attention_mask)
        loss = criterion(image_embeds, text_embeds)
        meter.update(loss.item(), images.size(0))
    return meter.avg


@torch.no_grad()
def evaluate_retrieval(model, loader, device, topk=(1, 5, 10)):
    model.eval()
    all_image_embeds, all_text_embeds = [], []
    all_image_ids, all_captions = [], []

    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image_embeds, text_embeds = model(images, input_ids, attention_mask)
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


class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * decay
        super().__init__(optimizer, lr_lambda)


def save_checkpoint(path, model, criterion, optimizer, scheduler, epoch, metrics, args, best_val_r1=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "criterion": criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "tokenizer_type": "distilbert",
        "metrics": metrics,
        "best_val_r1": best_val_r1,
        "args": vars(args),
    }, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer, train_loader, val_loader, test_loader = build_dataloaders(args)

    model = SigLIPDualPretrainedV2Model(
        embed_dim=args.embed_dim,
        num_unfrozen_vit_layers=args.num_unfrozen_vit_layers,
        num_unfrozen_db_layers=args.num_unfrozen_db_layers,
        text_proj_type=args.text_proj_type,
    ).to(device)

    criterion = SigLIPLoss().to(device)

    # Parameter groups for differential learning rates
    # 1. ViT top layers + image_adapter
    vit_unfrozen_params = []
    for i in range(model.vit_freeze_until, len(model.vision_model.encoder.layers)):
        vit_unfrozen_params.extend(model.vision_model.encoder.layers[i].parameters())
    vit_unfrozen_params = [p for p in vit_unfrozen_params if p.requires_grad]
    if hasattr(model.image_adapter, 'weight'):
        vit_unfrozen_params.extend(model.image_adapter.parameters())

    # 2. DistilBERT top layers
    db_unfrozen_params = []
    for i in range(model.db_freeze_until, len(model.text_model.transformer.layer)):
        db_unfrozen_params.extend(model.text_model.transformer.layer[i].parameters())
    db_unfrozen_params = [p for p in db_unfrozen_params if p.requires_grad]

    # 3. Projection + Loss
    proj_and_loss_params = list(model.text_projection.parameters()) + list(criterion.parameters())

    total_params = sum(p.numel() for p in model.parameters())
    vit_trainable = sum(p.numel() for p in vit_unfrozen_params)
    db_trainable = sum(p.numel() for p in db_unfrozen_params)
    proj_trainable = sum(p.numel() for p in proj_and_loss_params) - 2
    total_trainable = vit_trainable + db_trainable + proj_trainable + 2
    print(f"Total params: {total_params:,}")
    print(f"Trainable: {total_trainable:,} (ViT top={vit_trainable:,}, DistilBERT top={db_trainable:,}, proj={proj_trainable:,}, loss=2)")
    print(f"Frozen: {total_params - total_trainable:,}")

    optimizer = torch.optim.AdamW([
        {"params": vit_unfrozen_params, "lr": args.vit_lr, "weight_decay": args.vit_wd},
        {"params": db_unfrozen_params, "lr": args.db_lr, "weight_decay": args.db_wd},
        {"params": proj_and_loss_params, "lr": args.proj_lr, "weight_decay": args.proj_wd},
    ])

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps, args.min_lr_ratio)
    print(f"device={device} total_steps={total_steps} warmup_steps={warmup_steps}")
    print(f"LR groups: vit={args.vit_lr}, db_top={args.db_lr}, proj={args.proj_lr}")

    best_val_r1 = -1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        criterion.load_state_dict(ckpt["criterion"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val_r1 = ckpt.get("best_val_r1", -1.0)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, scheduler, device)
        val_loss = evaluate_loss(model, criterion, val_loader, device)
        val_metrics = evaluate_retrieval(model, val_loader, device)
        val_r1 = val_metrics["t2i_R@1"]
        scale = criterion.logit_scale.exp().item()
        bias = criterion.logit_bias.item()
        metric_text = " ".join([f"{name}={value * 100:.2f}" for name, value in val_metrics.items()])
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"scale={scale:.2f} bias={bias:.2f} {metric_text}")

        if val_r1 > best_val_r1:
            best_val_r1 = val_r1
            save_checkpoint(
                os.path.join(args.output_dir, "best_siglip.pt"),
                model, criterion, optimizer, scheduler, epoch, val_metrics, args, best_val_r1,
            )
        save_checkpoint(
            os.path.join(args.output_dir, "latest_siglip.pt"),
            model, criterion, optimizer, scheduler, epoch, val_metrics, args, best_val_r1,
        )

    # Final test evaluation
    test_loss = evaluate_loss(model, criterion, test_loader, device)
    test_metrics = evaluate_retrieval(model, test_loader, device)
    metric_text = " ".join([f"{name}={value * 100:.2f}" for name, value in test_metrics.items()])
    print(f"final test_loss={test_loss:.4f} {metric_text}")


if __name__ == "__main__":
    main()