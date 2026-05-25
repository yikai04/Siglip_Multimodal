"""Train SigLIP with frozen pretrained ViT vision encoder (Exp19).

Uses google/siglip-base-patch16-224 ViT-B/16 (frozen) as image encoder,
with a trainable Transformer text encoder and SigLIP loss.

Usage:
    python train_siglip_pretrained.py --data-dir Flickr8k --epochs 100 --output-dir outputs/exp19_pretrained_vit
"""

import argparse
import math
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision import transforms as T

from data_loader import Flickr8kDataset, SyntheticPairDataset
from loss import SigLIPLoss
from models.pretrained_siglip import SigLIPPretrainedModel
from utils import AverageMeter, SimpleTokenizer, read_caption_file, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train SigLIP with frozen pretrained ViT encoder.")
    parser.add_argument("--data-dir", default="Flickr8k")
    parser.add_argument("--output-dir", default="outputs/exp19_pretrained_vit")
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--text-dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


# SigLIP ViT uses different normalization than ImageNet
SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD = [0.5, 0.5, 0.5]


def build_siglip_transform(image_size=224, train=True):
    """Build transform with SigLIP normalization (mean=0.5, std=0.5)."""
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


def maybe_subset(dataset, limit):
    if limit and limit > 0:
        return Subset(dataset, range(min(limit, len(dataset))))
    return dataset


def build_dataloaders(args):
    all_rows = read_caption_file(os.path.join(args.data_dir, "captions.txt"))
    tokenizer = SimpleTokenizer((row["caption"] for row in all_rows), min_freq=args.min_freq, max_len=args.max_len)
    image_root = os.path.join(args.data_dir, "images")

    train_dataset = Flickr8kDataset(
        image_root=image_root,
        captions_file=os.path.join(args.data_dir, "train_captions.txt"),
        tokenizer=tokenizer,
        transform=build_siglip_transform(args.image_size, train=True),
    )
    val_dataset = Flickr8kDataset(
        image_root=image_root,
        captions_file=os.path.join(args.data_dir, "val_captions.txt"),
        tokenizer=tokenizer,
        transform=build_siglip_transform(args.image_size, train=False),
    )
    test_dataset = Flickr8kDataset(
        image_root=image_root,
        captions_file=os.path.join(args.data_dir, "test_captions.txt"),
        tokenizer=tokenizer,
        transform=build_siglip_transform(args.image_size, train=False),
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
        image_embeds, text_embeds = model(images, input_ids)
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
        image_embeds, text_embeds = model(images, input_ids)
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
        image_embeds, text_embeds = model(images, input_ids)
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


def save_checkpoint(path, model, criterion, optimizer, scheduler, tokenizer, epoch, metrics, args, best_val_r1=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "criterion": criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "tokenizer_word2idx": tokenizer.word2idx,
        "metrics": metrics,
        "best_val_r1": best_val_r1,
        "args": vars(args),
    }, path)


class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, min_lr_ratio=0.01):
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * decay
        super().__init__(optimizer, lr_lambda)
        self.base_lr = base_lr


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer, train_loader, val_loader, test_loader = build_dataloaders(args)

    model = SigLIPPretrainedModel(
        vocab_size=len(tokenizer),
        embed_dim=args.embed_dim,
        max_len=args.max_len,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        text_dropout=args.text_dropout,
    ).to(device)

    # Only optimize trainable parameters (frozen ViT excluded)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"Total params: {total_params:,}  Trainable: {trainable_count:,}  Frozen: {total_params - trainable_count:,}")

    criterion = SigLIPLoss().to(device)
    optimizer = torch.optim.AdamW(
        trainable_params + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    warmup_steps = min(warmup_steps, total_steps)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps, args.lr, args.min_lr_ratio)
    print(f"device={device} vocab_size={len(tokenizer)} total_steps={total_steps} warmup_steps={warmup_steps}")

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
        # Restore tokenizer
        w2i = ckpt.get("tokenizer_word2idx")
        if w2i:
            tokenizer.word2idx = dict(w2i)
            tokenizer.idx2word = [""] * len(tokenizer.word2idx)
            for w, idx in tokenizer.word2idx.items():
                tokenizer.idx2word[idx] = w

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
                model, criterion, optimizer, scheduler, tokenizer, epoch, val_metrics, args, best_val_r1,
            )
        save_checkpoint(
            os.path.join(args.output_dir, "latest_siglip.pt"),
            model, criterion, optimizer, scheduler, tokenizer, epoch, val_metrics, args, best_val_r1,
        )

    # Final test evaluation
    test_loss = evaluate_loss(model, criterion, test_loader, device)
    test_metrics = evaluate_retrieval(model, test_loader, device)
    metric_text = " ".join([f"{name}={value * 100:.2f}" for name, value in test_metrics.items()])
    print(f"final test_loss={test_loss:.4f} {metric_text}")


if __name__ == "__main__":
    main()
