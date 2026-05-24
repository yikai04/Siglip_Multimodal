import argparse
import copy
import math
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data_loader import Flickr8kDataset, SyntheticPairDataset, build_transform
from loss import SigLIPLoss
from models import SigLIPModel
from utils import AverageMeter, SimpleTokenizer, read_caption_file, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a small SigLIP model on Flickr8k.")
    parser.add_argument("--data-dir", default="Flickr8k", help="Directory containing images/ and caption split files.")
    parser.add_argument("--output-dir", default="outputs/siglip_resnet_transformer")
    parser.add_argument("--text-encoder", choices=["transformer", "mlp"], default="transformer")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=32, help="Base channel width of the custom ResNet.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads in Transformer text encoder.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of Transformer layers in text encoder.")
    parser.add_argument("--text-dropout", type=float, default=0.1, help="Dropout rate in Transformer text encoder.")
    parser.add_argument("--proj-head", choices=["linear", "mlp"], default="linear", help="Projection head type: linear or 2-layer MLP.")
    parser.add_argument("--augment", choices=["default", "randaugment", "trivialaugment"], default="default", help="Training augmentation strategy.")
    parser.add_argument("--cj-strength", type=float, default=0.2, help="ColorJitter strength (brightness/contrast/saturation). Hue is half this value.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps. Effective batch = batch_size * grad_accum.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=0, help="Fixed warmup steps; 0 means use --warmup-ratio.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup as fraction of total steps (used when --warmup-steps=0).")
    parser.add_argument("--min-lr-ratio", type=float, default=0.01, help="Minimum LR as fraction of base LR at the end of cosine decay.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.0, help="EMA decay for model weights; 0 means no EMA. Typical: 0.99.")
    parser.add_argument("--ema-start-epoch", type=int, default=0, help="Epoch at which to start EMA updates. 0 means from the start.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--limit-train", type=int, default=0, help="Use the first N train samples; 0 means all.")
    parser.add_argument("--limit-eval", type=int, default=0, help="Use the first N val/test samples; 0 means all.")
    parser.add_argument("--dry-run", action="store_true", help="Run on synthetic pairs to verify the code path.")
    parser.add_argument("--resume", default="", help="Path to a checkpoint to resume training from.")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def maybe_subset(dataset, limit):
    if limit and limit > 0:
        return Subset(dataset, range(min(limit, len(dataset))))
    return dataset


def build_dataloaders(args):
    if args.dry_run:
        captions = SyntheticPairDataset.captions()
        tokenizer = SimpleTokenizer(captions, min_freq=1, max_len=args.max_len)
        train_dataset = SyntheticPairDataset(tokenizer, size=256, image_size=64)
        val_dataset = SyntheticPairDataset(tokenizer, size=64, image_size=64)
        test_dataset = SyntheticPairDataset(tokenizer, size=64, image_size=64)
        args.image_size = 64
    else:
        all_rows = read_caption_file(os.path.join(args.data_dir, "captions.txt"))
        tokenizer = SimpleTokenizer((row["caption"] for row in all_rows), min_freq=args.min_freq, max_len=args.max_len)
        image_root = os.path.join(args.data_dir, "images")
        train_dataset = Flickr8kDataset(
            image_root=image_root,
            captions_file=os.path.join(args.data_dir, "train_captions.txt"),
            tokenizer=tokenizer,
            transform=build_transform(args.image_size, train=True, augment=args.augment, cj_strength=args.cj_strength),
        )
        val_dataset = Flickr8kDataset(
            image_root=image_root,
            captions_file=os.path.join(args.data_dir, "val_captions.txt"),
            tokenizer=tokenizer,
            transform=build_transform(args.image_size, train=False),
        )
        test_dataset = Flickr8kDataset(
            image_root=image_root,
            captions_file=os.path.join(args.data_dir, "test_captions.txt"),
            tokenizer=tokenizer,
            transform=build_transform(args.image_size, train=False),
        )

    train_dataset = maybe_subset(train_dataset, args.limit_train)
    val_dataset = maybe_subset(val_dataset, args.limit_eval)
    test_dataset = maybe_subset(test_dataset, args.limit_eval)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return tokenizer, train_loader, val_loader, test_loader


def train_one_epoch(model, criterion, loader, optimizer, scheduler, device, grad_accum=1):
    model.train()
    criterion.train()
    meter = AverageMeter()
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        image_embeds, text_embeds = model(images, input_ids)
        loss = criterion(image_embeds, text_embeds)
        loss = loss / grad_accum

        loss.backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        meter.update(loss.item() * grad_accum, images.size(0))
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
    # TODO(STUDENT): this complete grouped retrieval evaluator can be omitted from

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

    image_embeds = torch.cat(all_image_embeds)   # [N, D]
    text_embeds  = torch.cat(all_text_embeds)    # [N, D]

    # 归一化
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds  = F.normalize(text_embeds, dim=-1)

    # 按 image_id 分组（同一图像有多个 caption）
    from collections import defaultdict
    img2indices = defaultdict(list)
    img2text = defaultdict(list)
    for idx, img_id in enumerate(all_image_ids):
        img2indices[img_id].append(idx)
        img2text[img_id].append(idx)
    unique_images = list(img2indices.keys())
    img_idx_map = {img_id: i for i, img_id in enumerate(unique_images)}

    # 聚合同一图像的多个 embedding（取均值），得到每个 unique 图像的表示
    unique_image_embeds = []
    for img_id in unique_images:
        indices = img2indices[img_id]
        unique_image_embeds.append(image_embeds[indices].mean(dim=0))
    unique_image_embeds = torch.stack(unique_image_embeds)  # [n_unique_img, D]
    unique_image_embeds = F.normalize(unique_image_embeds, dim=-1)

    # 相似度矩阵
    sim = unique_image_embeds @ text_embeds.T  # [n_unique_img, N_text]

    # Text-to-Image 检索：对每个文本，找最相似的图像
    t2i_hits = {k: 0 for k in topk}
    for text_i in range(len(all_captions)):
        img_id = all_image_ids[text_i]
        target_img_idx = img_idx_map[img_id]
        scores = sim[:, text_i]                   # 所有 unique 图像对该文本的相似度
        ranked = scores.argsort(descending=True)
        for k in topk:
            if target_img_idx in ranked[:k].tolist():
                t2i_hits[k] += 1

    # Image-to-Text 检索：对每个图像，找最相似的文本
    i2t_hits = {k: 0 for k in topk}
    for img_idx, img_id in enumerate(unique_images):
        gt_text_indices = img2text[img_id]
        scores = sim[img_idx]                     # 该图像对所有文本的相似度
        ranked = scores.argsort(descending=True)
        for k in topk:
            if any(t in ranked[:k].tolist() for t in gt_text_indices):
                i2t_hits[k] += 1

    n_text = len(all_captions)
    n_img  = len(unique_images)
    metrics = {}
    for k in topk:
        metrics[f"t2i_R@{k}"] = t2i_hits[k] / n_text
        metrics[f"i2t_R@{k}"] = i2t_hits[k] / n_img
    return metrics


def save_checkpoint(path, model, criterion, optimizer, scheduler, tokenizer, epoch, metrics, args, best_val_r1=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "tokenizer_word2idx": tokenizer.word2idx,
            "metrics": metrics,
            "best_val_r1": best_val_r1,
            "args": vars(args),
        },
        path,
    )


def torch_load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def apply_resume_model_args(args, checkpoint):
    checkpoint_args = checkpoint.get("args", {})
    for name in ("text_encoder", "embed_dim", "image_width", "max_len", "num_heads", "num_layers", "text_dropout", "proj_head"):
        if name in checkpoint_args and getattr(args, name) != checkpoint_args[name]:
            print(
                f"resume overrides --{name.replace('_', '-')}={getattr(args, name)} "
                f"with checkpoint value {checkpoint_args[name]}"
            )
            setattr(args, name, checkpoint_args[name])


def restore_tokenizer(tokenizer, checkpoint):
    word2idx = checkpoint.get("tokenizer_word2idx")
    if not word2idx:
        return tokenizer
    tokenizer.word2idx = dict(word2idx)
    tokenizer.idx2word = [""] * len(tokenizer.word2idx)
    for word, idx in tokenizer.word2idx.items():
        tokenizer.idx2word[idx] = word
    return tokenizer


def load_training_state(checkpoint, model, criterion, optimizer, scheduler, device):
    model.load_state_dict(checkpoint["model"])
    criterion.load_state_dict(checkpoint["criterion"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])


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


class ModelEMA:
    """Exponential Moving Average of model parameters for more stable evaluation."""

    def __init__(self, model, decay=0.99, start_epoch=0):
        self.decay = decay
        self.start_epoch = start_epoch
        self.initialized = False
        self.shadow = copy.deepcopy(model).cpu()
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, epoch):
        if epoch < self.start_epoch:
            return
        model.eval()
        if not self.initialized:
            for p_ema, p_model in zip(self.shadow.parameters(), model.parameters()):
                p_ema.data.copy_(p_model.data.cpu())
            self.initialized = True
            model.train()
            return
        for p_ema, p_model in zip(self.shadow.parameters(), model.parameters()):
            p_ema.data.mul_(self.decay).add_(p_model.data.cpu(), alpha=1.0 - self.decay)
        model.train()

    def to(self, device):
        self.shadow = self.shadow.to(device)
        return self


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch_load_checkpoint(args.resume, map_location="cpu")
        apply_resume_model_args(args, resume_checkpoint)

    tokenizer, train_loader, val_loader, test_loader = build_dataloaders(args)
    if resume_checkpoint is not None:
        tokenizer = restore_tokenizer(tokenizer, resume_checkpoint)

    model = SigLIPModel(
        vocab_size=len(tokenizer),
        embed_dim=args.embed_dim,
        image_width=args.image_width,
        text_encoder=args.text_encoder,
        max_len=args.max_len,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        text_dropout=args.text_dropout,
        proj_head=args.proj_head,
    ).to(device)
    criterion = SigLIPLoss().to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * (len(train_loader) // args.grad_accum)
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(args.warmup_ratio * total_steps)
    warmup_steps = min(warmup_steps, total_steps)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps, args.lr, args.min_lr_ratio)
    print(f"total_steps={total_steps} warmup_steps={warmup_steps} min_lr={args.lr * args.min_lr_ratio:.2e}")

    print(f"device={device} vocab_size={len(tokenizer)} text_encoder={args.text_encoder}")
    start_epoch = 1
    best_val_r1 = -1.0

    ema = None
    if args.ema_decay > 0:
        ema = ModelEMA(model, decay=args.ema_decay, start_epoch=args.ema_start_epoch)
        print(f"EMA enabled with decay={args.ema_decay}, start_epoch={args.ema_start_epoch}")
    if args.resume:
        load_training_state(resume_checkpoint, model, criterion, optimizer, scheduler, device)
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_val_r1 = resume_checkpoint.get(
            "best_val_r1",
            resume_checkpoint.get("metrics", {}).get("t2i_R@1", best_val_r1),
        )
        print(
            f"resumed_from={args.resume} checkpoint_epoch={start_epoch - 1} "
            f"next_epoch={start_epoch} best_t2i_R@1={best_val_r1 * 100:.2f}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, scheduler, device, grad_accum=args.grad_accum)
        if ema is not None:
            ema.update(model, epoch)
        val_loss = evaluate_loss(model, criterion, val_loader, device)
        val_metrics = evaluate_retrieval(model, val_loader, device)
        val_r1 = val_metrics["t2i_R@1"]
        scale = criterion.logit_scale.exp().item()
        bias = criterion.logit_bias.item()
        metric_text = " ".join([f"{name}={value * 100:.2f}" for name, value in val_metrics.items()])
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"scale={scale:.2f} bias={bias:.2f} {metric_text}"
        )
        if val_r1 > best_val_r1:
            best_val_r1 = val_r1
            save_checkpoint(
                os.path.join(args.output_dir, "best_siglip.pt"),
                model,
                criterion,
                optimizer,
                scheduler,
                tokenizer,
                epoch,
                val_metrics,
                args,
                best_val_r1=best_val_r1,
            )
        save_checkpoint(
            os.path.join(args.output_dir, "latest_siglip.pt"),
            model,
            criterion,
            optimizer,
            scheduler,
            tokenizer,
            epoch,
            val_metrics,
            args,
            best_val_r1=best_val_r1,
        )

    final_model = model
    if ema is not None and ema.initialized:
        ema_model = ema.shadow.to(device)
        ema_test_metrics = evaluate_retrieval(ema_model, test_loader, device)
        ema_metric_text = " ".join([f"{name}={value * 100:.2f}" for name, value in ema_test_metrics.items()])
        print(f"EMA test: {ema_metric_text}")
    test_loss = evaluate_loss(final_model, criterion, test_loader, device)
    test_metrics = evaluate_retrieval(final_model, test_loader, device)
    metric_text = " ".join([f"{name}={value * 100:.2f}" for name, value in test_metrics.items()])
    print(f"final test_loss={test_loss:.4f} {metric_text}")


if __name__ == "__main__":
    main()
