"""Evaluate Exp20 best checkpoint on test set."""
import os

import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast
from tqdm import tqdm

from data_loader import Flickr8kDistilBERTDataset, build_siglip_transform
from loss import SigLIPLoss
from models.dual_pretrained_siglip import SigLIPDualPretrainedModel
from utils import AverageMeter


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


@torch.no_grad()
def evaluate_loss(model, criterion, loader, device):
    model.eval()
    criterion.eval()
    meter = AverageMeter()
    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image_embeds, text_embeds = model(images, input_ids, attention_mask)
        loss = criterion(image_embeds, text_embeds)
        meter.update(loss.item(), images.size(0))
    return meter.avg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "outputs/exp20_dual_pretrained/best_siglip.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args_dict = ckpt.get("args", {})

    class _Args:
        pass
    args = _Args()
    for k, v in args_dict.items():
        setattr(args, k, v)

    # Rebuild tokenizer
    local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pretrained_distilbert")
    hf_id = "distilbert-base-uncased"
    tok_source = local_path if os.path.isdir(local_path) else hf_id
    tokenizer = DistilBertTokenizerFast.from_pretrained(tok_source)

    # Rebuild model
    model = SigLIPDualPretrainedModel(
        embed_dim=args.embed_dim,
        num_unfrozen_vit_layers=args.num_unfrozen_vit_layers,
        text_proj_type=args.text_proj_type,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    criterion = SigLIPLoss().to(device)
    criterion.load_state_dict(ckpt["criterion"])

    # Build test loader
    img_dir = "Images" if os.path.isdir(os.path.join(args.data_dir, "Images")) else "images"
    image_root = os.path.join(args.data_dir, img_dir)
    test_dataset = Flickr8kDistilBERTDataset(
        image_root=image_root,
        captions_file=os.path.join(args.data_dir, "test_captions.txt"),
        tokenizer=tokenizer,
        transform=build_siglip_transform(args.image_size, train=False),
        max_len=args.max_len,
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # Evaluate
    print("Evaluating Exp20 on test set...")
    test_loss = evaluate_loss(model, criterion, test_loader, device)
    test_metrics = evaluate_retrieval(model, test_loader, device)

    scale = criterion.logit_scale.exp().item()
    bias = criterion.logit_bias.item()
    print(f"test_loss={test_loss:.4f} scale={scale:.2f} bias={bias:.2f}")
    for k, v in test_metrics.items():
        print(f"  {k}: {v*100:.2f}%")


if __name__ == "__main__":
    main()