"""Evaluate Exp19 best checkpoint on test set."""
import os
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms as T

from data_loader import Flickr8kDataset
from loss import SigLIPLoss
from models.pretrained_siglip import SigLIPPretrainedModel
from utils import SimpleTokenizer, read_caption_file

SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD = [0.5, 0.5, 0.5]

def build_siglip_transform(image_size=224):
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
    ])

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

@torch.no_grad()
def evaluate_loss(model, criterion, loader, device):
    model.eval()
    criterion.eval()
    total_loss = 0
    total_count = 0
    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        image_embeds, text_embeds = model(images, input_ids)
        loss = criterion(image_embeds, text_embeds)
        total_loss += loss.item() * images.size(0)
        total_count += images.size(0)
    return total_loss / total_count

def main():
    device = torch.device("cuda")
    ckpt_path = "outputs/exp19_pretrained_vit/best_siglip.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args_dict = ckpt.get("args", {})

    class _Args:
        pass
    args = _Args()
    for k, v in args_dict.items():
        setattr(args, k, v)

    # Rebuild tokenizer
    all_rows = read_caption_file(os.path.join(args.data_dir, "captions.txt"))
    tokenizer = SimpleTokenizer(
        (row["caption"] for row in all_rows),
        min_freq=getattr(args, "min_freq", 2),
        max_len=getattr(args, "max_len", 32),
    )
    w2i = ckpt.get("tokenizer_word2idx")
    if w2i:
        tokenizer.word2idx = dict(w2i)
        tokenizer.idx2word = [""] * len(tokenizer.word2idx)
        for w, idx in tokenizer.word2idx.items():
            tokenizer.idx2word[idx] = w

    # Rebuild model
    model = SigLIPPretrainedModel(
        vocab_size=len(tokenizer),
        embed_dim=getattr(args, "embed_dim", 768),
        max_len=getattr(args, "max_len", 32),
        num_heads=getattr(args, "num_heads", 12),
        num_layers=getattr(args, "num_layers", 2),
        text_dropout=getattr(args, "text_dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    criterion = SigLIPLoss().to(device)
    criterion.load_state_dict(ckpt["criterion"])

    # Build test loader
    image_root = os.path.join(args.data_dir, "images")
    test_dataset = Flickr8kDataset(
        image_root=image_root,
        captions_file=os.path.join(args.data_dir, "test_captions.txt"),
        tokenizer=tokenizer,
        transform=build_siglip_transform(getattr(args, "image_size", 224)),
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # Evaluate
    print("Evaluating on test set...")
    test_loss = evaluate_loss(model, criterion, test_loader, device)
    test_metrics = evaluate_retrieval(model, test_loader, device)

    scale = criterion.logit_scale.exp().item()
    bias = criterion.logit_bias.item()
    print(f"test_loss={test_loss:.4f} scale={scale:.2f} bias={bias:.2f}")
    for k, v in test_metrics.items():
        print(f"  {k}: {v*100:.2f}%")

if __name__ == "__main__":
    main()
