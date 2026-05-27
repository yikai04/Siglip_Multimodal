"""Generate Exp21 visualizations: heatmap, Top-K retrieval, Bad Cases."""
import argparse
import csv
import os
import random
import shutil
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import DistilBertTokenizerFast

from data_loader import Flickr8kDistilBERTDataset, build_siglip_transform
from models.dual_pretrained_v2_siglip import SigLIPDualPretrainedV2Model
from loss import SigLIPLoss

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args_dict = ckpt.get("args", {})

    class _Args:
        pass
    args = _Args()
    for k, v in args_dict.items():
        setattr(args, k, v)

    local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pretrained_distilbert")
    hf_id = "distilbert-base-uncased"
    tok_source = local_path if os.path.isdir(local_path) else hf_id
    tokenizer = DistilBertTokenizerFast.from_pretrained(tok_source)

    model = SigLIPDualPretrainedV2Model(
        embed_dim=args.embed_dim,
        num_unfrozen_vit_layers=args.num_unfrozen_vit_layers,
        num_unfrozen_db_layers=args.num_unfrozen_db_layers,
        text_proj_type=args.text_proj_type,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    criterion = SigLIPLoss().to(device)
    if "criterion" in ckpt:
        criterion.load_state_dict(ckpt["criterion"])

    return model, tokenizer, criterion, args


def load_test_pairs(data_dir, n_pairs=8, seed=42):
    test_file = os.path.join(data_dir, "test_captions.txt")
    img_dir = "Images" if os.path.isdir(os.path.join(data_dir, "Images")) else "images"
    image_root = os.path.join(data_dir, img_dir)
    rows = []
    with open(test_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            img_id = row[0].strip()
            caption = ",".join(row[1:]).strip()
            if img_id and caption:
                rows.append({"image_id": img_id, "caption": caption})
    seen = set()
    unique = []
    for r in rows:
        if r["image_id"] not in seen:
            seen.add(r["image_id"])
            unique.append(r)
    rng = random.Random(seed)
    rng.shuffle(unique)
    unique = unique[:n_pairs]
    images, captions, img_ids = [], [], []
    for r in unique:
        img_path = os.path.join(image_root, r["image_id"])
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            captions.append(r["caption"])
            img_ids.append(r["image_id"])
        except FileNotFoundError:
            continue
    return images, captions, img_ids


@torch.no_grad()
def encode_for_heatmap(model, tokenizer, criterion, images, captions, device):
    img_transform = build_siglip_transform(224, train=False)
    img_embeds, txt_embeds = [], []
    for img, cap in zip(images, captions):
        img_t = img_transform(img).unsqueeze(0).to(device)
        encoded = tokenizer(cap, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        ids_t = encoded["input_ids"].to(device)
        mask_t = encoded["attention_mask"].to(device)
        ie, te = model(img_t, ids_t, mask_t)
        img_embeds.append(ie)
        txt_embeds.append(te)
    img_embeds = F.normalize(torch.cat(img_embeds), dim=-1)
    txt_embeds = F.normalize(torch.cat(txt_embeds), dim=-1)
    logit_scale = criterion.logit_scale.exp().item()
    logit_bias = criterion.logit_bias.item()
    return img_embeds, txt_embeds, logit_scale, logit_bias


def truncate_caption(text, max_len=35):
    return text[:max_len - 1] + "..." if len(text) > max_len else text


def plot_heatmap(matrix, captions, img_ids, logit_scale, logit_bias,
                 use_siglip_scale=True, save_path=None, model_source="Exp21"):
    n = matrix.shape[0]
    x_labels = [f"Img {i}" for i in range(n)]
    y_labels = [truncate_caption(cap, 40) for cap in captions]
    if use_siglip_scale:
        cmap = "RdYlBu_r"
        fmt = ".2f"
        cbar_label = "Logits  (t · cos_sim + b)"
        title = f"SigLIP Logits Matrix ({model_source})\nt = {logit_scale:.2f},  b = {logit_bias:.2f}"
    else:
        cmap = "coolwarm"
        fmt = ".3f"
        cbar_label = "Cosine Similarity"
        title = f"Cosine Similarity Matrix ({model_source})"
    fig, ax = plt.subplots(figsize=(max(10, n * 1.4), max(8, n * 0.95)))
    if use_siglip_scale:
        diag_vals = np.diag(matrix)
        offdiag = matrix[~np.eye(n, dtype=bool)]
        vmin = min(offdiag.min(), diag_vals.min()) - 0.5
        vmax = diag_vals.max() + 0.5
    else:
        vmin, vmax = None, None
    sns.heatmap(matrix, ax=ax, cmap=cmap, annot=True, fmt=fmt,
                annot_kws={"size": 9, "fontweight": "bold"},
                xticklabels=x_labels, yticklabels=y_labels,
                cbar=True, cbar_kws={"label": cbar_label, "shrink": 0.8},
                vmin=vmin, vmax=vmax, linewidths=0.5, linecolor="white", square=True)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="#2ecc71", linewidth=2.5))
    ax.set_xlabel("Image", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("Text Caption", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    diag_mean = np.diag(matrix).mean()
    offdiag_mean = matrix[~np.eye(n, dtype=bool)].mean()
    stats_text = f"Diagonal mean={diag_mean:.3f}, Off-diagonal mean={offdiag_mean:.3f}, Margin={diag_mean - offdiag_mean:.3f}"
    fig.text(0.5, -0.02, stats_text, ha="center", fontsize=10, style="italic", color="#555555")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    plt.close(fig)


def build_test_dataset(data_dir, tokenizer, image_size=224, max_len=64):
    img_dir = "Images" if os.path.isdir(os.path.join(data_dir, "Images")) else "images"
    return Flickr8kDistilBERTDataset(
        image_root=os.path.join(data_dir, img_dir),
        captions_file=os.path.join(data_dir, "test_captions.txt"),
        tokenizer=tokenizer,
        transform=build_siglip_transform(image_size, train=False),
        max_len=max_len,
    )


@torch.no_grad()
def compute_all_embeddings(model, dataset, device, batch_size=128):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_img_embeds, all_txt_embeds, all_img_ids, all_captions = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        img_emb, txt_emb = model(images, input_ids, attention_mask)
        all_img_embeds.append(img_emb)
        all_txt_embeds.append(txt_emb)
        all_img_ids.extend(batch["image_id"])
        all_captions.extend(batch["caption"])
    img_embeds = F.normalize(torch.cat(all_img_embeds), dim=-1)
    txt_embeds = F.normalize(torch.cat(all_txt_embeds), dim=-1)
    return img_embeds, txt_embeds, all_img_ids, all_captions


def get_unique_images(img_embeds, all_img_ids):
    img2indices = defaultdict(list)
    for idx, img_id in enumerate(all_img_ids):
        img2indices[img_id].append(idx)
    unique_ids = list(img2indices.keys())
    unique_embeds = []
    for img_id in unique_ids:
        indices = img2indices[img_id]
        pooled = img_embeds[indices].mean(dim=0)
        unique_embeds.append(F.normalize(pooled, dim=-1))
    unique_embeds = torch.stack(unique_embeds)
    img_idx_map = {img_id: i for i, img_id in enumerate(unique_ids)}
    return unique_embeds, unique_ids, img2indices, img_idx_map


def retrieve_topk(query_caption, model, tokenizer, unique_img_embeds, unique_img_ids, device, topk=5, max_len=64):
    encoded = tokenizer(query_caption, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    txt_emb = model.encode_text(input_ids, attention_mask)
    txt_emb = F.normalize(txt_emb, dim=-1)
    scores = (txt_emb @ unique_img_embeds.T).squeeze(0)
    topk_scores, topk_indices = scores.topk(topk)
    results = []
    for rank, (score, idx) in enumerate(zip(topk_scores.tolist(), topk_indices.tolist()), 1):
        results.append({"rank": rank, "image_id": unique_img_ids[idx], "score": score})
    return results


def plot_topk_results(query_caption, results, image_root, gt_image_id=None, save_path=None, title_prefix=""):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 8))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        img_path = os.path.join(image_root, r["image_id"])
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "Image\nNot Found", ha="center", va="center", fontsize=12, color="gray")
        is_gt = (r["image_id"] == gt_image_id)
        if is_gt:
            for spine in ax.spines.values():
                spine.set_edgecolor("#2ecc71")
                spine.set_linewidth(4)
                spine.set_visible(True)
        label = f"Rank {r['rank']}"
        if is_gt:
            label += "  [GT]"
        ax.set_title(f"{label}\nscore = {r['score']:.4f}", fontsize=16,
                     color="#2ecc71" if is_gt else "#333333",
                     fontweight="bold" if is_gt else "normal")
        ax.axis("off")
    main_title = f"{title_prefix}{query_caption}"
    if len(main_title) > 80:
        main_title = main_title[:77] + "..."
    fig.suptitle(main_title, fontsize=16, fontweight="bold", y=0.98, wrap=True)
    if gt_image_id:
        legend_elements = [mpatches.Patch(facecolor="none", edgecolor="#2ecc71", linewidth=2, label="Ground Truth")]
        fig.legend(handles=legend_elements, loc="lower center", ncol=1, fontsize=10, frameon=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[Saved] {save_path}")
    plt.close(fig)


def find_bad_cases(txt_embeds, all_img_ids, all_captions, unique_img_embeds, unique_img_ids, img_idx_map,
                   rank_threshold=10, num_cases=3):
    img2cap_indices = defaultdict(list)
    for idx, img_id in enumerate(all_img_ids):
        img2cap_indices[img_id].append(idx)
    bad_cases = []
    for img_id in unique_img_ids:
        cap_indices = img2cap_indices[img_id]
        target_img_idx = img_idx_map[img_id]
        for ci in cap_indices:
            txt_emb = txt_embeds[ci].unsqueeze(0)
            scores = (txt_emb @ unique_img_embeds.T).squeeze(0)
            ranked = scores.argsort(descending=True)
            rank = (ranked == target_img_idx).nonzero(as_tuple=True)[0].item() + 1
            if rank > rank_threshold:
                bad_cases.append({"caption": all_captions[ci], "gt_image_id": img_id, "gt_rank": rank})
    bad_cases.sort(key=lambda x: x["gt_rank"], reverse=True)
    return bad_cases[:num_cases]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="Flickr8k")
    parser.add_argument("--output-dir", default="viz_outputs")
    parser.add_argument("--n-pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num-bad", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, criterion, ckpt_args = load_model(args.checkpoint, device)
    image_size = getattr(ckpt_args, "image_size", 224)
    max_len = getattr(ckpt_args, "max_len", 64)
    img_dir = "Images" if os.path.isdir(os.path.join(args.data_dir, "Images")) else "images"
    image_root = os.path.join(args.data_dir, img_dir)

    # 1. Heatmap
    print("Generating heatmap...")
    images, captions, img_ids = load_test_pairs(args.data_dir, args.n_pairs, args.seed)
    img_embeds, txt_embeds, logit_scale, logit_bias = encode_for_heatmap(model, tokenizer, criterion, images, captions, device)
    cos_sim = (txt_embeds @ img_embeds.T).cpu().numpy()
    logits_matrix = logit_scale * cos_sim + logit_bias
    plot_heatmap(logits_matrix, captions, img_ids, logit_scale, logit_bias,
                 use_siglip_scale=True, save_path=os.path.join(args.output_dir, "exp21_logits_heatmap.png"),
                 model_source="Exp21 Dual Pretrained V2")
    plot_heatmap(cos_sim, captions, img_ids, logit_scale, logit_bias,
                 use_siglip_scale=False, save_path=os.path.join(args.output_dir, "exp21_logits_heatmap_cosine.png"),
                 model_source="Exp21 Dual Pretrained V2")

    # 2. Top-K & Bad Cases
    print("Computing embeddings for Top-K...")
    dataset = build_test_dataset(args.data_dir, tokenizer, image_size, max_len)
    img_embeds_all, txt_embeds_all, all_img_ids, all_captions = compute_all_embeddings(model, dataset, device)
    unique_img_embeds, unique_img_ids, img2indices, img_idx_map = get_unique_images(img_embeds_all, all_img_ids)
    print(f"  {len(unique_img_ids)} unique images, {len(all_captions)} captions")

    query = "A little girl covered in paint sits in front of a painted rainbow"
    print(f"Top-K query: {query}")
    results = retrieve_topk(query, model, tokenizer, unique_img_embeds, unique_img_ids, device, args.topk, max_len)
    for r in results:
        print(f"  Rank {r['rank']}: {r['image_id']}  score={r['score']:.4f}")
    plot_topk_results(query, results, image_root, save_path=os.path.join(args.output_dir, "exp21_topk_query.png"))

    print("Finding bad cases...")
    bad_cases = find_bad_cases(txt_embeds_all, all_img_ids, all_captions, unique_img_embeds, unique_img_ids, img_idx_map)
    if bad_cases:
        for i, bc in enumerate(bad_cases[:args.num_bad]):
            print(f"  [{i+1}] GT rank={bc['gt_rank']} | \"{bc['caption'][:60]}\"")
            results = retrieve_topk(bc["caption"], model, tokenizer, unique_img_embeds, unique_img_ids, device, args.topk, max_len)
            plot_topk_results(bc["caption"], results, image_root, gt_image_id=bc["gt_image_id"],
                             save_path=os.path.join(args.output_dir, f"exp21_badcase_{i+1}.png"),
                             title_prefix=f"[Bad Case #{i+1}  GT-Rank={bc['gt_rank']}]  ")
            gt_src = os.path.join(image_root, bc["gt_image_id"])
            gt_dst = os.path.join(args.output_dir, f"exp21_gt_badcase_{i+1}.jpg")
            if os.path.exists(gt_src):
                shutil.copy2(gt_src, gt_dst)
                print(f"  GT image copied: {gt_dst}")
    else:
        print("  No bad cases found!")

    print("\nDone!")


if __name__ == "__main__":
    main()