"""
Zero-shot 跨模态相似度矩阵热力图
=================================
使用 HuggingFace 预训练的 google/siglip-base-patch16-224 模型，
或本地训练的 SigLIP 模型，从 Flickr8k 测试集随机抽取 N 对图文，
计算交叉余弦相似度矩阵，并应用 SigLIP 的缩放逻辑 logits = t * cos_sim + b，
绘制 Logits 热力图。

用法：
  # 方式一：使用 HuggingFace 预训练模型（需要网络下载）
  python visualize_heatmap.py --pretrained

  # 方式二：使用本地训练的模型（无需网络）
  python visualize_heatmap.py \
      --checkpoint outputs/exp11_wd005/best_siglip.pt \
      --data-dir Flickr8k

  # 指定参数
  python visualize_heatmap.py \
      --checkpoint outputs/exp11_wd005/best_siglip.pt \
      --data-dir Flickr8k \
      --n-pairs 8 \
      --output viz_outputs/siglip_logits_heatmap.png \
      --seed 42

  # 只画原始余弦相似度（不缩放）
  python visualize_heatmap.py --checkpoint ... --no-scale
"""

import argparse
import csv
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ── Matplotlib 全局设置 ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ═══════════════════════════════════════════════════════════════════
#  本地模型加载
# ═══════════════════════════════════════════════════════════════════

def load_local_model(checkpoint_path, device):
    """从本地 checkpoint 加载训练好的 SigLIP 模型。"""
    from models import SigLIPModel
    from utils import SimpleTokenizer, read_caption_file
    from loss import SigLIPLoss

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args_dict = ckpt.get("args", {})

    class _Args:
        pass
    args = _Args()
    for k, v in args_dict.items():
        setattr(args, k, v)

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

    model = SigLIPModel(
        vocab_size=len(tokenizer),
        embed_dim=getattr(args, "embed_dim", 256),
        image_width=getattr(args, "image_width", 32),
        text_encoder=getattr(args, "text_encoder", "transformer"),
        max_len=getattr(args, "max_len", 32),
        num_heads=getattr(args, "num_heads", 4),
        num_layers=getattr(args, "num_layers", 2),
        text_dropout=getattr(args, "text_dropout", 0.1),
        proj_head=getattr(args, "proj_head", "linear"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    criterion = SigLIPLoss().to(device)
    if "criterion" in ckpt:
        criterion.load_state_dict(ckpt["criterion"])

    img_transform = transforms.Compose([
        transforms.Resize((getattr(args, "image_size", 224),
                           getattr(args, "image_size", 224))),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    return model, tokenizer, criterion, img_transform


@torch.no_grad()
def encode_local(model, tokenizer, criterion, img_transform, images, captions, device):
    """用本地模型提取图文特征。"""
    img_embeds, txt_embeds = [], []
    for img, cap in zip(images, captions):
        img_t = img_transform(img).unsqueeze(0).to(device)
        ids_t = torch.tensor([tokenizer.encode(cap)], dtype=torch.long).to(device)
        ie, te = model(img_t, ids_t)
        img_embeds.append(ie)
        txt_embeds.append(te)

    img_embeds = F.normalize(torch.cat(img_embeds), dim=-1)
    txt_embeds = F.normalize(torch.cat(txt_embeds), dim=-1)
    logit_scale = criterion.logit_scale.exp().item()
    logit_bias = criterion.logit_bias.item()
    return img_embeds, txt_embeds, logit_scale, logit_bias


# ═══════════════════════════════════════════════════════════════════
#  HuggingFace 预训练模型加载
# ═══════════════════════════════════════════════════════════════════

def load_pretrained_model(device="cuda"):
    """加载 HuggingFace 预训练 SigLIP 模型。"""
    from transformers import AutoModel, AutoProcessor

    model_name = "google/siglip-base-patch16-224"
    print(f"Loading pretrained model: {model_name} ...")
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


@torch.no_grad()
def encode_pretrained(model, processor, images, captions, device, batch_size=16):
    """用预训练 SigLIP 提取图文特征。"""
    all_img_embeds, all_txt_embeds = [], []

    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i + batch_size]
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
        img_out = model.vision_model(**{k: v for k, v in inputs.items() if k.startswith("pixel")})
        img_emb = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out.last_hidden_state[:, 0, :]
        if hasattr(model, "visual_projection"):
            img_emb = model.visual_projection(img_emb)
        img_emb = F.normalize(img_emb, dim=-1)
        all_img_embeds.append(img_emb)

    for i in range(0, len(captions), batch_size):
        batch_txts = captions[i:i + batch_size]
        inputs = processor(text=batch_txts, return_tensors="pt", padding="max_length",
                           truncation=True, max_length=64).to(device)
        txt_out = model.text_model(**inputs)
        txt_emb = txt_out.pooler_output if (hasattr(txt_out, "pooler_output") and txt_out.pooler_output is not None) else txt_out.last_hidden_state[:, 0, :]
        if hasattr(model, "text_projection"):
            txt_emb = model.text_projection(txt_emb)
        txt_emb = F.normalize(txt_emb, dim=-1)
        all_txt_embeds.append(txt_emb)

    img_embeds = torch.cat(all_img_embeds)
    txt_embeds = torch.cat(all_txt_embeds)

    logit_scale = model.logit_scale.item()
    logit_bias = model.logit_bias.item()
    return img_embeds, txt_embeds, np.exp(logit_scale), logit_bias


# ═══════════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════════

def load_flickr8k_test(data_dir="Flickr8k", max_samples=None, seed=42):
    """从 Flickr8k 测试集加载图文对，每张图只取一个 caption。"""
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
    unique_rows = []
    for r in rows:
        if r["image_id"] not in seen:
            seen.add(r["image_id"])
            unique_rows.append(r)

    rng = random.Random(seed)
    rng.shuffle(unique_rows)
    if max_samples:
        unique_rows = unique_rows[:max_samples]

    images, captions, img_ids = [], [], []
    for r in unique_rows:
        img_path = os.path.join(image_root, r["image_id"])
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            captions.append(r["caption"])
            img_ids.append(r["image_id"])
        except FileNotFoundError:
            continue

    return images, captions, img_ids


# ═══════════════════════════════════════════════════════════════════
#  绘图
# ═══════════════════════════════════════════════════════════════════

def truncate_caption(text, max_len=35):
    if len(text) > max_len:
        return text[:max_len - 1] + "…"
    return text


def plot_heatmap(matrix, captions, img_ids, logit_scale, logit_bias,
                 use_siglip_scale=True, save_path=None, model_source="local"):
    """绘制 SigLIP Logits 或余弦相似度热力图。"""
    n = matrix.shape[0]

    x_labels = [f"Img {i}" for i in range(n)]
    y_labels = [truncate_caption(cap, 40) for cap in captions]

    if use_siglip_scale:
        cmap = "RdYlBu_r"
        fmt = ".2f"
        cbar_label = "Logits  (t · cos_sim + b)"
        title = (f"SigLIP Logits Matrix ({model_source})\n"
                 f"t = {logit_scale:.2f},  b = {logit_bias:.2f}")
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

    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=fmt,
        annot_kws={"size": 9, "fontweight": "bold"},
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar=True,
        cbar_kws={"label": cbar_label, "shrink": 0.8},
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        square=True,
    )

    # 对角线高亮
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                    edgecolor="#2ecc71", linewidth=2.5))

    ax.set_xlabel("Image", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("Text Caption", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)

    diag_mean = np.diag(matrix).mean()
    offdiag_mean = matrix[~np.eye(n, dtype=bool)].mean()
    diag_min = np.diag(matrix).min()
    offdiag_max = matrix[~np.eye(n, dtype=bool)].max()

    stats_text = (
        f"Diagonal (positive pairs): mean={diag_mean:.3f}, min={diag_min:.3f}\n"
        f"Off-diagonal (negative pairs): mean={offdiag_mean:.3f}, max={offdiag_max:.3f}\n"
        f"Margin: {diag_mean - offdiag_mean:.3f}"
    )
    fig.text(0.5, -0.02, stats_text, ha="center", fontsize=10,
             style="italic", color="#555555")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SigLIP Cross-modal Similarity Heatmap")
    parser.add_argument("--data-dir", default="Flickr8k", help="Flickr8k root directory")
    parser.add_argument("--n-pairs", type=int, default=10, help="Number of image-text pairs to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-scale", action="store_true",
                        help="Plot raw cosine similarity instead of SigLIP-scaled logits")
    parser.add_argument("--output", default="viz_outputs/siglip_logits_heatmap.png",
                        help="Output image path")

    # 两种模式二选一
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--pretrained", action="store_true",
                            help="Use HuggingFace pretrained google/siglip-base-patch16-224")
    mode_group.add_argument("--checkpoint", default="",
                            help="Path to local trained model checkpoint (best_siglip.pt)")

    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)

    # 1. 加载模型
    if args.pretrained:
        model, processor = load_pretrained_model(device)
        model_source = "Pretrained SigLIP"
    else:
        model, tokenizer, criterion, img_transform = load_local_model(args.checkpoint, device)
        model_source = "Local Trained SigLIP"

    # 2. 加载测试集图文对
    print(f"Loading {args.n_pairs} image-text pairs from Flickr8k test set ...")
    images, captions, img_ids = load_flickr8k_test(
        args.data_dir, max_samples=args.n_pairs, seed=args.seed
    )
    n = len(images)
    print(f"  Loaded {n} pairs")

    # 3. 提取特征
    print("Computing embeddings ...")
    if args.pretrained:
        img_embeds, txt_embeds, logit_scale, logit_bias = encode_pretrained(
            model, processor, images, captions, device
        )
    else:
        img_embeds, txt_embeds, logit_scale, logit_bias = encode_local(
            model, tokenizer, criterion, img_transform, images, captions, device
        )

    # 4. 计算余弦相似度矩阵
    cos_sim = (txt_embeds @ img_embeds.T).cpu().numpy()  # [N_text, N_img]

    # 5. 应用 SigLIP 缩放: logits = t * cos_sim + b
    logits_matrix = logit_scale * cos_sim + logit_bias
    print(f"  SigLIP params: t = {logit_scale:.2f},  b = {logit_bias:.4f}")

    # 6. 绘制热力图
    if not args.no_scale:
        plot_heatmap(
            logits_matrix, captions, img_ids,
            logit_scale, logit_bias,
            use_siglip_scale=True,
            save_path=args.output,
            model_source=model_source,
        )

    # 同时保存余弦相似度版本
    cos_output = args.output.replace(".png", "_cosine.png")
    plot_heatmap(
        cos_sim, captions, img_ids,
        logit_scale, logit_bias,
        use_siglip_scale=False,
        save_path=cos_output,
        model_source=model_source,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
