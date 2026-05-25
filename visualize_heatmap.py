"""
Zero-shot 跨模态相似度矩阵热力图
=================================
使用 HuggingFace 预训练的 google/siglip-base-patch16-224 模型，
从 Flickr8k 测试集随机抽取 N 对图文，计算交叉余弦相似度矩阵，
并应用 SigLIP 的缩放逻辑 logits = t * cos_sim + b，绘制 Logits 热力图。

用法：
  # 默认 10 对图文，输出到 viz_outputs/
  python visualize_heatmap.py

  # 指定参数
  python visualize_heatmap.py \
      --n-pairs 8 \
      --output viz_outputs/siglip_logits_heatmap.png \
      --seed 42

  # 只画原始余弦相似度（不缩放）
  python visualize_heatmap.py --no-scale
"""

import argparse
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


def load_siglip_pretrained(device="cuda"):
    """加载 HuggingFace 预训练 SigLIP 模型。"""
    from transformers import AutoModel, AutoProcessor

    model_name = "google/siglip-base-patch16-224"
    print(f"Loading pretrained model: {model_name} ...")
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def load_flickr8k_test(data_dir="Flickr8k", max_samples=None, seed=42):
    """从 Flickr8k 测试集加载图文对。

    Returns:
        images:  List[PIL.Image]
        captions: List[str]
        img_ids:  List[str]
    """
    import csv

    test_file = os.path.join(data_dir, "test_captions.txt")
    image_root = os.path.join(data_dir, "Images")

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

    # 按 image_id 去重：每张图只取第一个 caption（保证 N 对不重复图）
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


@torch.no_grad()
def compute_embeddings(model, processor, images, captions, device, batch_size=16):
    """用预训练 SigLIP 提取图文特征。"""
    all_img_embeds, all_txt_embeds = [], []

    # 图像 embedding
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i + batch_size]
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
        img_out = model.vision_model(**{k: v for k, v in inputs.items() if k.startswith("pixel")})
        img_emb = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out.last_hidden_state[:, 0, :]
        if hasattr(model, "visual_projection"):
            img_emb = model.visual_projection(img_emb)
        img_emb = F.normalize(img_emb, dim=-1)
        all_img_embeds.append(img_emb)

    # 文本 embedding
    for i in range(0, len(captions), batch_size):
        batch_txts = captions[i:i + batch_size]
        inputs = processor(text=batch_txts, return_tensors="pt", padding="max_length",
                           truncation=True, max_length=64).to(device)
        txt_out = model.text_model(**inputs)
        txt_emb = txt_out.pooler_output if hasattr(txt_out, "pooler_output") and txt_out.pooler_output is not None else txt_out.last_hidden_state[:, 0, :]
        if hasattr(model, "text_projection"):
            txt_emb = model.text_projection(txt_emb)
        txt_emb = F.normalize(txt_emb, dim=-1)
        all_txt_embeds.append(txt_emb)

    img_embeds = torch.cat(all_img_embeds)
    txt_embeds = torch.cat(all_txt_embeds)
    return img_embeds, txt_embeds


def truncate_caption(text, max_len=35):
    """截断过长文本，用于轴标签。"""
    if len(text) > max_len:
        return text[:max_len - 1] + "…"
    return text


def plot_heatmap(matrix, captions, img_ids, logit_scale, logit_bias,
                 use_siglip_scale=True, save_path=None):
    """绘制 SigLIP Logits 或余弦相似度热力图。

    Args:
        matrix: [N_text, N_img] 的 numpy 数组
        captions: 文本列表
        img_ids: 图像 ID 列表
        logit_scale: float, temperature 参数的标量值
        logit_bias: float, bias 参数的标量值
        use_siglip_scale: 是否应用 SigLIP 缩放
        save_path: 保存路径
    """
    n = matrix.shape[0]

    # 准备标签
    x_labels = [f"Img {i}" for i in range(n)]
    y_labels = [truncate_caption(cap, 40) for cap in captions]

    # 选 colormap：对角线（正值）vs 非对角线（负值）需要强烈对比
    if use_siglip_scale:
        cmap = "RdYlBu_r"
        fmt = ".2f"
        cbar_label = "Logits  (t·cos_sim + b)"
        title = (f"SigLIP Zero-shot Logits Matrix\n"
                 f"t = exp({logit_scale:.2f}) = {np.exp(logit_scale):.2f},  "
                 f"b = {logit_bias:.2f}")
    else:
        cmap = "coolwarm"
        fmt = ".3f"
        cbar_label = "Cosine Similarity"
        title = "Cosine Similarity Matrix (pre-SigLIP scaling)"

    fig, ax = plt.subplots(figsize=(max(10, n * 1.3), max(8, n * 0.9)))

    # 设置 vmin/vmax 让对角线和非对角线对比强烈
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

    # 对角线高亮：添加绿色边框
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                    edgecolor="#2ecc71", linewidth=2.5))

    ax.set_xlabel("Image", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("Text Caption", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Y 轴标签左对齐
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)

    # 统计注释
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


def main():
    parser = argparse.ArgumentParser(description="Zero-shot SigLIP Cross-modal Similarity Heatmap")
    parser.add_argument("--data-dir", default="Flickr8k", help="Flickr8k root directory")
    parser.add_argument("--n-pairs", type=int, default=10, help="Number of image-text pairs to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--no-scale", action="store_true",
                        help="Plot raw cosine similarity instead of SigLIP-scaled logits")
    parser.add_argument("--output", default="viz_outputs/siglip_logits_heatmap.png",
                        help="Output image path")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)

    # 1. 加载预训练模型
    model, processor = load_siglip_pretrained(device)

    # 2. 加载测试集图文对
    print(f"Loading {args.n_pairs} image-text pairs from Flickr8k test set ...")
    images, captions, img_ids = load_flickr8k_test(
        args.data_dir, max_samples=args.n_pairs, seed=args.seed
    )
    n = len(images)
    print(f"  Loaded {n} pairs")

    # 3. 提取特征
    print("Computing embeddings ...")
    img_embeds, txt_embeds = compute_embeddings(model, processor, images, captions, device)

    # 4. 计算余弦相似度矩阵
    cos_sim = (txt_embeds @ img_embeds.T).cpu().numpy()  # [N_text, N_img]

    # 5. 获取 SigLIP 可学习参数
    logit_scale_val = model.logit_scale.item()
    logit_bias_val = model.logit_bias.item()
    t = np.exp(logit_scale_val)
    b = logit_bias_val
    print(f"  SigLIP params: t = exp({logit_scale_val:.4f}) = {t:.2f},  b = {b:.4f}")

    # 6. 应用 SigLIP 缩放: logits = t * cos_sim + b
    logits_matrix = t * cos_sim + b

    # 7. 绘制热力图
    if args.no_scale:
        plot_heatmap(
            cos_sim, captions, img_ids,
            logit_scale_val, logit_bias_val,
            use_siglip_scale=False,
            save_path=args.output.replace(".png", "_cosine.png"),
        )
    else:
        plot_heatmap(
            logits_matrix, captions, img_ids,
            logit_scale_val, logit_bias_val,
            use_siglip_scale=True,
            save_path=args.output,
        )

    # 同时保存一份余弦相似度的
    cos_output = args.output.replace(".png", "_cosine.png")
    plot_heatmap(
        cos_sim, captions, img_ids,
        logit_scale_val, logit_bias_val,
        use_siglip_scale=False,
        save_path=cos_output,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
