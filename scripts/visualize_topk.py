"""
Top-K 检索结果可视化与 Bad Case 分析
=====================================
功能：
  1. 输入一段 Caption，检索 Top-K 最相似图像并可视化
  2. 自动遍历测试集，挖掘 Bad Case（GT 排在 Rank-10 外）
  3. 对 Bad Case 同样绘制 Top-K 检索图

用法：
  # 指定 Caption 检索 Top-5
  python visualize_topk.py \
      --checkpoint outputs/exp11_wd005/best_siglip.pt \
      --data-dir Flickr8k \
      --caption "A little girl covered in paint sits in front of a painted rainbow" \
      --topk 5

  # 自动挖掘 Bad Case 并可视化
  python visualize_topk.py \
      --checkpoint outputs/exp11_wd005/best_siglip.pt \
      --data-dir Flickr8k \
      --find-bad-cases \
      --num-bad 3 \
      --bad-rank-threshold 10

  # 两者同时执行
  python visualize_topk.py \
      --checkpoint outputs/exp11_wd005/best_siglip.pt \
      --data-dir Flickr8k \
      --caption "A dog runs through the grass" \
      --find-bad-cases
"""

import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data_loader import Flickr8kDataset, build_transform
from models import SigLIPModel
from utils import SimpleTokenizer, read_caption_file


# ── Matplotlib 全局设置 ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def load_model_and_tokenizer(checkpoint_path, device):
    """从 checkpoint 恢复模型和 tokenizer。"""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args_dict = ckpt.get("args", {})

    class _Args:
        pass
    args = _Args()
    for k, v in args_dict.items():
        setattr(args, k, v)

    # 重建 tokenizer
    all_rows = read_caption_file(os.path.join(args.data_dir, "captions.txt"))
    tokenizer = SimpleTokenizer(
        (row["caption"] for row in all_rows),
        min_freq=getattr(args, "min_freq", 2),
        max_len=getattr(args, "max_len", 32),
    )
    # 恢复 checkpoint 保存的词表
    w2i = ckpt.get("tokenizer_word2idx")
    if w2i:
        tokenizer.word2idx = dict(w2i)
        tokenizer.idx2word = [""] * len(tokenizer.word2idx)
        for w, idx in tokenizer.word2idx.items():
            tokenizer.idx2word[idx] = w

    # 重建模型
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

    return model, tokenizer, args


def build_test_dataset(data_dir, tokenizer, image_size=224):
    """构建测试集 Dataset。"""
    img_dir = "Images" if os.path.isdir(os.path.join(data_dir, "Images")) else "images"
    return Flickr8kDataset(
        image_root=os.path.join(data_dir, img_dir),
        captions_file=os.path.join(data_dir, "test_captions.txt"),
        tokenizer=tokenizer,
        transform=build_transform(image_size, train=False),
    )


@torch.no_grad()
def compute_all_embeddings(model, dataset, device, batch_size=128):
    """计算测试集所有样本的图像/文本 embedding。"""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_img_embeds, all_txt_embeds = [], []
    all_img_ids, all_captions = [], []

    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        img_emb, txt_emb = model(images, input_ids)
        all_img_embeds.append(img_emb)
        all_txt_embeds.append(txt_emb)
        all_img_ids.extend(batch["image_id"])
        all_captions.extend(batch["caption"])

    img_embeds = F.normalize(torch.cat(all_img_embeds), dim=-1)
    txt_embeds = F.normalize(torch.cat(all_txt_embeds), dim=-1)

    return img_embeds, txt_embeds, all_img_ids, all_captions


def get_unique_images(img_embeds, all_img_ids):
    """按 image_id 分组，聚合同一图像的多个 embedding。"""
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


def retrieve_topk_for_text(query_caption, model, tokenizer, unique_img_embeds,
                           unique_img_ids, device, topk=5):
    """给定一段文本，检索 Top-K 最相似的图像。"""
    input_ids = torch.tensor([tokenizer.encode(query_caption)], dtype=torch.long).to(device)
    txt_emb = model.text_encoder(input_ids)
    txt_emb = F.normalize(txt_emb, dim=-1)

    scores = (txt_emb @ unique_img_embeds.T).squeeze(0)
    topk_scores, topk_indices = scores.topk(topk)

    results = []
    for rank, (score, idx) in enumerate(zip(topk_scores.tolist(), topk_indices.tolist()), 1):
        results.append({
            "rank": rank,
            "image_id": unique_img_ids[idx],
            "score": score,
        })
    return results


def plot_topk_results(query_caption, results, image_root, gt_image_id=None,
                      save_path=None, title_prefix=""):
    """绘制 Top-K 检索结果。

    Args:
        query_caption: 查询文本
        results: retrieve_topk_for_text 返回的列表
        image_root: 图像目录
        gt_image_id: 正确的图像 ID（如果有的话）
        save_path: 保存路径
        title_prefix: 标题前缀
    """
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
            ax.text(0.5, 0.5, "Image\nNot Found", ha="center", va="center",
                    fontsize=12, color="gray")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        is_gt = (r["image_id"] == gt_image_id)
        if is_gt:
            for spine in ax.spines.values():
                spine.set_edgecolor("#2ecc71")
                spine.set_linewidth(4)
                spine.set_visible(True)

        label = f"Rank {r['rank']}"
        if is_gt:
            label += "  [GT]"
        ax.set_title(f"{label}\nscore = {r['score']:.4f}",
                     fontsize=16,
                     color="#2ecc71" if is_gt else "#333333",
                     fontweight="bold" if is_gt else "normal")
        ax.axis("off")

    main_title = f"{title_prefix}{query_caption}"
    if len(main_title) > 80:
        main_title = main_title[:77] + "..."
    fig.suptitle(main_title, fontsize=16, fontweight="bold", y=0.98, wrap=True)

    legend_elements = []
    if gt_image_id:
        legend_elements.append(mpatches.Patch(facecolor="none", edgecolor="#2ecc71",
                                               linewidth=2, label="Ground Truth"))
    if legend_elements:
        fig.legend(handles=legend_elements, loc="lower center", ncol=1,
                   fontsize=10, frameon=True, fancybox=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[Saved] {save_path}")
    plt.close(fig)


def find_bad_cases(model, tokenizer, txt_embeds, all_img_ids, all_captions,
                   unique_img_embeds, unique_img_ids, img_idx_map,
                   device, rank_threshold=10, num_cases=3):
    """遍历测试集，找出 Ground Truth 图像排在 rank_threshold 之外的 Bad Case。

    Returns:
        List of dicts: [{"caption": str, "gt_image_id": str, "gt_rank": int}, ...]
    """
    # 建立每个 image_id 对应的 caption indices
    img2cap_indices = defaultdict(list)
    for idx, img_id in enumerate(all_img_ids):
        img2cap_indices[img_id].append(idx)

    bad_cases = []

    # 对每个 unique image，取其对应的 captions
    for img_id in unique_img_ids:
        cap_indices = img2cap_indices[img_id]
        target_img_idx = img_idx_map[img_id]

        for ci in cap_indices:
            txt_emb = txt_embeds[ci].unsqueeze(0)
            scores = (txt_emb @ unique_img_embeds.T).squeeze(0)
            ranked = scores.argsort(descending=True)
            rank = (ranked == target_img_idx).nonzero(as_tuple=True)[0].item() + 1

            if rank > rank_threshold:
                bad_cases.append({
                    "caption": all_captions[ci],
                    "gt_image_id": img_id,
                    "gt_rank": rank,
                })

    # 按 gt_rank 从大到小排序，取最严重的
    bad_cases.sort(key=lambda x: x["gt_rank"], reverse=True)
    return bad_cases[:num_cases]


def main():
    parser = argparse.ArgumentParser(description="Top-K Retrieval Visualization & Bad Case Analysis")
    parser.add_argument("--checkpoint", required=True, help="Path to best_siglip.pt")
    parser.add_argument("--data-dir", default="Flickr8k", help="Flickr8k root directory")
    parser.add_argument("--caption", default="", help="Query caption for Top-K retrieval")
    parser.add_argument("--topk", type=int, default=5, help="Number of top results to show")
    parser.add_argument("--find-bad-cases", action="store_true", help="Find and visualize bad cases")
    parser.add_argument("--num-bad", type=int, default=3, help="Number of bad cases to show")
    parser.add_argument("--bad-rank-threshold", type=int, default=10,
                        help="GT rank above this is considered a bad case")
    parser.add_argument("--output-dir", default="viz_outputs", help="Output directory for figures")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading model from {args.checkpoint} ...")
    model, tokenizer, ckpt_args = load_model_and_tokenizer(args.checkpoint, device)

    image_size = getattr(ckpt_args, "image_size", 224)
    # 兼容大小写目录名
    img_dir = "Images" if os.path.isdir(os.path.join(args.data_dir, "Images")) else "images"
    image_root = os.path.join(args.data_dir, img_dir)

    print("Building test dataset & computing embeddings ...")
    dataset = build_test_dataset(args.data_dir, tokenizer, image_size)
    img_embeds, txt_embeds, all_img_ids, all_captions = compute_all_embeddings(
        model, dataset, device
    )
    unique_img_embeds, unique_img_ids, img2indices, img_idx_map = get_unique_images(
        img_embeds, all_img_ids
    )
    print(f"  {len(unique_img_ids)} unique images, {len(all_captions)} captions")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Task 1: 指定 Caption 检索 ──────────────────────────────────────────
    if args.caption:
        print(f"\nQuery: \"{args.caption}\"")
        results = retrieve_topk_for_text(
            args.caption, model, tokenizer, unique_img_embeds,
            unique_img_ids, device, topk=args.topk,
        )
        for r in results:
            print(f"  Rank {r['rank']}: {r['image_id']}  score={r['score']:.4f}")

        save_path = os.path.join(args.output_dir, "topk_query.png")
        plot_topk_results(args.caption, results, image_root, save_path=save_path)

    # ── Task 2: Bad Case 挖掘 ──────────────────────────────────────────────
    if args.find_bad_cases:
        print(f"\nFinding bad cases (GT rank > {args.bad_rank_threshold}) ...")
        bad_cases = find_bad_cases(
            model, tokenizer, txt_embeds, all_img_ids, all_captions,
            unique_img_embeds, unique_img_ids, img_idx_map,
            device, rank_threshold=args.bad_rank_threshold, num_cases=args.num_bad,
        )
        if not bad_cases:
            print("  No bad cases found! Model performs well on all test queries.")
        else:
            print(f"  Found {len(bad_cases)} bad cases:")
            for i, bc in enumerate(bad_cases):
                print(f"  [{i+1}] GT rank={bc['gt_rank']} | \"{bc['caption'][:60]}...\"")

            for i, bc in enumerate(bad_cases):
                results = retrieve_topk_for_text(
                    bc["caption"], model, tokenizer, unique_img_embeds,
                    unique_img_ids, device, topk=args.topk,
                )
                save_path = os.path.join(args.output_dir, f"badcase_{i+1}.png")
                title_prefix = f"[Bad Case #{i+1}  GT-Rank={bc['gt_rank']}]  "
                plot_topk_results(
                    bc["caption"], results, image_root,
                    gt_image_id=bc["gt_image_id"],
                    save_path=save_path,
                    title_prefix=title_prefix,
                )

    # ── 附加: 从测试集随机选一个 query 演示 ────────────────────────────────
    if not args.caption and not args.find_bad_cases:
        idx = np.random.randint(len(all_captions))
        cap = all_captions[idx]
        gt_id = all_img_ids[idx]
        print(f"\nRandom query #{idx}: \"{cap}\"")
        print(f"  GT image: {gt_id}")

        results = retrieve_topk_for_text(
            cap, model, tokenizer, unique_img_embeds,
            unique_img_ids, device, topk=args.topk,
        )
        for r in results:
            marker = " <-- GT" if r["image_id"] == gt_id else ""
            print(f"  Rank {r['rank']}: {r['image_id']}  score={r['score']:.4f}{marker}")

        save_path = os.path.join(args.output_dir, "topk_random.png")
        plot_topk_results(cap, results, image_root, gt_image_id=gt_id, save_path=save_path)


if __name__ == "__main__":
    main()
