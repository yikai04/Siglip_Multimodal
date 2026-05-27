"""Create comparison visualizations for Exp21 bad cases.

Run viz_exp21.py first to generate bad case images, then update this script
with actual bad case captions and ranks before running.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16, "savefig.dpi": 300})

legend_elements = [mpatches.Patch(facecolor="none", edgecolor="#e74c3c", linewidth=2, label="Ground Truth")]

output_dir = "viz_outputs"

bad_cases = [
    # Fill in after running viz_exp21.py
]

for i, bc in enumerate(bad_cases, 1):
    gt_path = os.path.join(output_dir, f"exp21_gt_badcase_{i}.jpg")
    topk_path = os.path.join(output_dir, f"exp21_badcase_{i}.png")
    if not os.path.exists(gt_path) or not os.path.exists(topk_path):
        print(f"Missing images for bad case {i}, skipping")
        continue

    gt = Image.open(gt_path).convert("RGB")
    topk = Image.open(topk_path).convert("RGB")
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1, 3]})
    axes[0].imshow(gt)
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#e74c3c")
        spine.set_linewidth(5)
        spine.set_visible(True)
    axes[0].set_title(f"Ground Truth (Rank {bc['rank']})", fontsize=14, fontweight="bold", color="#e74c3c")
    axes[0].axis("off")
    axes[1].imshow(topk)
    axes[1].set_title("Model Top-5 Retrieved Results", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    caption_short = bc["caption"][:70] + "..." if len(bc["caption"]) > 70 else bc["caption"]
    fig.suptitle(f"Exp21 Bad Case #{i}: \"{caption_short}\"", fontsize=14, fontweight="bold", color="#e74c3c", y=1.02)
    fig.legend(handles=legend_elements, loc="lower center", ncol=1, fontsize=12, frameon=True, fancybox=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_path = os.path.join(output_dir, f"exp21_badcase{i}_comparison.png")
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[Saved] {save_path}")
    plt.close(fig)

print("Done!")