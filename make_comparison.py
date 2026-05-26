"""Create comparison visualization: GT image vs Top-5 retrieved results for bad cases."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "savefig.dpi": 300,
})

# Load images
gt_img_path = os.path.join("viz_outputs", "gt_badcase2.jpg")
gt_img = Image.open(gt_img_path).convert("RGB")

# Load the top-5 retrieved results image
topk_img_path = os.path.join("viz_outputs", "local_badcase_2.png")
topk_img = Image.open(topk_img_path).convert("RGB")

fig, axes = plt.subplots(1, 2, figsize=(14, 8),
                         gridspec_kw={'width_ratios': [1, 3]})

# Show GT image
axes[0].imshow(gt_img)
for spine in axes[0].spines.values():
    spine.set_edgecolor("#e74c3c")
    spine.set_linewidth(5)
    spine.set_visible(True)
axes[0].set_title("Ground Truth (Rank 2252)\n3033612929_764d977bd5.jpg",
                  fontsize=14, fontweight="bold", color="#e74c3c")
axes[0].axis("off")

# Show top-5 retrieved results
axes[1].imshow(topk_img)
axes[1].set_title("Model Top-5 Retrieved Results",
                  fontsize=14, fontweight="bold", color="#333333")
axes[1].axis("off")

fig.suptitle("Bad Case #2: \"Two dogs play with each other outdoors\"\n"
             "GT image ranked at 2252 (out of 3290) — NOT found in Top-5",
             fontsize=16, fontweight="bold", color="#e74c3c", y=1.02)

legend_elements = [
    mpatches.Patch(facecolor="none", edgecolor="#e74c3c", linewidth=2, label="Ground Truth (correct answer)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=1,
           fontsize=12, frameon=True, fancybox=True)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
save_path = os.path.join("viz_outputs", "badcase2_comparison.png")
fig.savefig(save_path, bbox_inches="tight")
print(f"[Saved] {save_path}")
plt.close(fig)

# Also do the same for Bad Case #1
gt_img1_path = os.path.join("viz_outputs", "gt_badcase1.jpg")
gt_img1 = Image.open(gt_img1_path).convert("RGB")
topk_img1_path = os.path.join("viz_outputs", "local_badcase_1.png")
topk_img1 = Image.open(topk_img1_path).convert("RGB")

fig, axes = plt.subplots(1, 2, figsize=(14, 8),
                         gridspec_kw={'width_ratios': [1, 3]})

axes[0].imshow(gt_img1)
for spine in axes[0].spines.values():
    spine.set_edgecolor("#e74c3c")
    spine.set_linewidth(5)
    spine.set_visible(True)
axes[0].set_title("Ground Truth (Rank 2944)\n3239866450_3f8cfb0c83.jpg",
                  fontsize=14, fontweight="bold", color="#e74c3c")
axes[0].axis("off")

axes[1].imshow(topk_img1)
axes[1].set_title("Model Top-5 Retrieved Results",
                  fontsize=14, fontweight="bold", color="#333333")
axes[1].axis("off")

fig.suptitle("Bad Case #1: \"I have no idea!\"\n"
             "GT image ranked at 2944 (out of 3290) — NOT found in Top-5",
             fontsize=16, fontweight="bold", color="#e74c3c", y=1.02)

fig.legend(handles=legend_elements, loc="lower center", ncol=1,
           fontsize=12, frameon=True, fancybox=True)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
save_path = os.path.join("viz_outputs", "badcase1_comparison.png")
fig.savefig(save_path, bbox_inches="tight")
print(f"[Saved] {save_path}")
plt.close(fig)