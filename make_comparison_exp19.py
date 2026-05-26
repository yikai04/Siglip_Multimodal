"""Create comparison visualizations for Exp19 bad cases."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16, "savefig.dpi": 300})

legend_elements = [mpatches.Patch(facecolor="none", edgecolor="#e74c3c", linewidth=2, label="Ground Truth")]

# Bad Case 1
gt1 = Image.open("viz_outputs/exp19_gt_badcase_1.jpg").convert("RGB")
topk1 = Image.open("viz_outputs/exp19_badcase_1.png").convert("RGB")
fig, axes = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [1, 3]})
axes[0].imshow(gt1)
for spine in axes[0].spines.values():
    spine.set_edgecolor("#e74c3c"); spine.set_linewidth(5); spine.set_visible(True)
axes[0].set_title("Ground Truth (Rank 2441)", fontsize=14, fontweight="bold", color="#e74c3c")
axes[0].axis("off")
axes[1].imshow(topk1)
axes[1].set_title("Model Top-5 Retrieved Results", fontsize=14, fontweight="bold")
axes[1].axis("off")
fig.suptitle('Exp19 Bad Case #1: "A boy rides a skateboard with a bike to his right"', fontsize=14, fontweight="bold", color="#e74c3c", y=1.02)
fig.legend(handles=legend_elements, loc="lower center", ncol=1, fontsize=12, frameon=True, fancybox=True)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig("viz_outputs/exp19_badcase1_comparison.png", bbox_inches="tight")
print("[Saved] viz_outputs/exp19_badcase1_comparison.png")
plt.close(fig)

# Bad Case 2
gt2 = Image.open("viz_outputs/exp19_gt_badcase_2.jpg").convert("RGB")
topk2 = Image.open("viz_outputs/exp19_badcase_2.png").convert("RGB")
fig, axes = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [1, 3]})
axes[0].imshow(gt2)
for spine in axes[0].spines.values():
    spine.set_edgecolor("#e74c3c"); spine.set_linewidth(5); spine.set_visible(True)
axes[0].set_title("Ground Truth (Rank 2085)", fontsize=14, fontweight="bold", color="#e74c3c")
axes[0].axis("off")
axes[1].imshow(topk2)
axes[1].set_title("Model Top-5 Retrieved Results", fontsize=14, fontweight="bold")
axes[1].axis("off")
fig.suptitle('Exp19 Bad Case #2: "Where is the rest of his racket?"', fontsize=14, fontweight="bold", color="#e74c3c", y=1.02)
fig.legend(handles=legend_elements, loc="lower center", ncol=1, fontsize=12, frameon=True, fancybox=True)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig("viz_outputs/exp19_badcase2_comparison.png", bbox_inches="tight")
print("[Saved] viz_outputs/exp19_badcase2_comparison.png")
plt.close(fig)
