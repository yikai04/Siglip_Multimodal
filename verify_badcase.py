"""Verify Top-5 image IDs for Bad Case #2 and download those images."""
import torch, os, shutil
from visualize_topk import (
    load_model_and_tokenizer, build_test_dataset,
    compute_all_embeddings, get_unique_images, retrieve_topk_for_text
)

device = torch.device("cuda")
model, tokenizer, ckpt_args = load_model_and_tokenizer(
    "outputs/exp11_wd005/best_siglip.pt", device
)
dataset = build_test_dataset("Flickr8k", tokenizer)
img_embeds, txt_embeds, all_img_ids, all_captions = compute_all_embeddings(model, dataset, device)
unique_img_embeds, unique_img_ids, img2indices, img_idx_map = get_unique_images(img_embeds, all_img_ids)

# Run the same query as bad case #2
caption = "Two dogs play with each other outdoors ."
results = retrieve_topk_for_text(caption, model, tokenizer, unique_img_embeds, unique_img_ids, device, topk=5)

print("=" * 60)
print(f"Query: {caption}")
for r in results:
    img_path = os.path.join("Flickr8k/images", r["image_id"])
    exists = os.path.exists(img_path)
    print(f"  Rank {r['rank']}: {r['image_id']}  score={r['score']:.4f}  file_exists={exists}")

# Copy the top-5 images to viz_outputs for verification
os.makedirs("viz_outputs/verify_top5", exist_ok=True)
for r in results:
    src = os.path.join("Flickr8k/images", r["image_id"])
    dst = os.path.join("viz_outputs/verify_top5", f"rank{r['rank']}_{r['image_id']}")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied: {dst}")
    else:
        print(f"  NOT FOUND: {src}")

# Also copy GT image
gt_id = "3033612929_764d977bd5.jpg"
gt_src = os.path.join("Flickr8k/images", gt_id)
gt_dst = os.path.join("viz_outputs/verify_top5", f"GT_{gt_id}")
if os.path.exists(gt_src):
    shutil.copy2(gt_src, gt_dst)
    print(f"  Copied GT: {gt_dst}")
