"""Extract GT image IDs for bad cases and create comparison visualization."""
import torch
from visualize_topk import (
    load_model_and_tokenizer, build_test_dataset,
    compute_all_embeddings, get_unique_images, find_bad_cases,
    retrieve_topk_for_text
)

device = torch.device("cuda")
model, tokenizer, ckpt_args = load_model_and_tokenizer(
    "outputs/exp11_wd005/best_siglip.pt", device
)
dataset = build_test_dataset("Flickr8k", tokenizer)
img_embeds, txt_embeds, all_img_ids, all_captions = compute_all_embeddings(
    model, dataset, device
)
unique_img_embeds, unique_img_ids, img2indices, img_idx_map = get_unique_images(
    img_embeds, all_img_ids
)

bad_cases = find_bad_cases(
    model, tokenizer, txt_embeds, all_img_ids, all_captions,
    unique_img_embeds, unique_img_ids, img_idx_map,
    device, rank_threshold=10, num_cases=3
)

print("=" * 60)
print("Bad Case GT Image IDs:")
for i, bc in enumerate(bad_cases):
    print(f"  [{i+1}] caption: {bc['caption']}")
    print(f"      gt_image_id: {bc['gt_image_id']}")
    print(f"      gt_rank: {bc['gt_rank']}")