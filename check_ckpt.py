import torch

ckpt = torch.load("outputs/exp19_pretrained_vit/best_siglip.pt", map_location="cpu", weights_only=False)
print("Best checkpoint:")
print(f"  epoch: {ckpt.get('epoch', '?')}")
print(f"  best_val_r1: {ckpt.get('best_val_r1', 0)*100:.2f}%")
metrics = ckpt.get("metrics", {})
for k, v in metrics.items():
    print(f"  {k}: {v*100:.2f}%")

latest = torch.load("outputs/exp19_pretrained_vit/latest_siglip.pt", map_location="cpu", weights_only=False)
print("\nLatest checkpoint:")
print(f"  epoch: {latest.get('epoch', '?')}")
print(f"  best_val_r1: {latest.get('best_val_r1', 0)*100:.2f}%")
m2 = latest.get("metrics", {})
for k, v in m2.items():
    print(f"  {k}: {v*100:.2f}%")
