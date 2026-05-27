"""Dynamic INT8 quantization benchmark (CPU deployment scenario).

PyTorch dynamic quantization produces INT8 Linear layers that only work on CPU.
For GPU, use FP16/BF16 or BitsAndBytes instead.
"""
import argparse
import json
import os
import time

import torch

from benchmark_utils import (
    load_model, load_criterion, load_tokenizer, build_test_loader,
    evaluate_retrieval, evaluate_loss, measure_model_size, print_results,
)


def measure_latency_cpu(model, loader, n_warmup=3, n_runs=10):
    """Measure CPU inference latency with perf_counter."""
    model.eval()
    latencies = []
    batch_sizes = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_warmup + n_runs:
                break
            images = batch["image"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            if i < n_warmup:
                _ = model(images, input_ids, attention_mask)
                continue

            start = time.perf_counter()
            _ = model(images, input_ids, attention_mask)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
            batch_sizes.append(images.size(0))

    avg_ms = sum(latencies) / len(latencies)
    avg_batch = sum(batch_sizes) / len(batch_sizes)
    return {
        "avg_batch_ms": avg_ms,
        "avg_per_sample_ms": avg_ms / avg_batch,
        "n_runs": len(latencies),
        "avg_batch_size": avg_batch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", default="outputs/exp21_dual_pretrained_v2/best_siglip.pt")
    parser.add_argument("--scope", choices=["all", "frozen_only", "frozen_and_proj"], default="all")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt_path = args.ckpt_path

    model, ckpt_args = load_model(ckpt_path, device)
    criterion = load_criterion(ckpt_path, device)
    tokenizer = load_tokenizer()
    # Smaller batch for CPU, fewer workers
    from torch.utils.data import DataLoader
    from data_loader import Flickr8kDistilBERTDataset, build_siglip_transform
    img_dir = "Images" if os.path.isdir(os.path.join(ckpt_args.data_dir, "Images")) else "images"
    image_root = os.path.join(ckpt_args.data_dir, img_dir)
    test_dataset = Flickr8kDistilBERTDataset(
        image_root=image_root,
        captions_file=os.path.join(ckpt_args.data_dir, "test_captions.txt"),
        tokenizer=tokenizer,
        transform=build_siglip_transform(ckpt_args.image_size, train=False),
        max_len=ckpt_args.max_len,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Apply dynamic quantization
    if args.scope == "all":
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("Quantized ALL Linear layers (INT8)")
    elif args.scope == "frozen_only":
        # Quantize only frozen ViT bottom layers + frozen DistilBERT bottom layers
        freeze_until_vit = model.vit_freeze_until
        freeze_until_db = model.db_freeze_until
        for i in range(freeze_until_vit):
            torch.quantization.quantize_dynamic(
                model.vision_model.encoder.layers[i],
                {torch.nn.Linear}, dtype=torch.qint8, inplace=True,
            )
        for i in range(freeze_until_db):
            torch.quantization.quantize_dynamic(
                model.text_model.transformer.layer[i],
                {torch.nn.Linear}, dtype=torch.qint8, inplace=True,
            )
        # Also quantize embeddings
        torch.quantization.quantize_dynamic(
            model.vision_model.embeddings, {torch.nn.Linear}, dtype=torch.qint8, inplace=True,
        )
        torch.quantization.quantize_dynamic(
            model.text_model.embeddings, {torch.nn.Linear}, dtype=torch.qint8, inplace=True,
        )
        quantized_model = model
        print(f"Quantized frozen layers only (ViT 0-{freeze_until_vit-1}, DB 0-{freeze_until_db-1})")
    elif args.scope == "frozen_and_proj":
        freeze_until_vit = model.vit_freeze_until
        freeze_until_db = model.db_freeze_until
        for i in range(freeze_until_vit):
            torch.quantization.quantize_dynamic(
                model.vision_model.encoder.layers[i],
                {torch.nn.Linear}, dtype=torch.qint8, inplace=True,
            )
        for i in range(freeze_until_db):
            torch.quantization.quantize_dynamic(
                model.text_model.transformer.layer[i],
                {torch.nn.Linear}, dtype=torch.qint8, inplace=True,
            )
        torch.quantization.quantize_dynamic(
            model.vision_model.embeddings, {torch.nn.Linear}, dtype=torch.qint8, inplace=True,
        )
        torch.quantization.quantize_dynamic(
            model.text_model.embeddings, {torch.nn.Linear}, dtype=torch.qint8, inplace=True,
        )
        # Also quantize projection
        if hasattr(model.text_projection, 'weight'):
            torch.quantization.quantize_dynamic(
                model.text_projection, {torch.nn.Linear}, dtype=torch.qint8, inplace=True,
            )
        quantized_model = model
        print(f"Quantized frozen layers + projection")

    # Model size
    size_before = measure_model_size(load_model(ckpt_path, device)[0])
    size = measure_model_size(quantized_model)
    print(f"Model size before: {size_before['total_mb']:.1f} MB, after: {size['total_mb']:.1f} MB")
    print(f"Size reduction: {size_before['total_mb'] / size['total_mb']:.2f}x")

    # Latency (CPU)
    print("Measuring CPU latency...")
    latency = measure_latency_cpu(quantized_model, test_loader)

    # Accuracy (CPU)
    print("Evaluating retrieval on CPU...")
    metrics = evaluate_retrieval(quantized_model, test_loader, device)
    test_loss = evaluate_loss(quantized_model, criterion, test_loader, device)
    metrics["test_loss"] = test_loss

    print_results(f"Dynamic INT8 ({args.scope})", metrics, latency, size)

    # Save results
    out_dir = os.path.join(os.path.dirname(ckpt_path), "..", "benchmark_results")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"dynamic_int8_{args.scope}.json"
    results = {"name": f"dynamic_int8_{args.scope}", "scope": args.scope,
               "metrics": metrics, "latency": latency, "size": size, "size_before": size_before}
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_dir}/{fname}")


if __name__ == "__main__":
    main()