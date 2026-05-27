"""Layer dropping (structured pruning) benchmark for Exp21 model.

Drops bottom frozen ViT/DistilBERT layers to reduce inference compute.
Each dropped layer saves ~7M params and one transformer block of computation.
"""
import argparse
import json
import os

import torch

from benchmark_utils import (
    load_model, load_criterion, load_tokenizer, build_test_loader,
    evaluate_retrieval, evaluate_loss, measure_model_size, measure_latency, print_results,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", default="outputs/exp21_dual_pretrained_v2/best_siglip.pt")
    parser.add_argument("--vit-drop", type=int, default=0, help="Number of bottom frozen ViT layers to drop")
    parser.add_argument("--db-drop", type=int, default=0, help="Number of bottom frozen DistilBERT layers to drop")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt_path

    model, ckpt_args = load_model(ckpt_path, device, n_vit_drop=args.vit_drop, n_db_drop=args.db_drop)
    criterion = load_criterion(ckpt_path, device)
    tokenizer = load_tokenizer()
    test_loader = build_test_loader(ckpt_args, tokenizer, args.batch_size)

    # Report remaining layers
    vit_remaining = len(model.vision_model.encoder.layers)
    db_remaining = len(model.text_model.transformer.layer)
    print(f"ViT layers remaining: {vit_remaining} (dropped {args.vit_drop})")
    print(f"DistilBERT layers remaining: {db_remaining} (dropped {args.db_drop})")

    # Model size
    size = measure_model_size(model)
    print(f"Model size: {size['total_mb']:.1f} MB, params: {size['total_params']:,}")

    # Latency
    print("Measuring latency...")
    latency = measure_latency(model, test_loader, device)

    # Accuracy
    print("Evaluating retrieval...")
    metrics = evaluate_retrieval(model, test_loader, device)
    test_loss = evaluate_loss(model, criterion, test_loader, device)
    metrics["test_loss"] = test_loss

    name = f"Prune ViT-drop{args.vit_drop} DB-drop{args.db_drop}"
    print_results(name, metrics, latency, size)

    # Save results
    out_dir = os.path.join(os.path.dirname(ckpt_path), "..", "benchmark_results")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"prune_vit{args.vit_drop}_db{args.db_drop}.json"
    results = {
        "name": name,
        "vit_drop": args.vit_drop, "db_drop": args.db_drop,
        "vit_remaining": vit_remaining, "db_remaining": db_remaining,
        "metrics": metrics, "latency": latency, "size": size,
    }
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_dir}/{fname}")


if __name__ == "__main__":
    main()