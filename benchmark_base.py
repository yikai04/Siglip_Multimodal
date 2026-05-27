"""Baseline benchmark for Exp21 model (FP32 on GPU)."""
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt_path

    model, ckpt_args = load_model(ckpt_path, device)
    criterion = load_criterion(ckpt_path, device)
    tokenizer = load_tokenizer()
    test_loader = build_test_loader(ckpt_args, tokenizer, args.batch_size)

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

    print_results("Baseline (FP32)", metrics, latency, size)

    # Save results
    out_dir = os.path.join(os.path.dirname(ckpt_path), "..", "benchmark_results")
    os.makedirs(out_dir, exist_ok=True)
    results = {"name": "baseline_fp32", "metrics": metrics, "latency": latency, "size": size}
    with open(os.path.join(out_dir, "baseline_fp32.json"), "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, (torch.Tensor,)) else o)
    print(f"Results saved to {out_dir}/baseline_fp32.json")


if __name__ == "__main__":
    main()