"""FP16/BF16 half-precision inference benchmark."""
import argparse
import json
import os

import torch

from siglip.benchmark.utils import (
    load_model, load_criterion, load_tokenizer, build_test_loader,
    evaluate_retrieval, evaluate_loss, measure_model_size, measure_latency, print_results,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", default="outputs/exp21_dual_pretrained_v2/best_siglip.pt")
    parser.add_argument("--precision", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt_path
    dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16

    model, ckpt_args = load_model(ckpt_path, device)
    criterion = load_criterion(ckpt_path, device)
    tokenizer = load_tokenizer()
    test_loader = build_test_loader(ckpt_args, tokenizer, args.batch_size)

    # Convert model to half precision
    model = model.to(dtype)
    # Criterion stays in FP32 for numerical stability (logit_scale.exp() overflows in FP16)
    criterion = criterion.float()

    # Model size after conversion
    size = measure_model_size(model)
    print(f"Model size ({args.precision}): {size['total_mb']:.1f} MB, params: {size['total_params']:,}")

    # Latency
    print("Measuring latency...")
    latency = measure_latency(model, test_loader, device)

    # Accuracy (convert images to half before encoding, keep similarity in FP32)
    print("Evaluating retrieval...")
    metrics = evaluate_retrieval(model, test_loader, device)
    test_loss = evaluate_loss(model, criterion, test_loader, device)
    metrics["test_loss"] = test_loss

    print_results(f"{args.precision.upper()} Precision", metrics, latency, size)

    # Save results
    out_dir = os.path.join(os.path.dirname(ckpt_path), "..", "benchmark_results")
    os.makedirs(out_dir, exist_ok=True)
    results = {"name": f"{args.precision}_precision", "metrics": metrics, "latency": latency, "size": size}
    with open(os.path.join(out_dir, f"{args.precision}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_dir}/{args.precision}_results.json")


if __name__ == "__main__":
    main()