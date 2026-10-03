"""Assignment 14: merge the best LoRA checkpoint into the ViT and quantize it to INT8.

Example:
    python quantize.py --checkpoint outputs/lora_r16_best.pt --output outputs/vit_lora_r16_int8.pt
Add --config configs/lora_r16.yaml to also compare FP32 vs INT8 accuracy on the test set.
"""
import argparse
import copy
import io
import time

import torch
from torch.utils.data import DataLoader, Subset
from transformers import ViTImageProcessor

from src.config import load_config
from src.data import build_datasets
from src.lora import merge_lora_weights
from src.quantize import load_lora_model, load_quantized, quantize_int8, save_quantized


def size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 ** 2)


@torch.no_grad()
def evaluate(model, loader):
    correct, total, seconds = 0, 0, 0.0
    for images, labels in loader:
        start = time.perf_counter()
        logits = model(pixel_values=images).logits
        seconds += time.perf_counter() - start
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total, 1000.0 * seconds / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="LoRA checkpoint saved by training, e.g. lora_r16_best.pt")
    parser.add_argument("--output", required=True, help="Where to save the INT8 model")
    parser.add_argument("--config", help="Optional: training config, used to rebuild the test set for an accuracy check")
    parser.add_argument("--eval_samples", type=int, default=500, help="How many test images to use for the accuracy check")
    args = parser.parse_args()

    torch.manual_seed(0)

    # 1. Rebuild the trained LoRA model
    lora_model, ckpt = load_lora_model(args.checkpoint)
    print(f"Loaded {args.checkpoint} (best epoch {ckpt['epoch']}, val_acc {ckpt['val_accuracy']:.4f})")

    # 2. Merge LoRA into the base weights, and check the merge changed nothing
    size = lora_model.config.image_size  # 224 for ViT-Base
    x = torch.randn(4, 3, size, size)
    with torch.no_grad():
        before = lora_model(pixel_values=x).logits
        fp32_model = merge_lora_weights(lora_model)
        after = fp32_model(pixel_values=x).logits
    print(f"Merge check, max logit difference: {(before - after).abs().max().item():.2e}")

    # 3. Quantize to INT8 and compare sizes
    fp32_mb = size_mb(fp32_model)
    int8_model = quantize_int8(copy.deepcopy(fp32_model))
    int8_mb = size_mb(int8_model)
    print(f"FP32 size: {fp32_mb:.1f} MB | INT8 size: {int8_mb:.1f} MB | {fp32_mb / int8_mb:.2f}x smaller")

    # 4. Save, then reload from disk to prove the saved file works on its own
    processor = ViTImageProcessor.from_pretrained(ckpt["model_name"])
    save_quantized(int8_model, processor, ckpt["class_names"], args.output)
    reloaded, _, _ = load_quantized(args.output)
    with torch.no_grad():
        gap = (int8_model(pixel_values=x).logits - reloaded(pixel_values=x).logits).abs().max().item()
    print(f"Saved to {args.output}. Reload check, max logit difference: {gap:.2e}")

    # 5. Optional: accuracy and CPU speed on real test images
    if args.config:
        config = load_config(args.config)
        _, _, test_ds, _ = build_datasets(config)
        test_ds = Subset(test_ds, range(min(args.eval_samples, len(test_ds))))
        loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
        fp32_acc, fp32_ms = evaluate(fp32_model, loader)
        int8_acc, int8_ms = evaluate(reloaded, loader)
        print(f"\nTest accuracy on {len(test_ds)} images (CPU):")
        print(f"  FP32: {fp32_acc:.2f}%  ({fp32_ms:.1f} ms/image)")
        print(f"  INT8: {int8_acc:.2f}%  ({int8_ms:.1f} ms/image)")


if __name__ == "__main__":
    main()