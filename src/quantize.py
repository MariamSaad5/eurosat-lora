"""Turns a LoRA checkpoint into a small INT8 model for CPU inference.

Pipeline:  LoRA checkpoint  ->  ViT + LoRA  ->  merged ViT (FP32)  ->  INT8 ViT

The INT8 model is saved together with everything needed to rebuild it
(model config, image processor settings, class names), so the inference
service can load it without downloading anything from Hugging Face.
"""
import json

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

from .lora import apply_lora_to_vit, merge_lora_weights


def load_lora_model(checkpoint_path: str):
    """Rebuilds the trained model from a *_best.pt checkpoint: a fresh
    pretrained ViT, the same LoRA layers, then the saved LoRA + classifier
    weights on top. Returns (model, checkpoint)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    class_names = ckpt["class_names"]
    lora_cfg = ckpt["lora_config"]

    model = ViTForImageClassification.from_pretrained(
        ckpt["model_name"],
        num_labels=len(class_names),
        ignore_mismatched_sizes=True,
        id2label={i: c for i, c in enumerate(class_names)},
        label2id={c: i for i, c in enumerate(class_names)},
    )
    apply_lora_to_vit(
        model,
        target_blocks=lora_cfg["target_blocks"],
        target_modules=lora_cfg["target_modules"],
        r=lora_cfg["r"],
        alpha=lora_cfg["alpha"],
        dropout=lora_cfg.get("dropout", 0.0),
    )
    # strict=False because the checkpoint only holds LoRA + classifier
    # weights; the frozen backbone already came from from_pretrained.
    _, unexpected = model.load_state_dict(ckpt["adapter_state_dict"], strict=False)
    assert not unexpected, f"Checkpoint does not match this architecture: {unexpected[:5]}"
    return model.eval(), ckpt


def quantize_int8(model: nn.Module) -> nn.Module:
    """Dynamic INT8 quantization: every nn.Linear weight is stored as 8-bit
    integers (1 byte instead of 4). Activations are quantized on the fly
    during inference, so no calibration data is needed. Runs on CPU only."""
    return torch.ao.quantization.quantize_dynamic(model.eval(), {nn.Linear}, dtype=torch.qint8)


def save_quantized(model: nn.Module, processor: ViTImageProcessor, class_names: list, path: str):
    torch.save(
        {
            # Stored as JSON text, not Python objects, so torch.load can keep
            # its safe default (weights_only=True) when reading the file back.
            "model_config": model.config.to_json_string(),
            "processor_config": processor.to_json_string(),
            "class_names": class_names,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_quantized(path: str):
    """Loads a model saved by save_quantized. The empty ViT is built from the
    saved config and quantized first, so its layers have the same INT8 shape
    as the saved weights; then the weights are loaded into it.
    Returns (model, processor, class_names)."""
    saved = torch.load(path, map_location="cpu")
    model = ViTForImageClassification(ViTConfig.from_dict(json.loads(saved["model_config"])))
    model = quantize_int8(model)
    model.load_state_dict(saved["state_dict"])
    processor = ViTImageProcessor.from_dict(json.loads(saved["processor_config"]))
    return model.eval(), processor, saved["class_names"]