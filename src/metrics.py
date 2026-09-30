import io

import torch
import torch.nn as nn


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def state_dict_size_mb(state_dict: dict) -> float:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getbuffer().nbytes / (1024 ** 2)


def get_adapter_state_dict(model: nn.Module) -> dict:
    """Only the trainable pieces of a LoRA model: the LoRA A/B matrices plus
    the classifier head. This is the artifact you'd actually ship if you
    were distributing a LoRA adapter instead of a whole fine-tuned model."""
    return {
        k: v.cpu()
        for k, v in model.state_dict().items()
        if "lora_" in k or k.startswith("classifier")
    }
