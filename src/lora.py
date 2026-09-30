import math
from typing import List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear and adds a trainable low-rank update.

    Instead of learning a full weight update delta_W, LoRA factors it as
    delta_W ~= (alpha / r) * B @ A, where A is (r, in_features) and B is
    (out_features, r). Only A and B are trained; the wrapped linear layer's
    original weight and bias are frozen.

    A is Kaiming-initialized; B starts at all zeros, so at step 0 the layer's
    output is identical to the original frozen layer (B @ A == 0). Training
    then grows the adapter from nothing instead of starting from a random
    perturbation on top of pretrained features.
    """

    def __init__(self, base_linear: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear), (
            f"LoRALinear expected an nn.Linear, got {type(base_linear)}. "
            "This usually means MODULE_PATHS doesn't match the installed "
            "transformers version's internal module names - see the note "
            "in apply_lora_to_vit."
        )
        self.base = base_linear
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.zeros(r, base_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B intentionally left at zero - see class docstring.

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out


# Maps a friendly name to the dotted attribute path inside a ViTLayer block.
#
# NOTE ON TRANSFORMERS VERSIONS: this matches transformers v5's internal
# structure. The older v4-style layout was
# `layer.attention.attention.{query,key,value}` / `layer.intermediate.dense`
# / `layer.output.dense`. If a future transformers release renames things
# again, build a model, run `print(model.vit.layers[0])`, and update the
# paths below to match - the assertion inside LoRALinear will fail loudly
# and clearly if a path ever resolves to something that isn't an nn.Linear,
# rather than silently wrapping the wrong object.
MODULE_PATHS = {
    "query": ("attention", "q_proj"),
    "key": ("attention", "k_proj"),
    "value": ("attention", "v_proj"),
    "attention_output": ("attention", "o_proj"),
    "intermediate": ("mlp", "fc1"),
    "output": ("mlp", "fc2"),
}


def apply_lora_to_vit(
    model: nn.Module,
    target_blocks: List[int],
    target_modules: List[str],
    r: int,
    alpha: int,
    dropout: float = 0.0,
) -> nn.Module:
    """Freezes the whole model, then replaces the chosen nn.Linear submodules
    in the chosen transformer blocks with LoRALinear wrappers. The
    classification head is left trainable regardless, since it's freshly
    initialized for the target task and has to train no matter what."""
    for p in model.parameters():
        p.requires_grad = False

    for block_idx in target_blocks:
        layer = model.vit.layers[block_idx]
        for name in target_modules:
            parent_path, attr = MODULE_PATHS[name]
            parent = layer
            for part in parent_path.split("."):
                parent = getattr(parent, part)
            original_linear = getattr(parent, attr)
            setattr(parent, attr, LoRALinear(original_linear, r=r, alpha=alpha, dropout=dropout))

    for p in model.classifier.parameters():
        p.requires_grad = True

    return model
