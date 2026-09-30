import pytest
import torch
import torch.nn as nn

from src.lora import LoRALinear, apply_lora_to_vit
from src.metrics import count_parameters
from tests.helpers import tiny_vit


def test_lora_is_a_no_op_at_start():
    base = nn.Linear(16, 8)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(3, 16)
    assert torch.allclose(lora(x), base(x))


def test_lora_rejects_non_linear_layers():
    with pytest.raises(AssertionError):
        LoRALinear(nn.Conv2d(3, 3, 1), r=4, alpha=8)


def test_injection_matches_real_vit_structure():
    model = tiny_vit()
    apply_lora_to_vit(model, target_blocks=[9, 10, 11], target_modules=["query", "value"], r=4, alpha=8)

    attn = model.vit.layers[9].attention
    assert isinstance(attn.q_proj, LoRALinear)
    assert isinstance(attn.v_proj, LoRALinear)
    assert not isinstance(attn.k_proj, LoRALinear)
    assert not isinstance(model.vit.layers[0].attention.q_proj, LoRALinear)


def test_trainable_parameter_count():
    hidden, r, labels = 32, 4, 10
    model = tiny_vit(num_labels=labels)
    apply_lora_to_vit(model, target_blocks=[9, 10, 11], target_modules=["query", "value"], r=r, alpha=8)

    _, trainable = count_parameters(model)
    lora_params = 3 * 2 * (r * hidden + hidden * r)
    classifier_params = hidden * labels + labels
    assert trainable == lora_params + classifier_params
