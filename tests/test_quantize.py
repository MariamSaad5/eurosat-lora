import copy
import io

import torch
import torch.nn as nn
from transformers import ViTImageProcessor

from src.lora import LoRALinear, apply_lora_to_vit, merge_lora_weights
from src.quantize import load_quantized, quantize_int8, save_quantized
from tests.helpers import tiny_vit


def _trained_looking_lora_vit():
    torch.manual_seed(0)
    model = tiny_vit()
    apply_lora_to_vit(model, target_blocks=[10, 11], target_modules=["query", "value"], r=4, alpha=8)
    # B starts at zero, which would make the merge trivially correct.
    # Give it random values so the test checks the real formula.
    for m in model.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B, std=0.1)
    return model.eval()


def test_merge_gives_same_outputs_and_removes_lora_layers():
    model = _trained_looking_lora_vit()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        before = model(pixel_values=x).logits
        merge_lora_weights(model)
        after = model(pixel_values=x).logits
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert torch.allclose(before, after, atol=1e-5)


def _size_bytes(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


def test_int8_model_is_smaller_and_close_to_fp32():
    model = merge_lora_weights(_trained_looking_lora_vit())
    int8 = quantize_int8(copy.deepcopy(model))
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        gap = (model(pixel_values=x).logits - int8(pixel_values=x).logits).abs().max().item()
    # every float nn.Linear was swapped for an INT8 one
    assert not any(type(m) is nn.Linear for m in int8.modules())
    assert _size_bytes(int8) < _size_bytes(model)
    assert gap < 0.1


def test_saved_int8_model_reloads_with_identical_outputs(tmp_path):
    int8 = quantize_int8(merge_lora_weights(_trained_looking_lora_vit()))
    path = tmp_path / "int8.pt"
    class_names = [f"c{i}" for i in range(10)]
    save_quantized(int8, ViTImageProcessor(), class_names, str(path))

    reloaded, processor, names = load_quantized(str(path))
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(int8(pixel_values=x).logits, reloaded(pixel_values=x).logits)
    assert names == class_names
    assert processor.size == ViTImageProcessor().size