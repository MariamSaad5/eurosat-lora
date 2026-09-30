from transformers import ViTConfig, ViTForImageClassification


def tiny_vit(num_labels: int = 10) -> ViTForImageClassification:
    """A very small ViT with random weights. It has the same structure as
    google/vit-base-patch16-224 (12 blocks, same layer names), just much
    narrower, so tests run in seconds and need no internet download."""
    config = ViTConfig(
        hidden_size=32,
        num_hidden_layers=12,
        num_attention_heads=2,
        intermediate_size=64,
        image_size=32,
        patch_size=8,
        num_labels=num_labels,
    )
    return ViTForImageClassification(config)
