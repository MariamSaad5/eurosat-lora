import torch
from torch.utils.data import DataLoader, TensorDataset

from src.lora import apply_lora_to_vit
from src.metrics import get_adapter_state_dict
from src.train import run_one_epoch
from tests.helpers import tiny_vit


def _fake_loader():
    torch.manual_seed(0)
    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))
    return DataLoader(TensorDataset(images, labels), batch_size=4)


def test_one_training_epoch_only_updates_lora_and_classifier():
    model = tiny_vit()
    apply_lora_to_vit(model, target_blocks=[11], target_modules=["query", "value"], r=4, alpha=8)

    frozen_before = model.vit.layers[11].attention.q_proj.base.weight.clone()
    lora_b_before = model.vit.layers[11].attention.q_proj.lora_B.clone()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    loss, acc, f1 = run_one_epoch(model, _fake_loader(), torch.device("cpu"), optimizer)

    assert torch.isfinite(torch.tensor(loss))
    assert 0.0 <= acc <= 1.0 and 0.0 <= f1 <= 1.0
    assert torch.equal(frozen_before, model.vit.layers[11].attention.q_proj.base.weight)
    assert not torch.equal(lora_b_before, model.vit.layers[11].attention.q_proj.lora_B)


def test_adapter_checkpoint_only_holds_lora_and_classifier():
    model = tiny_vit()
    apply_lora_to_vit(model, target_blocks=[11], target_modules=["query"], r=4, alpha=8)
    keys = get_adapter_state_dict(model).keys()
    assert len(keys) > 0
    assert all("lora_" in k or k.startswith("classifier") for k in keys)


def test_finetuner_saves_best_checkpoint_by_val_loss(tmp_path):
    from src.config import ExperimentConfig, LoRAConfig
    from src.finetuner import LoRAFineTuner

    class TinyLoRAFineTuner(LoRAFineTuner):
        # Same as LoRAFineTuner, but builds the tiny test ViT instead of
        # downloading google/vit-base-patch16-224.
        def build_model(self, class_names):
            model = tiny_vit(num_labels=len(class_names))
            lora = self.config.lora
            return apply_lora_to_vit(model, lora.target_blocks, lora.target_modules, lora.r, lora.alpha)

    config = ExperimentConfig(
        name="tiny", model_name="tiny", data_dir="unused", output_dir=str(tmp_path),
        epochs=3, lr=1e-2,
        lora=LoRAConfig(r=4, alpha=8, target_blocks=[11], target_modules=["query", "value"]),
    )
    class_names = [f"c{i}" for i in range(10)]
    result = TinyLoRAFineTuner(config, torch.device("cpu")).run(
        _fake_loader(), _fake_loader(), class_names, verbose=False
    )

    ckpt = torch.load(result["checkpoint_path"])
    assert result["checkpoint_path"].endswith("tiny_best.pt")
    assert ckpt["epoch"] == result["best_epoch"]
    assert ckpt["val_loss"] == result["best_val_loss"]
    assert ckpt["class_names"] == class_names
    assert ckpt["lora_config"]["r"] == 4
    assert all("lora_" in k or k.startswith("classifier") for k in ckpt["adapter_state_dict"])