import os

from PIL import Image
from transformers import ViTImageProcessor

import src.data as data_module
from src.config import ExperimentConfig


def _make_fake_eurosat(root, classes=10, per_class=10):
    """Creates a tiny folder-per-class dataset shaped like EuroSAT_RGB."""
    for c in range(classes):
        folder = os.path.join(root, f"Class{c}")
        os.makedirs(folder)
        for i in range(per_class):
            Image.new("RGB", (64, 64), color=(c * 20, i * 20, 100)).save(os.path.join(folder, f"{i}.jpg"))


def _offline_processor(monkeypatch):
    """Replaces the internet download of the image processor with a local one."""
    monkeypatch.setattr(
        data_module.ViTImageProcessor,
        "from_pretrained",
        lambda *args, **kwargs: ViTImageProcessor(size={"height": 32, "width": 32}),
    )


def _config(tmp_path, **overrides):
    return ExperimentConfig(
        name="test",
        model_name="unused-in-tests",
        data_dir=str(tmp_path / "EuroSAT_RGB"),
        output_dir=str(tmp_path / "outputs"),
        **overrides,
    )


def test_split_is_70_15_15_and_cached(tmp_path, monkeypatch):
    _offline_processor(monkeypatch)
    _make_fake_eurosat(tmp_path / "EuroSAT_RGB")
    cfg = _config(tmp_path)

    train_ds, val_ds, test_ds, classes = data_module.build_datasets(cfg)
    assert len(classes) == 10
    assert (len(train_ds), len(val_ds), len(test_ds)) == (70, 15, 15)
    assert os.path.exists(tmp_path / "outputs" / "split_indices.json")

    image, label = train_ds[0]
    assert tuple(image.shape) == (3, 32, 32)

    train_again, _, _, _ = data_module.build_datasets(cfg)
    assert train_again.indices == train_ds.indices


def test_debug_limits_shrink_the_dataset(tmp_path, monkeypatch):
    _offline_processor(monkeypatch)
    _make_fake_eurosat(tmp_path / "EuroSAT_RGB")
    cfg = _config(tmp_path, debug_max_train=5, debug_max_val=3)

    train_ds, val_ds, _, _ = data_module.build_datasets(cfg)
    assert (len(train_ds), len(val_ds)) == (5, 3)
