import json
import os
from typing import Optional, Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from transformers import ViTImageProcessor

from .config import ExperimentConfig


class EuroSATTransform:
    """Wraps the HF image processor so it works as a torchvision-style
    transform (PIL image in, normalized tensor out)."""

    def __init__(self, processor: ViTImageProcessor):
        self.processor = processor

    def __call__(self, image):
        out = self.processor(image.convert("RGB"), return_tensors="pt")
        return out["pixel_values"].squeeze(0)


def find_existing_split(data_dir: str) -> Optional[dict]:
    """Looks for common split file patterns (train/val/test folders or CSVs)
    alongside data_dir. Returns None if nothing is found, so the caller
    falls back to a manual stratified split."""
    parent = os.path.dirname(data_dir.rstrip("/"))
    for base in (data_dir, parent):
        train_csv = os.path.join(base, "train.csv")
        val_csv = os.path.join(base, "validation.csv")
        test_csv = os.path.join(base, "test.csv")
        if os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv):
            return {"type": "csv", "train": train_csv, "val": val_csv, "test": test_csv}

        train_dir = os.path.join(base, "train")
        val_dir = os.path.join(base, "val")
        test_dir = os.path.join(base, "test")
        if os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(test_dir):
            return {"type": "folders", "train": train_dir, "val": val_dir, "test": test_dir}

    return None


def _maybe_subset(dataset: Dataset, limit: Optional[int]) -> Dataset:
    """Used only for debug_max_train / debug_max_val - trims a dataset down
    to its first `limit` items so a sanity-check config can run in seconds."""
    if limit is None:
        return dataset
    n = min(limit, len(dataset))
    return Subset(dataset, list(range(n)))


def build_datasets(config: ExperimentConfig) -> Tuple[Dataset, Dataset, Dataset, list]:
    """Builds train/val/test datasets for EuroSAT.

    If an existing split is found alongside config.data_dir, it's used as-is.
    Otherwise, a stratified 70/15/15 split is computed once and cached to
    <output_dir>/split_indices.json, so every experiment that shares the same
    output_dir trains and validates on identical data - important for a fair
    comparison across configs.
    """
    processor = ViTImageProcessor.from_pretrained(config.model_name)
    transform = EuroSATTransform(processor)

    split_info = find_existing_split(config.data_dir)

    if split_info is not None and split_info["type"] == "folders":
        print("Found existing train/val/test folders, using those.")
        train_ds = datasets.ImageFolder(split_info["train"], transform=transform)
        val_ds = datasets.ImageFolder(split_info["val"], transform=transform)
        test_ds = datasets.ImageFolder(split_info["test"], transform=transform)
        class_names = train_ds.classes
    else:
        print("No pre-existing split found. Using a stratified 70/15/15 split.")
        full_ds = datasets.ImageFolder(config.data_dir, transform=transform)
        class_names = full_ds.classes

        os.makedirs(config.output_dir, exist_ok=True)
        split_cache = os.path.join(config.output_dir, "split_indices.json")

        if os.path.exists(split_cache):
            print("Loading cached split indices so all experiments match.")
            with open(split_cache) as f:
                idx = json.load(f)
            train_idx, val_idx, test_idx = idx["train"], idx["val"], idx["test"]
        else:
            targets = [s[1] for s in full_ds.samples]
            indices = list(range(len(full_ds)))
            train_idx, temp_idx = train_test_split(
                indices, test_size=0.30, stratify=targets, random_state=config.seed
            )
            temp_targets = [targets[i] for i in temp_idx]
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.50, stratify=temp_targets, random_state=config.seed
            )
            with open(split_cache, "w") as f:
                json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)

        train_ds = Subset(full_ds, train_idx)
        val_ds = Subset(full_ds, val_idx)
        test_ds = Subset(full_ds, test_idx)

    train_ds = _maybe_subset(train_ds, config.debug_max_train)
    val_ds = _maybe_subset(val_ds, config.debug_max_val)

    return train_ds, val_ds, test_ds, class_names


def build_loaders(
    train_ds: Dataset, val_ds: Dataset, test_ds: Dataset, batch_size: int, num_workers: int
):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader
