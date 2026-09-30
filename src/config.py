from dataclasses import dataclass
from typing import List, Optional

import yaml


@dataclass
class LoRAConfig:
    r: int
    alpha: int
    target_blocks: List[int]
    target_modules: List[str]
    dropout: float = 0.0


@dataclass
class ExperimentConfig:
    name: str
    model_name: str
    data_dir: str
    output_dir: str
    seed: int = 42
    batch_size: int = 16
    epochs: int = 3
    num_workers: int = 2
    lr: float = 1e-3
    lora: Optional[LoRAConfig] = None
    # Optional debug knobs: subsample the dataset for a fast end-to-end
    # sanity run before committing to a full experiment. Leave unset (null
    # in the YAML) to use the full dataset.
    debug_max_train: Optional[int] = None
    debug_max_val: Optional[int] = None


def load_config(path: str) -> ExperimentConfig:
    """Reads a YAML file into an ExperimentConfig. The optional `lora:` block
    is parsed into a LoRAConfig; everything else maps directly onto
    ExperimentConfig's fields."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    lora_raw = raw.pop("lora", None)
    lora_cfg = LoRAConfig(**lora_raw) if lora_raw is not None else None

    return ExperimentConfig(lora=lora_cfg, **raw)
