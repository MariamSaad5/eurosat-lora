import argparse
import os

import pandas as pd
import torch

from src.config import load_config
from src.data import build_datasets, build_loaders
from src.finetuner import LoRAFineTuner


def main():
    parser = argparse.ArgumentParser(description="Run one LoRA fine-tuning experiment from a config file.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file, e.g. configs/lora_r16.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds, class_names = build_datasets(config)
    print(f"Classes (order = label ids): {class_names}")
    print(f"Sizes -> train: {len(train_ds)} val: {len(val_ds)} test: {len(test_ds)}")

    train_loader, val_loader, _test_loader = build_loaders(
        train_ds, val_ds, test_ds, config.batch_size, config.num_workers
    )

    finetuner = LoRAFineTuner(config, device)
    result = finetuner.run(train_loader, val_loader, class_names)

    os.makedirs(config.output_dir, exist_ok=True)
    results_csv = os.path.join(config.output_dir, "results.csv")
    if os.path.exists(results_csv):
        existing = pd.read_csv(results_csv)
        updated = pd.concat([existing, pd.DataFrame([result])], ignore_index=True)
    else:
        updated = pd.DataFrame([result])
    updated.to_csv(results_csv, index=False)

    print(f"\nResult appended to {results_csv}")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
