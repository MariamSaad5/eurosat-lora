import os
import time
from dataclasses import asdict

import torch
import torch.nn as nn
from transformers import ViTForImageClassification

from .config import ExperimentConfig
from .lora import apply_lora_to_vit
from .metrics import count_parameters, get_adapter_state_dict, state_dict_size_mb
from .train import run_one_epoch
from .utils import set_seed


class BaseFineTuner:
    """Owns the whole train/evaluate/checkpoint workflow. This part is
    identical no matter which fine-tuning strategy is used, so it lives here
    once. Subclasses only need to define:

      - build_model(class_names)          how the model is constructed
      - get_checkpoint_state_dict(model)  what gets saved to disk
      - extra_result_fields()             extra columns for the results row

    run() calls those three hooks but never needs to know which strategy
    it's dealing with - that's the whole point of the split.
    """

    mode_name = "base"

    def __init__(self, config: ExperimentConfig, device: torch.device):
        self.config = config
        self.device = device

    def build_model(self, class_names: list) -> nn.Module:
        raise NotImplementedError("Subclasses must implement build_model()")

    def get_checkpoint_state_dict(self, model: nn.Module) -> dict:
        """Default: save the whole model. Strategies that only train a small
        subset of parameters (e.g. LoRA) should override this."""
        return model.state_dict()

    def extra_result_fields(self) -> dict:
        """Hook for subclasses to add extra columns to the results row
        (e.g. LoRA rank) without touching run()."""
        return {}

    def extra_checkpoint_fields(self) -> dict:
        """Hook for subclasses to store whatever is needed to rebuild the
        model later (e.g. LoRA rank and target layers) inside the checkpoint."""
        return {}

    def run(self, train_loader, val_loader, class_names: list, verbose: bool = True) -> dict:
        cfg = self.config
        if verbose:
            print(f"\n{'=' * 60}\nRunning experiment: {cfg.name} ({self.mode_name})\n{'=' * 60}")
        set_seed(cfg.seed)

        model = self.build_model(class_names).to(self.device)
        total_params, trainable_params = count_parameters(model)
        if verbose:
            print(
                f"Total params: {total_params:,} | Trainable: {trainable_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
            )

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        train_losses, val_losses, val_accs, val_f1s = [], [], [], []
        start_time = time.time()

        # Best-model tracking: keep the checkpoint from the epoch with the
        # lowest validation loss, not just whatever the last epoch produced.
        os.makedirs(cfg.output_dir, exist_ok=True)
        best_ckpt_path = os.path.join(cfg.output_dir, f"{cfg.name}_best.pt")
        best_val_loss = float("inf")
        best_epoch = None

        for epoch in range(cfg.epochs):
            train_loss, _, _ = run_one_epoch(model, train_loader, self.device, optimizer)
            val_loss, val_acc, val_f1 = run_one_epoch(model, val_loader, self.device, optimizer=None)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_f1s.append(val_f1)
            if verbose:
                print(
                    f"Epoch {epoch + 1}/{cfg.epochs} | train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save(
                    {
                        "epoch": best_epoch,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "val_f1": val_f1,
                        "model_name": cfg.model_name,
                        "class_names": class_names,
                        "adapter_state_dict": self.get_checkpoint_state_dict(model),
                        **self.extra_checkpoint_fields(),
                    },
                    best_ckpt_path,
                )
                if verbose:
                    print(f"  -> new best model (val_loss={val_loss:.4f}), saved to {best_ckpt_path}")

        train_time_sec = time.time() - start_time
        gpu_mem_gb = (
            torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            if torch.cuda.is_available()
            else None
        )

        full_size_mb = state_dict_size_mb(model.state_dict())
        checkpoint_state = self.get_checkpoint_state_dict(model)
        checkpoint_size_mb = state_dict_size_mb(checkpoint_state)

        result = {
            "name": cfg.name,
            "mode": self.mode_name,
            "lr": cfg.lr,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_pct": 100 * trainable_params / total_params,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "val_accuracy": val_accs[-1],
            "val_f1": val_f1s[-1],
            "gpu_mem_gb": gpu_mem_gb,
            "train_time_sec": train_time_sec,
            "full_state_dict_size_mb": full_size_mb,
            "checkpoint_size_mb": checkpoint_size_mb,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "checkpoint_path": best_ckpt_path,
        }
        result.update(self.extra_result_fields())

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result


class LoRAFineTuner(BaseFineTuner):
    """LoRA-specific strategy: freeze the backbone, inject LoRA adapters into
    the configured blocks/modules, and only checkpoint the trainable pieces
    instead of the whole model."""

    mode_name = "lora"

    def build_model(self, class_names: list) -> nn.Module:
        cfg = self.config
        model = ViTForImageClassification.from_pretrained(
            cfg.model_name,
            num_labels=len(class_names),
            ignore_mismatched_sizes=True,
            id2label={i: c for i, c in enumerate(class_names)},
            label2id={c: i for i, c in enumerate(class_names)},
        )
        apply_lora_to_vit(
            model,
            target_blocks=cfg.lora.target_blocks,
            target_modules=cfg.lora.target_modules,
            r=cfg.lora.r,
            alpha=cfg.lora.alpha,
            dropout=cfg.lora.dropout,
        )
        return model

    def get_checkpoint_state_dict(self, model: nn.Module) -> dict:
        return get_adapter_state_dict(model)

    def extra_checkpoint_fields(self) -> dict:
        return {"lora_config": asdict(self.config.lora)}

    def extra_result_fields(self) -> dict:
        lora_cfg = self.config.lora
        return {
            "rank": lora_cfg.r,
            "alpha": lora_cfg.alpha,
            "target_modules": ",".join(lora_cfg.target_modules),
            "target_blocks": str(lora_cfg.target_blocks),
        }