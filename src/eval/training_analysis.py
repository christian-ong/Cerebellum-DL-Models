import os
from typing import Dict, Any, Optional

import numpy as np


def load_losses(loss_file: str) -> Dict[str, Any]:
    data = np.load(loss_file, allow_pickle=True)

    out = {
        "train_losses": np.asarray(data["train_losses"], dtype=float),
        "batch_val_losses": np.asarray(data["batch_val_losses"], dtype=float),
        "epoch_val_losses": np.asarray(data["epoch_val_losses"], dtype=float),
    }

    if "loss_components_val" in data:
        try:
            out["loss_components_val"] = data["loss_components_val"].item()
        except Exception:
            out["loss_components_val"] = None
    else:
        out["loss_components_val"] = None

    return out


def summarize_losses(losses: Dict[str, Any]) -> Dict[str, np.ndarray]:
    train_losses = losses["train_losses"]
    batch_val_losses = losses["batch_val_losses"]
    epoch_val_losses = losses["epoch_val_losses"]

    best_epoch = int(np.argmin(epoch_val_losses)) if len(epoch_val_losses) > 0 else -1
    best_val = float(np.min(epoch_val_losses)) if len(epoch_val_losses) > 0 else np.nan
    final_val = float(epoch_val_losses[-1]) if len(epoch_val_losses) > 0 else np.nan
    final_train = float(train_losses[-1]) if len(train_losses) > 0 else np.nan

    summary = {
        "n_train_steps": np.array(len(train_losses)),
        "n_batch_val_steps": np.array(len(batch_val_losses)),
        "n_epochs": np.array(len(epoch_val_losses)),
        "best_epoch": np.array(best_epoch),
        "best_epoch_val_loss": np.array(best_val),
        "final_epoch_val_loss": np.array(final_val),
        "final_train_loss": np.array(final_train),
    }

    if len(epoch_val_losses) >= 2:
        summary["val_loss_drop_abs"] = np.array(float(epoch_val_losses[0] - epoch_val_losses[-1]))
        summary["val_loss_drop_rel"] = np.array(
            float((epoch_val_losses[0] - epoch_val_losses[-1]) / max(abs(epoch_val_losses[0]), 1e-12))
        )
    else:
        summary["val_loss_drop_abs"] = np.array(np.nan)
        summary["val_loss_drop_rel"] = np.array(np.nan)

    loss_components = losses.get("loss_components_val", None)
    if isinstance(loss_components, dict):
        for key, values in loss_components.items():
            values = np.asarray(values, dtype=float)
            if len(values) == 0:
                continue
            safe_key = key.replace(" ", "_")
            summary[f"{safe_key}_best"] = np.array(float(np.min(values)))
            summary[f"{safe_key}_final"] = np.array(float(values[-1]))

    return summary


def save_training_summary_npz(out_path: str, summary: Dict[str, np.ndarray]):
    np.savez(out_path, **summary)


def infer_loss_file_from_model_path(model_path: str) -> str:
    return os.path.join(os.path.dirname(model_path), "losses.npz")