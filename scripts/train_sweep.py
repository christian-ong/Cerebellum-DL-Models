import argparse
import os
import time
import numpy as np
import torch
import wandb
from pathlib import Path

from torch.utils.data import DataLoader

from src.data_generation.load_data import OneStepTrajectoryDataset, resolve_split_npz_path
from src.eval.sweep_utils import (
    build_run_name,
    build_model,
    compute_loader_metrics,
    compute_rollout_metrics,
)
from src.train.train_onestep_sweep import train_onestep_sweep

EVAL_HORIZONS = [5, 50, 100]

def fmt_metric(x):
    return f"{x:.6e}" if x is not None else "None"

def _is_scalar_number(v):
    return isinstance(v, (int, float, np.floating)) and np.isfinite(v)

def update_best_metrics(best_metrics, current_metrics, current_epoch):
    for key, value in current_metrics.items():
        if not _is_scalar_number(value):
            continue
        # For losses and errors, lower is better.
        if key not in best_metrics or value < best_metrics[key]["value"]:
            best_metrics[key] = {"value": float(value), "epoch": int(current_epoch)}

def main():
    parser = argparse.ArgumentParser(description="Fast W&B sweep training for Koopman models")

    parser.add_argument("--model", type=str, required=True, choices=["ml_linear_dynamics", "ml_dmd_free", "ml_dmd_band"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--trajectory_length", type=str, choices=["short", "long"], default="long")
    parser.add_argument("--name", type=str, default="run")

    parser.add_argument("--subset", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)

    parser.add_argument("--bias", type=str.lower, choices=["true", "false"], default="true")
    parser.add_argument("--expansion_type", type=str, choices=["general", "specific"], default="general")
    parser.add_argument("--expansion_degree", type=int, default=3)
    parser.add_argument("--sine_cosine_expansion", type=str.lower, choices=["true", "false"], default="false")

    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--max_val_rollout_trajs", type=int, default=None)
    parser.add_argument("--rollout_horizon", type=int, default=5, help="Training rollout horizon for rollout loss (steps)")
    parser.add_argument("--rollout_gamma", type=float, default=0.99, help="Discount factor for eval")
    parser.add_argument("--log_phi_every", type=int, default=1, help="Print get_Phi() every N epochs")
    parser.add_argument("--phi_print_max_dim", type=int, default=12, help="When Phi is larger than this, print only the top-left block")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--print_every_batch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load metadata
    train_meta_path = resolve_split_npz_path(args.data_path, "train")
    meta = np.load(train_meta_path, allow_pickle=True)
    system_name = str(meta["system"])
    train_X = meta["X"]
    state_dim = train_X.shape[-1]

    val_data_path = resolve_split_npz_path(args.data_path, "val")
    val_data = np.load(val_data_path)
    val_X = val_data["X"]

    # W&B init
    run = wandb.init(
        project="koopman-operator-learning",
        config=vars(args),
        group=f"{system_name}_{args.model}",
        tags=[system_name, args.model, args.expansion_type, args.trajectory_length],
    )

    config = wandb.config
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    run.name = build_run_name(args, system_name, run.id)

    wandb.config.update(
        {
            **vars(args),
            "system": system_name,
            "state_dim": int(state_dim),
            "group_name": f"{system_name}_{args.model}",
            "model_name": args.model,
            "system_name": system_name,
            "trajectory_length": args.trajectory_length
        },
        allow_val_change=True,
    )

    # Setup output directory (for checkpoint saving)
    save_dir = os.path.join("data/models", args.model, system_name, run.id)
    os.makedirs(save_dir, exist_ok=True)

    # Data
    train_ds = OneStepTrajectoryDataset(args.data_path, split="train", subset=args.subset, rollout_horizon=args.rollout_horizon)
    val_ds = OneStepTrajectoryDataset(args.data_path, split="val", subset=args.subset, rollout_horizon=args.rollout_horizon)

    pin_memory = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory) if len(val_ds) > 0 else None

    # Model
    model = build_model(args, state_dim, system_name, device)

    # --------------------------------------------------
    # Evaluation Callback setup
    # --------------------------------------------------
    best_metrics = {}

    # Inside main() of train_sweep.py
    def run_eval_callback(epoch, train_loss, val_loss):
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        # 1-Step RMSE
        if val_loader is not None:
            _, val_rmse = compute_loader_metrics(model, val_loader, device)
            metrics["val_onestep_rmse"] = val_rmse

        # Multi-step Rollout Metrics (5, 50, 100)
        rollout_metrics = compute_rollout_metrics(
            model=model,
            X=val_X,
            device=device,
            eval_horizons=EVAL_HORIZONS, # [5, 50, 100]
            gamma=args.rollout_gamma,
            max_trajs=args.max_val_rollout_trajs # Set this in your .sh for extra speed
        )
        
        if rollout_metrics is not None:
            for k, v in rollout_metrics.items():
                metrics[f"val_{k}"] = v

        # Log per-epoch charts
        wandb.log(metrics, step=epoch)

        # Track "best_XXX" for the final summary
        update_best_metrics(best_metrics, metrics, epoch)

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    print("\n===== TRAINING =====")
    t_train_start = time.time()
    
    model, (train_losses, epoch_val_losses, loss_components_val), best_checkpoint = train_onestep_sweep(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_phi_every=args.log_phi_every,
        phi_print_max_dim=args.phi_print_max_dim,
        eval_callback=run_eval_callback  # <--- PASSING THE CALLBACK
    )
    
    print(f"Training completed in {time.time() - t_train_start:.1f}s")
    
    # --------------------------------------------------
    # Save checkpoints (matching train.py logic)
    # --------------------------------------------------
    checkpoint_base = {
        "model": args.model,
        "system": system_name,
        "state_dim": int(state_dim),
        "train_args": vars(args),
        "data_path": args.data_path,
        "expand_names": model.expand_names if hasattr(model, "expand_names") else None,
        "best_epoch": best_checkpoint["epoch"],
        "best_val_loss": best_checkpoint["val_loss"],
    }

    best_save_path = os.path.join(save_dir, "model_best.pt")
    torch.save(
        {
            **checkpoint_base,
            "model_state_dict": best_checkpoint["state_dict"],
            "checkpoint_type": "best",
        },
        best_save_path,
    )

    last_save_path = os.path.join(save_dir, "model_last.pt")
    torch.save(
        {
            **checkpoint_base,
            "model_state_dict": model.state_dict(),
            "checkpoint_type": "last",
        },
        last_save_path,
    )

    # Keep backward-compatible default path, now pointing to best checkpoint.
    save_path = os.path.join(save_dir, "model.pt")
    torch.save(
        {
            **checkpoint_base,
            "model_state_dict": best_checkpoint["state_dict"],
            "checkpoint_type": "best",
        },
        save_path,
    )
    
    loss_path = os.path.join(save_dir, "losses.npz")
    np.savez(loss_path, train_losses=train_losses, epoch_val_losses=epoch_val_losses, loss_components_val=loss_components_val)
    
    print(f"Saved best model to {best_save_path}")
    print(f"Saved last model to {last_save_path}")
    print(f"Saved default model to {save_path}")
    print(f"Saved losses to {loss_path}")
    
    # --------------------------------------------------
    # Summary (Log all Best Metrics)
    # --------------------------------------------------
    # Dump out the 'best' version of everything we captured during epochs 
    # to W&B's summary pane so you can sort/filter by it later in the UI.
    for metric_name, data in best_metrics.items():
        wandb.summary[f"best_{metric_name}"] = data["value"]
        wandb.summary[f"best_{metric_name}_epoch"] = data["epoch"]

    wandb.finish()


if __name__ == "__main__":
    main()