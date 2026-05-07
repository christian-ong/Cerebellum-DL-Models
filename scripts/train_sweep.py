import argparse
import os
import time
import numpy as np
import torch
import wandb

from torch.utils.data import DataLoader

from src.data_generation.load_data import OneStepTrajectoryDataset, resolve_split_npz_path
from src.eval.sweep_utils import (
    build_run_name,
    build_model,
    compute_loader_metrics,
    compute_rollout_metrics,
)
from src.train.train_onestep import train_onestep

EVAL_HORIZONS = [10, 100, 500]

def fmt_metric(x):
    return f"{x:.6e}" if x is not None else "None"

def print_multi_horizon_metrics(metrics_dict, gamma, prefix=""):
    if metrics_dict is None:
        print(f"{prefix}No rollout metrics.")
        return

    for h in EVAL_HORIZONS:
        rmse = metrics_dict.get(f"rollout_rmse_h{h}")
        w_rmse = metrics_dict.get(f"discounted_mean_rmse_h{h}_g{gamma:.2f}")

        print(
            f"{prefix}h={h:3d} | "
            f"RMSE: {fmt_metric(rmse)} | "
            f"W-RMSE: {fmt_metric(w_rmse)} | "
        )

def add_prefixed_metrics(log_dict, metrics_dict, prefix):
    if metrics_dict is None:
        return

    for k, v in metrics_dict.items():
        log_dict[f"{prefix}{k}"] = v

def _is_scalar_number(v):
    return isinstance(v, (int, float, np.floating)) and np.isfinite(v)

def update_best_metrics(best_metrics, current_metrics, current_epoch):
    for key, value in current_metrics.items():
        if not _is_scalar_number(value):
            continue
        if key not in best_metrics or value < best_metrics[key]["value"]:
            best_metrics[key] = {"value": float(value), "epoch": int(current_epoch)}

def main():
    parser = argparse.ArgumentParser(description="Fast W&B sweep training for Koopman models")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "ml_linear_dynamics",
            "ml_dmd_free",
            "ml_dmd_band",
        ],
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--trajectory_length", type=str, choices=["short", "long"], default="long")
    parser.add_argument("--name", type=str, default="run")

    # training
    parser.add_argument("--subset", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)

    # model hyperparameters
    parser.add_argument("--bias", type=str.lower, choices=["true", "false"], default="true")
    parser.add_argument("--expansion_type", type=str, choices=["general", "specific"], default="general")
    parser.add_argument("--expansion_degree", type=int, default=3)
    parser.add_argument("--sine_cosine_expansion", type=str.lower, choices=["true", "false"], default="false")

    # validation / selection
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--max_val_rollout_trajs", type=int, default=None)
    parser.add_argument("--max_test_rollout_trajs", type=int, default=None)
    parser.add_argument("--rollout_horizon", type=int, default=20, help="Training rollout horizon for rollout loss (steps)")
    parser.add_argument("--rollout_gamma", type=float, default=0.99, help="Discount factor for eval")

    # misc
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--print_every_batch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --------------------------------------------------
    # Load metadata
    # --------------------------------------------------
    train_meta_path = resolve_split_npz_path(args.data_path, "train")

    meta = np.load(train_meta_path, allow_pickle=True)
    system_name = str(meta["system"])
    train_X = meta["X"]
    state_dim = train_X.shape[-1]

    if train_X.ndim != 3:
        raise ValueError("Expected X to have shape (T, n_traj, d).")

    val_data_path = resolve_split_npz_path(args.data_path, "val")

    val_data = np.load(val_data_path)

    val_X = val_data["X"]

    print(f"System: {system_name}")
    print(f"State dim: {state_dim}")
    print(f"Train trajectory tensor shape: {train_X.shape}")
    print(f"Val trajectory tensor shape:   {val_X.shape}")

    # --------------------------------------------------
    # W&B init
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    train_ds = OneStepTrajectoryDataset(args.data_path, split="train", subset=args.subset, rollout_horizon=args.rollout_horizon)
    val_ds = OneStepTrajectoryDataset(args.data_path, split="val", subset=args.subset, rollout_horizon=args.rollout_horizon)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    ) if len(val_ds) > 0 else None

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = build_model(args, state_dim, system_name, device)

    if hasattr(model, "expansion_type"):
        print(f"Expansion type: {args.expansion_type}")
    if hasattr(model, "expansion_degree"):
        print(f"Expansion degree: {args.expansion_degree}")
    if hasattr(model, "expand_names"):
        print(f"Expanded features: {len(model.expand_names)}")
        print(model.expand_names)

    # --------------------------------------------------
    # Training (via train_onestep for consistency)
    # --------------------------------------------------
    print("\n===== TRAINING =====")
    t_train_start = time.time()
    
    model, (train_losses, batch_val_losses, epoch_val_losses, loss_components_val), best_checkpoint = train_onestep(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_phi_every=0,  # Set to > 0 if you want per-epoch matrix logging
    )
    
    print(f"Training completed in {time.time() - t_train_start:.1f}s")
    
    # Load best checkpoint for rollout evaluation
    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint["state_dict"])
    model.eval()

    # --------------------------------------------------
    # Final validation (last epoch weights)
    # --------------------------------------------------
    print("\n===== FINAL EPOCH VALIDATION =====")
    val_final_loss, val_final_one_step_rmse = compute_loader_metrics(
        model, val_loader, device
    )

    val_final_rollout_metrics = compute_rollout_metrics(
        model=model,
        X=val_X,
        device=device,
        horizon=args.rollout_horizon,
        gamma=args.rollout_gamma,
        max_trajs=args.max_val_rollout_trajs
    )

    print(f"val_final_loss:          {fmt_metric(val_final_loss)}")
    print(f"val_final_one_step_rmse: {fmt_metric(val_final_one_step_rmse)}")
    print_multi_horizon_metrics(
        val_final_rollout_metrics,
        gamma=args.rollout_gamma,
        prefix="val_final ",
    )

    val_final_log = {
        "val_final_loss": val_final_loss,
        "val_final_one_step_rmse": val_final_one_step_rmse
    }
    add_prefixed_metrics(val_final_log, val_final_rollout_metrics, "val_final_")
    wandb.log(val_final_log)

    # --------------------------------------------------
    # Save best checkpoint (matching train.py behavior)
    # --------------------------------------------------
    if best_checkpoint is not None:
        save_dir = f"data/models/{args.model}/{system_name}/{run.name}"
        os.makedirs(save_dir, exist_ok=True)
        best_save_path = os.path.join(save_dir, "model_best.pt")
        torch.save(
            {
                "model": args.model,
                "model_state_dict": best_checkpoint["state_dict"],
                "train_args": vars(args),
                "state_dim": state_dim,
                "system": system_name,
                "best_epoch": best_checkpoint["epoch"],
                "best_val_loss": best_checkpoint["val_loss"],
            },
            best_save_path,
        )
        print(f"Saved best checkpoint to {best_save_path}")

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    wandb.summary["val_final_one_step_rmse"] = val_final_one_step_rmse

    wandb.summary["val_final_loss"] = val_final_loss

    if best_checkpoint is not None:
        wandb.summary["best_epoch"] = best_checkpoint["epoch"]
        wandb.summary["best_val_loss"] = best_checkpoint["val_loss"]

    if val_final_rollout_metrics is not None:
        for k, v in val_final_rollout_metrics.items():
            wandb.summary[f"val_final_{k}"] = v

    wandb.finish()


if __name__ == "__main__":
    main()