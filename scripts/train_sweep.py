import argparse
import time
import numpy as np
import torch
import wandb

from torch.utils.data import DataLoader

from src.data_generation.load_data import OneStepTrajectoryDataset, resolve_split_npz_path
from src.eval.sweep_utils import (
    maybe_set_z_scale,
    build_run_name,
    build_model,
    compute_loader_loss_and_rmse,
    compute_rollout_metrics,
)


EVAL_HORIZONS = [10, 50, 100, 200, 500]
SELECTION_HORIZON = 100  # choose 100, 200, or 500 depending on what you want to optimize


def fmt_metric(x):
    return f"{x:.6e}" if x is not None else "None"


def print_multi_horizon_metrics(metrics_dict, gamma, prefix=""):
    if metrics_dict is None:
        print(f"{prefix}No rollout metrics.")
        return

    for h in EVAL_HORIZONS:
        rmse = metrics_dict.get(f"rollout_rmse_h{h}")
        nrmse = metrics_dict.get(f"rollout_nrmse_h{h}")
        w_nrmse = metrics_dict.get(f"weighted_cumulative_nrmse_h{h}_g{gamma:.2f}")

        print(
            f"{prefix}h={h:3d} | "
            f"RMSE: {fmt_metric(rmse)} | "
            f"NRMSE: {fmt_metric(nrmse)} | "
            f"W-NRMSE: {fmt_metric(w_nrmse)}"
        )


def add_prefixed_metrics(log_dict, metrics_dict, prefix):
    if metrics_dict is None:
        return

    for k, v in metrics_dict.items():
        log_dict[f"{prefix}{k}"] = v


def get_selection_metric(metrics_dict, gamma, selection_horizon):
    if metrics_dict is None:
        return None
    return metrics_dict.get(
        f"weighted_cumulative_nrmse_h{selection_horizon}_g{gamma:.2f}"
    )

def main():
    parser = argparse.ArgumentParser(description="Fast W&B sweep training for Koopman models")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "ml_linear_dynamics",
            "ml_dmd",
        ],
    )
    parser.add_argument("--data_path", type=str, required=True)
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
    parser.add_argument("--rollout_horizon", type=int, default=500)
    parser.add_argument("--rollout_gamma", type=float, default=0.99)

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
    test_data_path = resolve_split_npz_path(args.data_path, "test")

    val_data = np.load(val_data_path)
    test_data = np.load(test_data_path)

    val_X = val_data["X"]
    test_X = test_data["X"]

    print(f"System: {system_name}")
    print(f"State dim: {state_dim}")
    print(f"Train trajectory tensor shape: {train_X.shape}")
    print(f"Val trajectory tensor shape:   {val_X.shape}")
    print(f"Test trajectory tensor shape:  {test_X.shape}")

    # --------------------------------------------------
    # W&B init
    # --------------------------------------------------
    selection_metric_name = f"val_weighted_cumulative_nrmse_h{SELECTION_HORIZON}_g{args.rollout_gamma:.2f}"

    run = wandb.init(
        project="koopman-operator-learning",
        config=vars(args),
        group=f"{system_name}_{args.model}",
        tags=[system_name, args.model, args.expansion_type, "short"],
    )
    wandb.define_metric(selection_metric_name, summary="min")

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
            "selection_horizon": SELECTION_HORIZON,
            "selection_metric_name": selection_metric_name,
            "trajectory_length": "short"
        },
        allow_val_change=True,
    )

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    train_ds = OneStepTrajectoryDataset(args.data_path, split="train", subset=args.subset)
    val_ds = OneStepTrajectoryDataset(args.data_path, split="val", subset=args.subset)
    test_ds = OneStepTrajectoryDataset(args.data_path, split="test", subset=args.subset)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")

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

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    ) if len(test_ds) > 0 else None

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

    maybe_set_z_scale(model, train_loader, device)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=100,
        gamma=0.5,
    )

    loss_fn = torch.nn.MSELoss()

    best_metric = np.inf
    best_epoch = -1
    best_state = None

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"\n===== EPOCH {epoch + 1}/{args.epochs} =====")

        model.train()
        train_loss_sum = 0.0
        n_train = 0

        print("Training...")
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            if hasattr(model, "compute_loss"):
                loss_out = model.compute_loss(x, y)
                loss = loss_out if isinstance(loss_out, torch.Tensor) else sum(loss_out)
            else:
                y_hat = model(x)
                loss = loss_fn(y_hat, y)

            loss.backward()
            optimizer.step()

            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            n_train += bs

            if args.print_every_batch > 0 and batch_idx % args.print_every_batch == 0:
                print(
                    f"  batch {batch_idx:04d}/{len(train_loader):04d} "
                    f"| loss {loss.item():.6e}"
                )

        train_loss = train_loss_sum / max(n_train, 1)
        print(f"Train loss: {train_loss:.6e}")

        # one-step validation every epoch
        print("Computing validation one-step metrics...")
        val_loss, val_one_step_rmse = compute_loader_loss_and_rmse(model, val_loader, device)

        print(f"Val loss:          {fmt_metric(val_loss)}")
        print(f"Val one-step RMSE: {fmt_metric(val_one_step_rmse)}")

        # rollout validation
        do_rollout_eval = ((epoch + 1) % args.eval_every == 0) or (epoch == args.epochs - 1)

        val_rollout_metrics = None

        if do_rollout_eval:
            print("Computing validation rollout metrics...")
            t0 = time.time()

            val_rollout_metrics = compute_rollout_metrics(
                model=model,
                X=val_X,
                device=device,
                horizon=args.rollout_horizon,
                gamma=args.rollout_gamma,
                max_trajs=args.max_val_rollout_trajs,
            )

            print_multi_horizon_metrics(
                val_rollout_metrics,
                gamma=args.rollout_gamma,
                prefix="val ",
            )
            print(f"Rollout eval time: {time.time() - t0:.1f}s")
        else:
            print("Skipping rollout eval this epoch.")

        log_dict = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
        }

        if val_loss is not None:
            log_dict["val_loss"] = val_loss

        if val_one_step_rmse is not None:
            log_dict["val_one_step_rmse"] = val_one_step_rmse

        if val_rollout_metrics is not None:
            add_prefixed_metrics(log_dict, val_rollout_metrics, "val_")

            selection_metric = get_selection_metric(
                val_rollout_metrics,
                gamma=args.rollout_gamma,
                selection_horizon=SELECTION_HORIZON,
            )

            if selection_metric is not None:
                log_dict[selection_metric_name] = selection_metric

            if selection_metric is not None and selection_metric < best_metric:
                print(f"New best model based on {selection_metric_name}.")
                best_metric = selection_metric
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

        wandb.log(log_dict, step=epoch)
        scheduler.step()

        print(f"Epoch time: {time.time() - epoch_start:.1f}s")

    # --------------------------------------------------
    # Restore best model
    # --------------------------------------------------
    if best_state is not None:
        print(f"\nRestoring best model from epoch {best_epoch}...")
        model.load_state_dict(best_state)
    else:
        print("\nNo rollout checkpoint was saved. Keeping last epoch weights.")

    model.eval()

    wandb.log({
        "best_epoch": best_epoch,
        f"best_{selection_metric_name}": float(best_metric) if np.isfinite(best_metric) else None,
    })

    # --------------------------------------------------
    # Final validation
    # --------------------------------------------------
    print("\n===== FINAL VALIDATION =====")
    val_final_loss, val_final_one_step_rmse = compute_loader_loss_and_rmse(model, val_loader, device)

    val_final_rollout_metrics = compute_rollout_metrics(
        model=model,
        X=val_X,
        device=device,
        horizon=args.rollout_horizon,
        gamma=args.rollout_gamma,
        max_trajs=args.max_val_rollout_trajs,
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
        "val_final_one_step_rmse": val_final_one_step_rmse,
    }
    add_prefixed_metrics(val_final_log, val_final_rollout_metrics, "val_final_")
    wandb.log(val_final_log)

    # --------------------------------------------------
    # Final test
    # --------------------------------------------------
    print("\n===== FINAL TEST =====")
    test_loss, test_one_step_rmse = compute_loader_loss_and_rmse(model, test_loader, device)

    test_rollout_metrics = compute_rollout_metrics(
        model=model,
        X=test_X,
        device=device,
        horizon=args.rollout_horizon,
        gamma=args.rollout_gamma,
        max_trajs=args.max_test_rollout_trajs,
    )

    print(f"test_loss:          {fmt_metric(test_loss)}")
    print(f"test_one_step_rmse: {fmt_metric(test_one_step_rmse)}")
    print_multi_horizon_metrics(
        test_rollout_metrics,
        gamma=args.rollout_gamma,
        prefix="test ",
    )

    test_log = {
        "test_loss": test_loss,
        "test_one_step_rmse": test_one_step_rmse,
    }
    add_prefixed_metrics(test_log, test_rollout_metrics, "test_")
    wandb.log(test_log)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary[f"best_{selection_metric_name}"] = (
        float(best_metric) if np.isfinite(best_metric) else None
    )

    wandb.summary["val_final_one_step_rmse"] = val_final_one_step_rmse
    wandb.summary["test_one_step_rmse"] = test_one_step_rmse

    if val_final_rollout_metrics is not None:
        for k, v in val_final_rollout_metrics.items():
            wandb.summary[f"val_final_{k}"] = v

    if test_rollout_metrics is not None:
        for k, v in test_rollout_metrics.items():
            wandb.summary[f"test_{k}"] = v

    wandb.finish()


if __name__ == "__main__":
    main()