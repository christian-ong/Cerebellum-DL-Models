import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from src.data_generation.load_data import OneStepTrajectoryDataset
from src.models.ml_dmd import ML_DMD
from src.models.ml_eigen_dmd import MLEigenDMD
from src.models.manual_expansion_ml_dmd import ManualExpansion_MLDMD
from src.models.manual_expansion_eigen_dmd import ManualExpansion_EigenDMD

from src.eval.metrics import (
    compute_one_step_metrics,
    compute_horizon_metrics,
    compute_full_rollout_metrics,
    compute_composite_validation_score,
    get_state_scale_from_train_split,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def parse_int_list(text: str):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("At least one horizon must be provided.")
    return tuple(sorted(set(values)))


def maybe_set_z_scale(model, train_loader, device):
    if hasattr(model, "expand") and hasattr(model, "set_z_scale"):
        with torch.no_grad():
            zs = []
            for x_batch, _ in train_loader:
                x_batch = x_batch.to(device)
                z_batch = model.expand(x_batch)
                zs.append(z_batch)

            if len(zs) > 0:
                z_all = torch.cat(zs, dim=0)
                z_scale = torch.mean(torch.abs(z_all), dim=0) + 1e-6
                model.set_z_scale(z_scale)


def compute_one_step_loss(model, loader, device):
    if loader is None or len(loader.dataset) == 0:
        return None

    model.eval()
    loss_fn = torch.nn.MSELoss()
    total = 0.0
    n = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if hasattr(model, "compute_loss"):
                loss_tuple = model.compute_loss(x, y)
                loss = loss_tuple if isinstance(loss_tuple, torch.Tensor) else sum(loss_tuple)
            else:
                y_hat = model(x)
                loss = loss_fn(y_hat, y)

            bs = x.size(0)
            total += loss.item() * bs
            n += bs

    return total / max(n, 1)


def build_run_name(args, system, run_id=None):
    parts = [system, args.model]
    parts.append(f"lr{args.lr:.0e}")
    parts.append(f"bs{args.batch_size}")

    if "manual_expansion" in args.model:
        parts.append(f"{args.expansion_type}")
        parts.append(f"deg{args.expansion_degree}")
        parts.append(f"trig{args.sine_cosine_expansion}")

    if run_id is not None:
        parts.append(run_id[:8])

    return "_".join(parts)


def build_model(args, state_dim, system_name, device):
    if args.model == "ml_dmd":
        model = ML_DMD(
            state_dim=state_dim,
        ).to(device)

    elif args.model == "ml_eigen_dmd":
        model = MLEigenDMD(
            state_dim=state_dim,
        ).to(device)

    elif args.model == "manual_expansion_ml_dmd":
        model = ManualExpansion_MLDMD(
            state_dim=state_dim,
            expansion_degree=args.expansion_degree,
            expansion_type=args.expansion_type,
            bias=args.bias == "true",
            sine_cosine_expansion=args.sine_cosine_expansion == "true",
            system=system_name,
        ).to(device)

    elif args.model == "manual_expansion_eigen_dmd":
        model = ManualExpansion_EigenDMD(
            state_dim=state_dim,
            expansion_degree=args.expansion_degree,
            bias=args.bias == "true",
            sine_cosine_expansion=args.sine_cosine_expansion == "true",
            expansion_type=args.expansion_type,
            system=system_name if args.expansion_type == "specific" else None,
        ).to(device)

    else:
        raise ValueError(
            "This sweep script only supports trainable neural models: "
            "ml_dmd, ml_eigen_dmd, manual_expansion_ml_dmd, manual_expansion_eigen_dmd."
        )

    return model


def compute_split_diagnostics(
    model,
    X,
    traj_idx,
    model_name,
    scale_std,
    horizons,
    rollout_horizons,
    max_one_step_pairs_per_traj=None,
    max_horizon_starts_per_traj=None,
):
    if len(traj_idx) == 0:
        return None

    one_step_metrics = compute_one_step_metrics(
        X=X,
        traj_indices=traj_idx,
        model_name=model_name,
        model=model,
        extras=None,
        scale_std=scale_std,
        max_pairs_per_traj=max_one_step_pairs_per_traj,
    )

    horizon_metrics = compute_horizon_metrics(
        X=X,
        traj_indices=traj_idx,
        horizons=list(horizons),
        model_name=model_name,
        model=model,
        extras=None,
        scale_std=scale_std,
        max_starts_per_traj=max_horizon_starts_per_traj,
    )

    rollout_metrics = compute_full_rollout_metrics(
        X=X,
        traj_indices=traj_idx,
        rollout_horizons=list(rollout_horizons),
        model_name=model_name,
        model=model,
        extras=None,
        scale_std=scale_std,
    )

    score = compute_composite_validation_score(
        one_step_nrmse=float(one_step_metrics["one_step_nrmse"]),
        horizon_nrmse=horizon_metrics["horizon_nrmse"],
        rollout_nrmse=rollout_metrics["rollout_nrmse"],
    )

    horizon_nrmse_map = {}
    if "horizons" in horizon_metrics:
        for h, v in zip(horizon_metrics["horizons"], horizon_metrics["horizon_nrmse"]):
            horizon_nrmse_map[int(h)] = float(v)

    rollout_nrmse_map = {}
    if "rollout_horizons" in rollout_metrics:
        for h, v in zip(rollout_metrics["rollout_horizons"], rollout_metrics["rollout_nrmse"]):
            rollout_nrmse_map[int(h)] = float(v)

    return {
        "one_step_mse": float(one_step_metrics["one_step_mse"]),
        "one_step_rmse": float(one_step_metrics["one_step_rmse"]),
        "one_step_nrmse": float(one_step_metrics["one_step_nrmse"]),
        "horizon_nrmse_mean": float(np.mean(horizon_metrics["horizon_nrmse"])),
        "rollout_nrmse_mean": float(np.mean(rollout_metrics["rollout_nrmse"])),
        "horizon_nrmse_map": horizon_nrmse_map,
        "rollout_nrmse_map": rollout_nrmse_map,
        "score": float(score),
    }


def add_selected_logs(log_dict, prefix, diagnostics, selected_horizons, selected_rollout_horizons):
    if diagnostics is None:
        return

    log_dict[f"{prefix}_one_step_nrmse"] = diagnostics["one_step_nrmse"]
    log_dict[f"{prefix}_horizon_nrmse_mean"] = diagnostics["horizon_nrmse_mean"]
    log_dict[f"{prefix}_rollout_nrmse_mean"] = diagnostics["rollout_nrmse_mean"]
    log_dict[f"{prefix}_score"] = diagnostics["score"]

    for h in selected_horizons:
        if h in diagnostics["horizon_nrmse_map"]:
            log_dict[f"{prefix}_horizon_nrmse_{h}"] = diagnostics["horizon_nrmse_map"][h]

    for h in selected_rollout_horizons:
        if h in diagnostics["rollout_nrmse_map"]:
            log_dict[f"{prefix}_rollout_nrmse_{h}"] = diagnostics["rollout_nrmse_map"][h]


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sweep-only train + eval script for W&B")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "ml_dmd",
            "ml_eigen_dmd",
            "manual_expansion_ml_dmd",
            "manual_expansion_eigen_dmd",
        ],
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--name", type=str, default="run")

    # training
    parser.add_argument("--subset", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)

    # model hyperparameters
    parser.add_argument("--bias", type=str.lower, choices=["true", "false"], default="true")
    parser.add_argument("--expansion_type", type=str, default="general", choices=["general", "specific"])
    parser.add_argument("--expansion_degree", type=int, default=3)
    parser.add_argument("--sine_cosine_expansion", type=str.lower, choices=["true", "false"], default="false")

    # eval
    parser.add_argument("--val_horizons", type=str, default="10,50,200")
    parser.add_argument("--val_rollout_horizons", type=str, default="10,50,200")  
    parser.add_argument("--max_one_step_pairs_per_traj", type=int, default=None)
    parser.add_argument("--max_horizon_starts_per_traj", type=int, default=None)

    # misc
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    val_horizons = parse_int_list(args.val_horizons)
    val_rollout_horizons = parse_int_list(args.val_rollout_horizons)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # --------------------------------------------------
    # Load metadata
    # --------------------------------------------------

    meta = np.load(args.data_path, allow_pickle=True)
    system_name = str(meta["system"])
    X = meta["X"]
    state_dim = X.shape[-1]

    if X.ndim != 3:
        raise ValueError("Expected X to have shape (T, n_traj, d).")

    if "val_idx" not in meta or "test_idx" not in meta:
        raise ValueError("Dataset must contain val_idx and test_idx.")

    val_idx = meta["val_idx"]
    test_idx = meta["test_idx"]

    max_needed = max(max(val_horizons), max(val_rollout_horizons))
    if X.shape[0] <= max_needed:
        raise ValueError(
            f"Trajectory length T={X.shape[0]} is too short for requested max horizon {max_needed}."
        )

    scales = get_state_scale_from_train_split(args.data_path)
    scale_std = scales["std"]

    # --------------------------------------------------
    # W&B init
    # --------------------------------------------------

    run = wandb.init(
        project="koopman-operator-learning",
        config=vars(args),
        group=f"{system_name}_{args.model}",
        tags=[system_name, args.model],
    )

    config = wandb.config
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    # in case sweep overrides horizon args
    val_horizons = parse_int_list(args.val_horizons)
    val_rollout_horizons = parse_int_list(args.val_rollout_horizons)

    run.name = build_run_name(args, system_name, run.id)

    wandb.config.update(
        {
            **vars(args),
            "system": system_name,
            "state_dim": int(state_dim),
            "group_name": f"{system_name}_{args.model}",
            "model_name": args.model,
            "system_name": system_name,
        },
        allow_val_change=True,
    )

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    train_ds = OneStepTrajectoryDataset(
        args.data_path,
        split="train",
        subset=args.subset,
    )
    val_ds = OneStepTrajectoryDataset(
        args.data_path,
        split="val",
        subset=args.subset,
    )
    test_ds = OneStepTrajectoryDataset(
        args.data_path,
        split="test",
        subset=args.subset,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
    ) if len(val_ds) > 0 else None
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
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
        print(f"Expand names: {model.expand_names}")

    maybe_set_z_scale(model, train_loader, device)

    # --------------------------------------------------
    # Training
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

    best_score = np.inf
    best_state = None
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()

        train_loss_sum = 0.0
        n_train = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            if hasattr(model, "compute_loss"):
                loss_tuple = model.compute_loss(x, y)
                loss = loss_tuple if isinstance(loss_tuple, torch.Tensor) else sum(loss_tuple)
            else:
                y_hat = model(x)
                loss = loss_fn(y_hat, y)

            loss.backward()
            optimizer.step()

            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            n_train += bs

        train_loss = train_loss_sum / max(n_train, 1)
        val_loss = compute_one_step_loss(model, val_loader, device)

        model.eval()

        val_diagnostics = compute_split_diagnostics(
            model=model,
            X=X,
            traj_idx=val_idx,
            model_name=args.model,
            scale_std=scale_std,
            horizons=val_horizons,
            rollout_horizons=val_rollout_horizons,
            max_one_step_pairs_per_traj=args.max_one_step_pairs_per_traj,
            max_horizon_starts_per_traj=args.max_horizon_starts_per_traj,
        )

        log_dict = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
        }

        if val_loss is not None:
            log_dict["val_loss"] = val_loss

        if val_diagnostics is not None:
            add_selected_logs(
                log_dict,
                "val",
                val_diagnostics,
                selected_horizons=val_horizons,
                selected_rollout_horizons=val_rollout_horizons,
            )
            log_dict["score"] = float(val_diagnostics["score"])

            if float(val_diagnostics["score"]) < best_score:
                best_score = float(val_diagnostics["score"])
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

        wandb.log(log_dict, step=epoch)
        scheduler.step()

        if val_loss is not None and val_diagnostics is not None:
            print(
                f"Epoch {epoch:03d} | "
                f"train {train_loss:.6e} | "
                f"val {val_loss:.6e} | "
                f"val_score {float(val_diagnostics['score']):.6e}"
            )
        elif val_loss is not None:
            print(
                f"Epoch {epoch:03d} | "
                f"train {train_loss:.6e} | "
                f"val {val_loss:.6e}"
            )
        else:
            print(f"Epoch {epoch:03d} | train {train_loss:.6e}")

    # --------------------------------------------------
    # Restore best checkpoint in memory
    # --------------------------------------------------

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    wandb.log(
        {
            "best_epoch": best_epoch,
            "best_val_score": float(best_score) if np.isfinite(best_score) else None,
        }
    )

    # --------------------------------------------------
    # Final validation
    # --------------------------------------------------

    final_val_loss = compute_one_step_loss(model, val_loader, device)
    final_val_diagnostics = compute_split_diagnostics(
        model=model,
        X=X,
        traj_idx=val_idx,
        model_name=args.model,
        scale_std=scale_std,
        horizons=val_horizons,
        rollout_horizons=val_rollout_horizons,
        max_one_step_pairs_per_traj=args.max_one_step_pairs_per_traj,
        max_horizon_starts_per_traj=args.max_horizon_starts_per_traj,
    )

    final_val_log = {}
    if final_val_loss is not None:
        final_val_log["val_final_loss"] = final_val_loss
    add_selected_logs(
        final_val_log,
        "val_final",
        final_val_diagnostics,
        selected_horizons=val_horizons,
        selected_rollout_horizons=val_rollout_horizons,
    )
    wandb.log(final_val_log)

    # --------------------------------------------------
    # Final test
    # --------------------------------------------------

    final_test_loss = compute_one_step_loss(model, test_loader, device)
    final_test_diagnostics = compute_split_diagnostics(
        model=model,
        X=X,
        traj_idx=test_idx,
        model_name=args.model,
        scale_std=scale_std,
        horizons=val_horizons,
        rollout_horizons=val_rollout_horizons,
        max_one_step_pairs_per_traj=args.max_one_step_pairs_per_traj,
        max_horizon_starts_per_traj=args.max_horizon_starts_per_traj,
    )

    final_test_log = {}
    if final_test_loss is not None:
        final_test_log["test_loss"] = final_test_loss
    add_selected_logs(
        final_test_log,
        "test_final",
        final_test_diagnostics,
        selected_horizons=val_horizons,
        selected_rollout_horizons=val_rollout_horizons,
    )
    if final_test_diagnostics is not None:
        final_test_log["test_score"] = float(final_test_diagnostics["score"])
    wandb.log(final_test_log)

    print("\n=== FINAL VALIDATION METRICS ===")
    if final_val_loss is not None:
        print(f"val_final_loss: {final_val_loss:.4e}")
    if final_val_diagnostics is not None:
        print(f"val_final_score: {final_val_diagnostics['score']:.4e}")
        print(f"val_final_one_step_nrmse: {final_val_diagnostics['one_step_nrmse']:.4e}")
        print(f"val_final_horizon_nrmse_mean: {final_val_diagnostics['horizon_nrmse_mean']:.4e}")
        print(f"val_final_rollout_nrmse_mean: {final_val_diagnostics['rollout_nrmse_mean']:.4e}")

    print("\n=== FINAL TEST METRICS ===")
    if final_test_loss is not None:
        print(f"test_loss: {final_test_loss:.4e}")
    if final_test_diagnostics is not None:
        print(f"test_score: {final_test_diagnostics['score']:.4e}")
        print(f"test_final_one_step_nrmse: {final_test_diagnostics['one_step_nrmse']:.4e}")
        print(f"test_final_horizon_nrmse_mean: {final_test_diagnostics['horizon_nrmse_mean']:.4e}")
        print(f"test_final_rollout_nrmse_mean: {final_test_diagnostics['rollout_nrmse_mean']:.4e}")

    wandb.finish()


if __name__ == "__main__":
    main()