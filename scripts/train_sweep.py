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
from src.eval.model_io import load_model as load_saved_model
from src.train.train_onestep_sweep import train_onestep_sweep
from src.models.regression_dmd import Regression_DMD
from src.models.linear_baseline import fit_linear_map, rollout_linear_map
from src.models.dmd_baseline import fit_dmd, rollout_dmd_eig

def prepare_ml_expander_and_lift_stats(
    *,
    model,
    train_ds,
    device,
    max_fit_samples: int = 0,
):
    """Fit data-dependent ML expanders (RBF/Hankel/Delay) and initialize lifted stats."""
    expander = getattr(model, "expander", None)
    if expander is None:
        return

    if not hasattr(train_ds, "x"):
        return

    X_fit = train_ds.x
    if not torch.is_tensor(X_fit):
        X_fit = torch.as_tensor(X_fit)

    if max_fit_samples and max_fit_samples > 0 and X_fit.shape[0] > max_fit_samples:
        idx = torch.linspace(
            0,
            X_fit.shape[0] - 1,
            steps=max_fit_samples,
            dtype=torch.long,
        )
        X_fit = X_fit[idx]

    X_fit = X_fit.to(device)

    with torch.no_grad():
        # 1. ALWAYS fit the state scaler FIRST so RBFs and Polynomials are stable
        if hasattr(expander, "fit_state_scaler"):
            expander.fit_state_scaler(X_fit)
            model._state_scaler_initialized = True

        # 2. Fit data-dependent expanders (RBF/SVD) using the scaled state
        if hasattr(expander, "fit") and not getattr(expander, "is_fitted", False):
            print(
                f"Fitting data-dependent expander on {X_fit.shape[0]} samples "
                f"for expansion_type={getattr(model, 'expansion_type', 'unknown')}..."
            )
            expander.fit(X_fit)

        # 3. Refresh public aliases after fitting
        if hasattr(expander, "expand_names"):
            model.expand_names = expander.expand_names
        if hasattr(expander, "state_indices"):
            model.state_indices = expander.state_indices
        if hasattr(expander, "expanded_dim"):
            model.expanded_dim = expander.expanded_dim
            model.latent_dim = expander.expanded_dim

EVAL_HORIZONS = [10, 20, 100]

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


def dataloader_to_numpy(loader):
    """Collect all (x, y) pairs from a DataLoader into NumPy arrays."""
    xs, ys = [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.vstack(xs), np.vstack(ys)

def main():
    parser = argparse.ArgumentParser(description="Fast W&B sweep training for Koopman models")

    parser.add_argument("--model", type=str, required=True, choices=["ml_linear_dynamics", "ml_lineardynamics", "ml_dmd", "regression_dmd", "mlp_baseline", "sindy_baseline", "linear_baseline", "dmd_baseline"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--name", type=str, default="run")

    parser.add_argument("--subset", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden layer width for mlp_baseline.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers for mlp_baseline.")

    parser.add_argument("--bias", type=str.lower, choices=["true", "false"], default="true")
    parser.add_argument("--expansion_type", type=str, choices=["general", "specific", "rbf", "delay", "hankel_svd"], default="general")
    parser.add_argument("--expansion_degree", type=int, default=3)
    parser.add_argument("--sine_cosine_expansion", type=str.lower, choices=["true", "false"], default="false")
    parser.add_argument("--normalize_state", type=str.lower, choices=["true", "false"], default="false")
    parser.add_argument("--normalize_lifted", type=str.lower, choices=["true", "false"], default="true")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--regression_rollout_mode", type=str, default="DMD", choices=["linear_dynamics", "DMD", "projected_DMD"], help="Default rollout mode for regression_dmd checkpoints.")
    parser.add_argument("--delay_depth", type=int, default=1, help="Number of stacked delay coordinates to use when expansion_type='delay'.")
    parser.add_argument("--hankel_rank", type=int, default=None, help="Number of SVD delay coordinates when expansion_type='hankel_svd'.")
    parser.add_argument("--expander_fit_samples", type=int, default=0, help="Max training samples used to fit data-dependent ML expanders. Use 0 for all available.")
    parser.add_argument("--rbf_n_centers", type=int, default=50, help="Number of RBF centers when expansion_type='rbf'.")
    parser.add_argument("--rbf_center_selection", type=str, default="farthest", choices=["random", "farthest"], help="How to choose RBF centers from training states.")
    parser.add_argument("--rbf_bandwidth_mode", type=str, default="knn", choices=["global", "knn"], help="How to choose RBF widths (sigmas).")
    parser.add_argument("--rbf_knn_k", type=int, default=5, help="k for k-nearest-center bandwidth when expansion_type='rbf'.")
    parser.add_argument("--load_rbf_from", type=str, default=None, help="Path to a model file to load fixed RBF centers from.")
    parser.add_argument("--l1_weight", type=float, default=1e-6, help="L1 regularization weight for regression DMD")
    parser.add_argument("--biorth_weight", type=float, default=0.1, help="Weight for bi-orthogonality regularization")

    parser.add_argument("--sindy_discrete_time", type=str.lower, choices=["true", "false"], default="false")
    parser.add_argument("--sindy_poly_order", type=int, default=3)
    parser.add_argument("--sindy_threshold", type=float, default=0.1)
    parser.add_argument("--sindy_alpha", type=float, default=0.0)
    parser.add_argument("--sindy_include_bias", type=str.lower, choices=["true", "false"], default="true")
    parser.add_argument("--sindy_include_interaction", type=str.lower, choices=["true", "false"], default="true")
    parser.add_argument(
        "--sindy_diff_method",
        type=str,
        default="finite_difference",
        choices=["finite_difference", "smoothed_finite_difference"],
    )
    parser.add_argument(
        "--sindy_library_type",
        type=str,
        default="polynomial",
        choices=["polynomial", "fourier", "poly_fourier", "specific"],
    )
    parser.add_argument("--sindy_fourier_n_frequencies", type=int, default=1)
    parser.add_argument(
        "--sindy_specific_basis_size",
        type=int,
        default=None,
        help="If using sindy_library_type='specific', use the first k basis terms for that system.",
    )

    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--max_val_rollout_trajs", type=int, default=None)
    parser.add_argument(
        "--eval_horizon_divisor",
        type=int,
        default=1,
        help="Divide EVAL_HORIZONS by this integer (integer division). Default 1 = no change.",
    )
    parser.add_argument("--rollout_horizon", type=int, default=None, help="Rollout horizon for training loss (only used in the loss function)")
    parser.add_argument(
        "--dataset_rollout_reserve",
        type=int,
        default=None,
        help="(Optional) Reserve this many future steps when constructing the dataset. If omitted the script will try to infer a sensible reserve from the train split (defaults to 100 or less).",
    )
    parser.add_argument("--log_phi_every", type=int, default=-1, help="Print get_Phi() every N epochs (default: auto-detect based on model type)")
    parser.add_argument("--phi_print_max_dim", type=int, default=12, help="When Phi is larger than this, print only the top-left block")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--print_every_batch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="data/models")

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
        tags=[
            system_name,
            args.model,
            args.sindy_library_type if args.model == "sindy_baseline" else args.expansion_type,
        ],
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
        },
        allow_val_change=True,
    )

    # Compute effective eval horizons based on optional divisor (default: no change)
    divisor = int(getattr(args, "eval_horizon_divisor", 1)) if getattr(args, "eval_horizon_divisor", None) is not None else 1
    if divisor < 1:
        print(f"Warning: --eval_horizon_divisor must be >=1, got {divisor}. Using 1.")
        divisor = 1
    if divisor == 1:
        eval_horizons = list(EVAL_HORIZONS)
    else:
        # Integer divide each horizon, clamp to at least 1, and preserve order/uniqueness
        seen = set()
        eval_horizons = []
        for h in EVAL_HORIZONS:
            h2 = max(1, int(h) // divisor)
            if h2 not in seen:
                seen.add(h2)
                eval_horizons.append(h2)
        print(f"Using eval_horizons={eval_horizons} (EVAL_HORIZONS={EVAL_HORIZONS} divided by {divisor})")

    # Setup output directory (for checkpoint saving)
    save_dir = os.path.join(args.outdir, args.model, system_name, run.id)
    os.makedirs(save_dir, exist_ok=True)

    # Data
    is_ml_model = args.model in {"ml_linear_dynamics", "ml_lineardynamics", "ml_dmd"}

    if args.dataset_rollout_reserve is not None:
        dataset_rollout_horizon = args.dataset_rollout_reserve
    else:
        try:
            with np.load(train_meta_path) as d:
                X = d["X"]
                T = X.shape[0]
            dataset_rollout_horizon = min(100, max(0, T - 2))
        except Exception:
            dataset_rollout_horizon = 20 if is_ml_model else 0

    # Validate delay settings: using delay_depth>1 requires a delay-based expansion
    if args.delay_depth > 1 and args.expansion_type not in {"delay", "hankel_svd"}:
        raise ValueError(
            "delay_depth > 1 requires --expansion_type delay or --expansion_type hankel_svd."
        )

    train_ds = OneStepTrajectoryDataset(args.data_path, split="train", subset=args.subset, rollout_horizon=dataset_rollout_horizon, delay_depth=getattr(args, "delay_depth", 1))
    val_ds = OneStepTrajectoryDataset(args.data_path, split="val", subset=args.subset, rollout_horizon=dataset_rollout_horizon, delay_depth=getattr(args, "delay_depth", 1))

    pin_memory = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory) if len(val_ds) > 0 else None

    # Track best metrics for the final W&B summary.
    best_metrics = {}

    # Handle simple non-ML baselines that are fit-once and use NumPy rollouts
    if args.model == "linear_baseline":
        print("Fitting linear baseline...")
        X_train, Y_train = dataloader_to_numpy(train_loader)
        M = fit_linear_map(X_train, Y_train)

        class LinearWrap:
            def __init__(self, M):
                self.M = M
                self.state_dim = M.shape[0]

            def rollout(self, x0, steps):
                return rollout_linear_map(self.M, x0, steps)

        model = LinearWrap(M)

        train_loss, train_rmse = compute_loader_metrics(model, train_loader, device)
        val_loss, val_rmse = compute_loader_metrics(model, val_loader, device) if val_loader is not None else (None, None)

        metrics = {
            "train_loss": train_loss,
            "train_onestep_rmse": train_rmse,
            "val_loss": val_loss,
            "val_onestep_rmse": val_rmse,
        }

        rollout_metrics = compute_rollout_metrics(
            model=model,
            X=val_X,
            device=device,
            eval_horizons=eval_horizons,
            max_trajs=args.max_val_rollout_trajs,
        )
        if rollout_metrics is not None:
            for k, v in rollout_metrics.items():
                metrics[f"val_{k}"] = v

        wandb.log(metrics, step=0)
        update_best_metrics(best_metrics, metrics, 0)

        save_kwargs = dict(
            train_args=vars(args),
            model="linear_baseline",
            system=system_name,
            M=M,
        )

        save_path = os.path.join(save_dir, "model.npz")
        np.savez(save_path, **save_kwargs)
        print(f"Saved linear baseline checkpoint to: {save_path}")

        for metric_name, data in best_metrics.items():
            wandb.summary[f"best_{metric_name}"] = data["value"]
            wandb.summary[f"best_{metric_name}_epoch"] = data["epoch"]

        wandb.finish()
        return

    if args.model == "dmd_baseline":
        print("Fitting DMD baseline...")
        X_train, Y_train = dataloader_to_numpy(train_loader)
        Lambda, Phi = fit_dmd(X_train, Y_train, rank=getattr(args, "rank", None), ridge=getattr(args, "ridge", 0.0))

        class DMDWrap:
            def __init__(self, Lambda, Phi):
                self.Lambda = Lambda
                self.Phi = Phi
                self.state_dim = Phi.shape[0]

            def rollout(self, x0, steps):
                return rollout_dmd_eig(self.Lambda, self.Phi, x0, steps)

        model = DMDWrap(Lambda, Phi)

        train_loss, train_rmse = compute_loader_metrics(model, train_loader, device)
        val_loss, val_rmse = compute_loader_metrics(model, val_loader, device) if val_loader is not None else (None, None)

        metrics = {
            "train_loss": train_loss,
            "train_onestep_rmse": train_rmse,
            "val_loss": val_loss,
            "val_onestep_rmse": val_rmse,
        }

        rollout_metrics = compute_rollout_metrics(
            model=model,
            X=val_X,
            device=device,
            eval_horizons=eval_horizons,
            max_trajs=args.max_val_rollout_trajs,
        )
        if rollout_metrics is not None:
            for k, v in rollout_metrics.items():
                metrics[f"val_{k}"] = v

        wandb.log(metrics, step=0)
        update_best_metrics(best_metrics, metrics, 0)

        save_kwargs = dict(
            train_args=vars(args),
            model="dmd_baseline",
            system=system_name,
            Lambda=Lambda,
            Phi=Phi,
        )

        save_path = os.path.join(save_dir, "model.npz")
        np.savez(save_path, **save_kwargs)
        print(f"Saved DMD baseline checkpoint to: {save_path}")

        for metric_name, data in best_metrics.items():
            wandb.summary[f"best_{metric_name}"] = data["value"]
            wandb.summary[f"best_{metric_name}_epoch"] = data["epoch"]

        wandb.finish()
        return

    # Model
    model = build_model(args, state_dim, system_name, device)

    # Ensure models without buffers (e.g. Regression_DMD) expose a device-valued
    # buffer so Regression_DMD._to_tensor can detect the correct device at runtime.
    # This avoids moving inputs to CPU incorrectly when calling model(x).
    try:
        if isinstance(model, torch.nn.Module) and len(list(model.buffers())) == 0:
            model.register_buffer("_device_marker", torch.zeros(1, device=device))
    except Exception:
        # Best-effort; if registration fails, continue — training may still work.
        pass

    if args.model == "sindy_baseline":
        print("Fitting SINDy baseline...")

        sindy_discrete_time = args.sindy_discrete_time == "true"

        if sindy_discrete_time:
            X_train, Y_train = dataloader_to_numpy(train_loader)
            model.fit_discrete_pairs(X_train, Y_train)
        else:
            train_split_path = resolve_split_npz_path(args.data_path, "train")
            meta_data = np.load(train_split_path)
            X_traj = meta_data["X"]
            dt = float(meta_data["dt"])
            if X_traj.ndim != 3:
                raise ValueError("Continuous SINDy expects X with shape (T, n_traj, d)")
            model.fit_continuous_trajectories(X_traj, dt=dt)

        train_loss, train_rmse = compute_loader_metrics(model, train_loader, device)
        val_loss, val_rmse = compute_loader_metrics(model, val_loader, device) if val_loader is not None else (None, None)

        metrics = {
            "train_loss": train_loss,
            "train_onestep_rmse": train_rmse,
            "val_loss": val_loss,
            "val_onestep_rmse": val_rmse,
        }

        rollout_metrics = compute_rollout_metrics(
            model=model,
            X=val_X,
            device=device,
            eval_horizons=eval_horizons,
            max_trajs=args.max_val_rollout_trajs,
        )
        if rollout_metrics is not None:
            for k, v in rollout_metrics.items():
                metrics[f"val_{k}"] = v

        wandb.log(metrics, step=0)
        update_best_metrics(best_metrics, metrics, 0)

        save_kwargs = dict(
            train_args=vars(args),
            model="sindy_baseline",
            system=system_name,
            data_path=args.data_path,
            discrete_time=sindy_discrete_time,
            poly_order=args.sindy_poly_order,
            threshold=args.sindy_threshold,
            alpha=args.sindy_alpha,
            include_bias=args.sindy_include_bias == "true",
            include_interaction=args.sindy_include_interaction == "true",
            diff_method=args.sindy_diff_method,
            library_type=args.sindy_library_type,
            fourier_n_frequencies=args.sindy_fourier_n_frequencies,
            specific_system=system_name if args.sindy_library_type == "specific" else "",
            specific_basis_size=-1 if args.sindy_specific_basis_size is None else args.sindy_specific_basis_size,
            coefficients=model.get_coefficients(),
            equations=np.array(model.equations(), dtype=object),
        )

        save_path = os.path.join(save_dir, "model.npz")
        np.savez(save_path, **save_kwargs)

        with open(os.path.join(save_dir, "equations.txt"), "w", encoding="utf-8") as handle:
            for equation in model.equations():
                handle.write(equation + "\n")

        print(f"Saved SINDy checkpoint to: {save_path}")

        for metric_name, data in best_metrics.items():
            wandb.summary[f"best_{metric_name}"] = data["value"]
            wandb.summary[f"best_{metric_name}_epoch"] = data["epoch"]

        wandb.finish()
        return

    # Regression DMD is fit once, then evaluated and saved using the same
    # data flow as scripts/train.py.
    if args.model == "regression_dmd":
        if hasattr(args, "load_rbf_from") and args.load_rbf_from:
            print(f"Loading fixed RBF centers and sigmas from: {args.load_rbf_from}")
            try:
                loaded_data = np.load(args.load_rbf_from)
                c_tensor = torch.as_tensor(loaded_data["rbf_centers"], dtype=torch.float32, device=device)
                s_tensor = torch.as_tensor(loaded_data["rbf_sigmas"], dtype=torch.float32, device=device)

                model.expander.centers.copy_(c_tensor)
                model.expander.sigmas.copy_(s_tensor)
                model.expander.is_fitted = True
                model.expander.freeze_centers = True
            except Exception as e:
                print(f"Warning: Failed to load RBF from {args.load_rbf_from}: {e}")

        X_train, Y_train = dataloader_to_numpy(train_loader)
        print("Fitting regression_dmd...")
        model.fit(X_train, Y_train)

        # ---------------------------------------------------------
        # 1. COMPUTE & LOG METRICS FIRST! 
        # (Guarantees W&B gets the data)
        # ---------------------------------------------------------
        train_loss, train_rmse = compute_loader_metrics(model, train_loader, device)
        val_loss, val_rmse = compute_loader_metrics(model, val_loader, device) if val_loader is not None else (None, None)

        metrics = {
            "train_loss": train_loss,
            "train_onestep_rmse": train_rmse,
            "val_loss": val_loss,
            "val_onestep_rmse": val_rmse,
        }

        print("Computing rollout metrics...")
        rollout_metrics = compute_rollout_metrics(
            model=model,
            X=val_X,
            device=device,
            eval_horizons=eval_horizons,
            max_trajs=args.max_val_rollout_trajs,
        )
        if rollout_metrics is not None:
            for k, v in rollout_metrics.items():
                metrics[f"val_{k}"] = v

        wandb.log(metrics, step=0)
        update_best_metrics(best_metrics, metrics, 0)

        for metric_name, data in best_metrics.items():
            wandb.summary[f"best_{metric_name}"] = data["value"]
            wandb.summary[f"best_{metric_name}_epoch"] = data["epoch"]

        # ---------------------------------------------------------
        # 2. UNIFIED PYTORCH CHECKPOINT SAVING
        # ---------------------------------------------------------
        def _to_np(tensor):
            return tensor.detach().cpu().numpy() if tensor is not None else None

        dmd_matrices = {
            "K_full": _to_np(getattr(model, "K_fitted", None)),
            "C": _to_np(getattr(model, "C_fitted", None)),
            "K_tilde": _to_np(getattr(model, "K_tilde_fitted", None)),
            "U_r": _to_np(getattr(model, "U_r_fitted", None)),
            "W_reduced": _to_np(getattr(model, "W_reduced_fitted", None)),
            "Lambda": _to_np(getattr(model, "Lambda_fitted", None)),
            "Phi_lift": _to_np(getattr(model, "Phi_lift_fitted", None)),
            "Phi_state": _to_np(getattr(model, "Phi_state_fitted", None)),
            "Phi": _to_np(getattr(model, "Phi_fitted", None)),
            "x_mean": _to_np(getattr(model, "x_mean", None)),
            "x_scale": _to_np(getattr(model, "x_scale", None)),
            "psi_scale": _to_np(getattr(model, "psi_scale", None)),
        }

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "checkpoint_type": "best",
            "train_args": vars(args),
            "dmd_matrices": dmd_matrices,
            "model_name": "regression_dmd",
            "system": system_name,
            "state_dim": state_dim,
        }

        save_path = os.path.join(save_dir, "model.pt")
        torch.save(checkpoint, save_path)
        print(f"Saved unified Regression_DMD checkpoint to: {save_path}")

        wandb.finish()
        return

    # Fit RBF/Hankel/other data-dependent expanders if needed
    if args.model in {"ml_linear_dynamics", "ml_lineardynamics", "ml_dmd", "regression_dmd"}:
        prepare_ml_expander_and_lift_stats(
            model=model,
            train_ds=train_ds,
            device=device,
            max_fit_samples=args.expander_fit_samples if hasattr(args, "expander_fit_samples") else 0,
        )

    # Load fixed RBF centers if provided
    if hasattr(args, "load_rbf_from") and args.load_rbf_from:
        print(f"Loading fixed RBF centers and sigmas from: {args.load_rbf_from}")
        try:
            loaded_data = np.load(args.load_rbf_from)
            c_tensor = torch.as_tensor(loaded_data["rbf_centers"], dtype=torch.float32, device=device)
            s_tensor = torch.as_tensor(loaded_data["rbf_sigmas"], dtype=torch.float32, device=device)

            model.expander.centers.copy_(c_tensor)
            model.expander.sigmas.copy_(s_tensor)
            model.expander.is_fitted = True
            model.expander.freeze_centers = True
        except Exception as e:
            print(f"Warning: Failed to load RBF from {args.load_rbf_from}: {e}")

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

        # Multi-step Rollout Metrics
        rollout_metrics = compute_rollout_metrics(
            model=model,
            X=val_X,
            device=device,
            eval_horizons=eval_horizons,
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
    
    # Compute log_phi_every: use auto-detection if not explicitly set
    if args.log_phi_every >= 0:
        log_phi_every = args.log_phi_every
    else:
        log_phi_every = 1 if is_ml_model else 0
    
    training_rollout_horizon = args.rollout_horizon if args.rollout_horizon is not None else (20 if is_ml_model else 0)

    model, (train_losses, epoch_val_losses, loss_components_val), best_checkpoint = train_onestep_sweep(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_phi_every=log_phi_every,
        phi_print_max_dim=args.phi_print_max_dim,
        eval_callback=run_eval_callback,
        rollout_horizon=training_rollout_horizon,
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