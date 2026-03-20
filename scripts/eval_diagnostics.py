
import argparse
import os
from typing import Dict, List, Tuple

from matplotlib.ticker import FixedLocator, FuncFormatter
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.eval.metrics import (
    compute_composite_validation_score,
    compute_full_rollout_metrics,
    compute_horizon_metrics,
    compute_one_step_metrics,
    get_state_scale_from_train_split,
    save_summary_npz,
)
from src.data_generation.load_data import resolve_split_npz_path
from src.eval.model_io import (
    infer_run_name,
    load_model,
    predict_rollout_from_x0,
)
"""
Validation-side diagnostic evaluation for trained dynamical-system models.

This script is used for model selection and debugging on `val_idx`, not for final test reporting.

It computes:
- one-step MSE / RMSE / NRMSE
- terminal horizon MSE / RMSE / NRMSE
- full-rollout MSE / RMSE / NRMSE
- composite validation score

It saves:
- diagnostics_summary.npz

It also produces diagnostic plots:
- error vs horizon
- phase-space error maps
- rollout error summary
- initial-condition error map

Typical workflow:
train -> eval_diagnostics on validation -> compare models -> eval on test

Examples:
python -m scripts.eval_diagnostics --model linear_baseline --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/linear_baseline/saddle_point/default/model.npz --horizons 1,2,5,10 --rollout_horizons 5,10 --phase_horizons 1,5 --heatmap_horizon 5
python -m scripts.eval_diagnostics --model dmd_baseline --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/dmd_baseline/saddle_point/default/model.npz --horizons 1,2,5,10 --rollout_horizons 5,10 --phase_horizons 1,5 --heatmap_horizon 5
python -m scripts.eval_diagnostics --model ml_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/ml_dmd/saddle_point/default/model.pt --horizons 1,2,5,10 --rollout_horizons 5,10 --phase_horizons 1,5 --heatmap_horizon 5
python -m scripts.eval_diagnostics --model ml_eigen_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/ml_eigen_dmd/saddle_point/default/model.pt --horizons 1,2,5,10 --rollout_horizons 5,10 --phase_horizons 1,5 --heatmap_horizon 5
python -m scripts.eval_diagnostics --model regression_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/regression_dmd/saddle_point/default/model.npz --horizons 1,2,5,10 --rollout_horizons 5,10 --phase_horizons 1,5 --heatmap_horizon 5
python -m scripts.eval_diagnostics --model ml_lineardynamics --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/ml_lineardynamics/saddle_point/default/model.pt --horizons 1,2,5,10 --rollout_horizons 5,10 --phase_horizons 1,5 --heatmap_horizon 5
python -m scripts.eval_diagnostics --model ml_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/ml_dmd/saddle_point/default/model.pt --horizons 1,2,5,10 --rollout_horizons 5,10 --phase_horizons 1,5 --heatmap_horizon 5
python -m scripts.eval_diagnostics --model sindy_baseline --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/sindy_baseline/saddle_point/default/model.npz --horizons 1,2,5,10 --rollout_horizons 5,10 --phase_horizons 1,5 --heatmap_horizon 5
"""

def parse_int_list(text: str) -> List[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("At least one horizon must be provided.")
    return sorted(set(values))


def get_phase_dims(system: str, state_dim: int) -> Tuple[int, int]:
    if system == "lorenz" and state_dim >= 3:
        return 0, 2
    if state_dim < 2:
        raise ValueError("Phase-space plots require state_dim >= 2.")
    return 0, 1


def compute_phase_error_for_trajectory(
    X: np.ndarray,
    traj_id: int,
    horizons: List[int],
    model_name: str,
    model,
    extras: Dict[str, np.ndarray],
) -> Dict[int, Dict[str, np.ndarray]]:
    X_traj = X[:, traj_id, :]
    T = X_traj.shape[0]
    max_h = max(horizons)

    starts_per_h = {h: [] for h in horizons}
    errors_per_h = {h: [] for h in horizons}

    for t0 in range(T - max_h):
        x0 = X_traj[t0]
        rollout = predict_rollout_from_x0(
            x0=x0,
            steps=max_h,
            model_name=model_name,
            model=model,
            extras=extras,
        )

        for h in horizons:
            starts_per_h[h].append(X_traj[t0].copy())
            errors_per_h[h].append(np.mean((rollout[h] - X_traj[t0 + h]) ** 2))

    result = {}
    for h in horizons:
        result[h] = {
            "starts": np.asarray(starts_per_h[h]),
            "errors": np.asarray(errors_per_h[h]),
        }
    return result


def compute_initial_condition_heatmap_data(
    X: np.ndarray,
    val_idx: np.ndarray,
    horizon: int,
    model_name: str,
    model,
    extras: Dict[str, np.ndarray],
    mode: str = "traj_initials",
) -> Dict[str, np.ndarray]:
    starts = []
    errors = []

    for traj_id in val_idx:
        X_traj = X[:, traj_id, :]

        if mode == "traj_initials":
            start_indices = [0]
        elif mode == "all_valid_starts":
            start_indices = range(X_traj.shape[0] - horizon)
        else:
            raise ValueError(f"Unknown heatmap mode: {mode}")

        for t0 in start_indices:
            x0 = X_traj[t0]
            rollout = predict_rollout_from_x0(
                x0=x0,
                steps=horizon,
                model_name=model_name,
                model=model,
                extras=extras,
            )
            starts.append(X_traj[t0].copy())
            errors.append(np.mean((rollout[horizon] - X_traj[t0 + horizon]) ** 2))

    return {
        "starts": np.asarray(starts),
        "errors": np.asarray(errors),
    }


def plot_error_vs_horizon(metrics: Dict[str, np.ndarray], figdir: str, logy: bool = True) -> None:
    horizons = metrics["horizons"]
    mse = metrics["horizon_mse"]
    rmse = metrics["horizon_rmse"]
    nrmse = metrics["horizon_nrmse"]

    plt.figure(figsize=(7, 5))
    plt.plot(horizons, mse, label="MSE")
    plt.plot(horizons, rmse, label="RMSE")
    plt.plot(horizons, nrmse, label="NRMSE")
    plt.xlabel("Prediction horizon")
    plt.ylabel("Metric value")
    plt.title("Error vs prediction horizon")
    if logy:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "error_vs_horizon.png"), dpi=200)
    plt.close()


def plot_phase_space_colored_errors(
    phase_data: Dict[int, Dict[str, np.ndarray]],
    system: str,
    figdir: str,
) -> None:
    horizons = sorted(phase_data.keys())
    n = len(horizons)

    sample_starts = phase_data[horizons[0]]["starts"]
    state_dim = sample_starts.shape[1]
    i, j = get_phase_dims(system, state_dim)

    if n == 1:
        h = horizons[0]
        starts = phase_data[h]["starts"]
        errors = phase_data[h]["errors"]

        positive_errors = errors[errors > 0]
        if len(positive_errors) == 0:
            vmin, vmax = 1e-16, 1.0
        else:
            vmin = positive_errors.min()
            vmax = errors.max()
            if vmax <= vmin:
                vmax = vmin * 10.0

        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(
            starts[:, i],
            starts[:, j],
            c=np.maximum(errors, vmin),
            s=14,
            norm=norm,
        )
        ax.set_xlabel(f"x{i + 1}")
        ax.set_ylabel(f"x{j + 1}")
        ax.set_title(f"Phase-space error map (h={h})")
        tick_mid = np.sqrt(vmin * vmax)
        ticks = [vmin, tick_mid, vmax]

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Terminal h-step MSE")
        cbar.locator = FixedLocator(ticks)
        cbar.formatter = FuncFormatter(lambda x, pos: f"{x:.1e}")
        cbar.update_ticks()
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "phase_space_error_maps.png"), dpi=200)
        plt.close(fig)
        return

    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    all_errors = np.concatenate([phase_data[h]["errors"] for h in horizons])
    positive_errors = all_errors[all_errors > 0]
    if len(positive_errors) == 0:
        vmin, vmax = 1e-16, 1.0
    else:
        vmin = positive_errors.min()
        vmax = all_errors.max()
        if vmax <= vmin:
            vmax = vmin * 10.0

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    fig.subplots_adjust(right=0.88, wspace=0.25, hspace=0.30)

    scatter_for_colorbar = None

    for ax, h in zip(axes.flatten(), horizons):
        starts = phase_data[h]["starts"]
        errors = phase_data[h]["errors"]

        sc = ax.scatter(
            starts[:, i],
            starts[:, j],
            c=np.maximum(errors, vmin),
            s=12,
            norm=norm,
        )
        if scatter_for_colorbar is None:
            scatter_for_colorbar = sc

        ax.set_xlabel(f"x{i + 1}")
        ax.set_ylabel(f"x{j + 1}")
        ax.set_title(f"Phase-space error map (h={h})")

    for ax in axes.flatten()[n:]:
        ax.axis("off")

    cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    tick_mid = np.sqrt(vmin * vmax)
    ticks = [vmin, tick_mid, vmax]

    cbar = fig.colorbar(scatter_for_colorbar, cax=cax)
    cbar.set_label("Terminal h-step MSE")
    cbar.locator = FixedLocator(ticks)
    cbar.formatter = FuncFormatter(lambda x, pos: f"{x:.1e}")
    cbar.update_ticks()

    fig.savefig(os.path.join(figdir, "phase_space_error_maps.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_error_distribution(
    rollout_metrics: Dict[str, np.ndarray],
    figdir: str,
) -> None:
    horizons = rollout_metrics["rollout_horizons"]

    plt.figure(figsize=(8, 5))
    plt.plot(horizons, rollout_metrics["rollout_mse"], marker="o", label="MSE")
    plt.plot(horizons, rollout_metrics["rollout_rmse"], marker="o", label="RMSE")
    plt.plot(horizons, rollout_metrics["rollout_nrmse"], marker="o", label="NRMSE")
    plt.xlabel("Prediction horizon")
    plt.ylabel("Metric value")
    plt.title("Full-rollout error across validation trajectories")
    plt.yscale("log")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "rollout_error_distribution.png"), dpi=200)
    plt.close()


def plot_initial_condition_heatmap(
    heatmap_data: Dict[str, np.ndarray],
    system: str,
    figdir: str,
    horizon: int,
    mode: str,
) -> None:
    starts = heatmap_data["starts"]
    errors = heatmap_data["errors"]

    state_dim = starts.shape[1]
    i, j = get_phase_dims(system, state_dim)

    positive_errors = errors[errors > 0]
    if len(positive_errors) == 0:
        vmin, vmax = 1e-16, 1.0
    else:
        vmin = max(np.percentile(positive_errors, 1), 1e-16)
        vmax = np.percentile(errors, 99)
        if vmax <= vmin:
            vmax = positive_errors.max()
        if vmax <= vmin:
            vmax = vmin * 10.0

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        starts[:, i],
        starts[:, j],
        c=np.clip(errors, vmin, vmax),
        s=18,
        norm=norm,
    )
    plt.xlabel(f"x{i + 1}")
    plt.ylabel(f"x{j + 1}")
    plt.title(f"Initial-condition error map (h={horizon}, mode={mode})")
    tick_mid = np.sqrt(vmin * vmax)
    ticks = [vmin, tick_mid, vmax]

    cbar = plt.colorbar(sc, label="Terminal h-step MSE")
    cbar.locator = FixedLocator(ticks)
    cbar.formatter = FuncFormatter(lambda x, pos: f"{x:.1e}")
    cbar.update_ticks()
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, f"initial_condition_error_map_h{horizon}_{mode}.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Diagnostic evaluation plots for dynamical-system prediction models.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "linear_baseline",
            "dmd_baseline",
            # "ml_dmd",
            # "ml_eigen_dmd",
            "regression_dmd",
            "ml_lineardynamics",
            "ml_dmd",
            "sindy_baseline",
        ],
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--name", type=str, help="Optional suffix for saved figure folder")
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,2,5,10,20,50,100",
        help="Comma-separated terminal horizons.",
    )
    parser.add_argument(
        "--rollout_horizons",
        type=str,
        default="5,10,20,50,100",
        help="Comma-separated rollout horizons from x(0).",
    )
    parser.add_argument(
        "--phase_horizons",
        type=str,
        default="1,10,50",
        help="Comma-separated horizons for phase-space colored error maps.",
    )
    parser.add_argument(
        "--heatmap_horizon",
        type=int,
        default=50,
        help="Horizon for initial-condition difficulty map.",
    )
    parser.add_argument(
        "--heatmap_mode",
        type=str,
        default="traj_initials",
        choices=["traj_initials", "all_valid_starts"],
        help="Use only validation trajectory initial conditions or every valid starting point.",
    )
    parser.add_argument(
        "--traj_index",
        type=int,
        default=0,
        help="Which validation trajectory to use for phase-space colored error maps.",
    )
    parser.add_argument(
        "--linear_error_scale",
        action="store_true",
        help="Use linear instead of log scale on the error-vs-horizon plot.",
    )
    parser.add_argument(
        "--max_one_step_pairs_per_traj",
        type=int,
        default=None,
        help="Optional cap on one-step pairs per validation trajectory.",
    )
    parser.add_argument(
        "--max_horizon_starts_per_traj",
        type=int,
        default=None,
        help="Optional cap on number of start points per validation trajectory for horizon metrics.",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_data_path = resolve_split_npz_path(args.data_path, "val")

    data = np.load(val_data_path)
    X = data["X"]
    T, _, state_dim = X.shape

    if X.ndim != 3:
        raise ValueError("Diagnostics expect multiple trajectories (X must be 3D).")

    if X.shape[1] == 0:
        raise ValueError("No validation trajectories available.")

    val_idx = np.arange(X.shape[1])

    system = str(data["system"])
    run_name = infer_run_name(args.model_path, args.name)

    figdir = os.path.join("data", "figures", args.model, system, run_name, "diagnostics")
    os.makedirs(figdir, exist_ok=True)

    print(f"Loaded X with shape {X.shape}.")
    print(f"Using {len(val_idx)} validation trajectories.")
    print(f"Saving diagnostic figures to: {figdir}")

    model, extras = load_model(
        model_name=args.model,
        model_path=args.model_path,
        data_path=args.data_path,
        state_dim=state_dim,
        system=system,
        device=device,
    )

    scales = get_state_scale_from_train_split(args.data_path)
    scale_std = scales["std"]

    horizons = parse_int_list(args.horizons)
    rollout_horizons = parse_int_list(args.rollout_horizons)
    phase_horizons = parse_int_list(args.phase_horizons)

    max_needed = max(max(horizons), max(rollout_horizons), max(phase_horizons), args.heatmap_horizon)
    if T <= max_needed:
        raise ValueError(
            f"Trajectory length T={T} is too short for requested max horizon {max_needed}. Use smaller horizons."
        )

    if args.traj_index >= len(val_idx):
        raise IndexError(f"traj_index={args.traj_index} but only {len(val_idx)} validation trajectories exist.")
    traj_id = val_idx[args.traj_index]

    print("Computing one-step metrics...")
    one_step_metrics = compute_one_step_metrics(
        X=X,
        traj_indices=val_idx,
        model_name=args.model,
        model=model,
        extras=extras,
        scale_std=scale_std,
        max_pairs_per_traj=args.max_one_step_pairs_per_traj,
    )

    print("Computing horizon metrics...")
    horizon_metrics = compute_horizon_metrics(
        X=X,
        traj_indices=val_idx,
        horizons=horizons,
        model_name=args.model,
        model=model,
        extras=extras,
        scale_std=scale_std,
        max_starts_per_traj=args.max_horizon_starts_per_traj,
    )

    print("Computing full-rollout metrics...")
    rollout_metrics = compute_full_rollout_metrics(
        X=X,
        traj_indices=val_idx,
        rollout_horizons=rollout_horizons,
        model_name=args.model,
        model=model,
        extras=extras,
        scale_std=scale_std,
    )

    composite_score = compute_composite_validation_score(
        one_step_nrmse=float(one_step_metrics["one_step_nrmse"]),
        horizon_nrmse=horizon_metrics["horizon_nrmse"],
        rollout_nrmse=rollout_metrics["rollout_nrmse"],
    )

    print("Computing phase-space colored errors...")
    phase_data = compute_phase_error_for_trajectory(
        X=X,
        traj_id=traj_id,
        horizons=phase_horizons,
        model_name=args.model,
        model=model,
        extras=extras,
    )

    print("Computing initial-condition difficulty map...")
    heatmap_data = compute_initial_condition_heatmap_data(
        X=X,
        val_idx=val_idx,
        horizon=args.heatmap_horizon,
        model_name=args.model,
        model=model,
        extras=extras,
        mode=args.heatmap_mode,
    )

    print("Saving plots...")
    plot_error_vs_horizon(horizon_metrics, figdir, logy=not args.linear_error_scale)
    plot_phase_space_colored_errors(phase_data, system, figdir)
    plot_rollout_error_distribution(rollout_metrics, figdir)
    plot_initial_condition_heatmap(heatmap_data, system, figdir, args.heatmap_horizon, args.heatmap_mode)

    summary_path = os.path.join(figdir, "diagnostics_summary.npz")
    summary_payload = {
        "model_name": np.array(args.model),
        "system": np.array(system),
        "run_name": np.array(run_name),
        "split": np.array("validation"),
        "val_idx": np.asarray(val_idx),
        "scale_std": scale_std,
        "composite_validation_score": np.array(composite_score),
        **one_step_metrics,
        **horizon_metrics,
        **rollout_metrics,
        "phase_traj_id": np.array(traj_id),
        "heatmap_horizon": np.array(args.heatmap_horizon),
        "heatmap_mode": np.array(args.heatmap_mode),
    }

    save_summary_npz(summary_path, summary_payload)

    print("\nDone.")
    print(f"One-step MSE   : {float(one_step_metrics['one_step_mse']):.6e}")
    print(f"One-step RMSE  : {float(one_step_metrics['one_step_rmse']):.6e}")
    print(f"One-step NRMSE : {float(one_step_metrics['one_step_nrmse']):.6e}")
    print(f"Composite val score (lower is better): {composite_score:.6e}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()