import os
from typing import Dict, List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

from src.eval.model_io import predict_rollout_from_x0


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
    rollout_cache: Dict[int, Dict[str, np.ndarray]] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    X_traj = X[:, traj_id, :]
    max_h = max(horizons)

    starts_per_h = {h: [] for h in horizons}
    errors_per_h = {h: [] for h in horizons}

    if rollout_cache is not None and traj_id in rollout_cache:
        starts = rollout_cache[traj_id]["starts"]
        rollouts = rollout_cache[traj_id]["rollouts"]

        for t0, rollout in zip(starts, rollouts):
            for h in horizons:
                if h > rollout.shape[0] - 1 or t0 + h >= X_traj.shape[0]:
                    continue
                starts_per_h[h].append(X_traj[t0].copy())
                errors_per_h[h].append(np.mean((rollout[h] - X_traj[t0 + h]) ** 2))
    else:
        T = X_traj.shape[0]
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
    split_idx: np.ndarray,
    horizon: int,
    model_name: str,
    model,
    extras: Dict[str, np.ndarray],
    mode: str = "traj_initials",
    rollout_cache: Dict[int, Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    starts = []
    errors = []

    for traj_id in split_idx:
        X_traj = X[:, traj_id, :]

        if rollout_cache is not None and traj_id in rollout_cache:
            cache_starts = rollout_cache[traj_id]["starts"]
            cache_rollouts = rollout_cache[traj_id]["rollouts"]

            if mode == "traj_initials":
                selected = [(t0, rollout) for t0, rollout in zip(cache_starts, cache_rollouts) if t0 == 0]
            elif mode == "all_valid_starts":
                selected = list(zip(cache_starts, cache_rollouts))
            else:
                raise ValueError(f"Unknown heatmap mode: {mode}")

            for t0, rollout in selected:
                if horizon > rollout.shape[0] - 1 or t0 + horizon >= X_traj.shape[0]:
                    continue
                starts.append(X_traj[t0].copy())
                errors.append(np.mean((rollout[horizon] - X_traj[t0 + horizon]) ** 2))
        else:
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


def _format_three_tick_colorbar(cbar, vmin: float, vmax: float):
    tick_mid = np.sqrt(vmin * vmax)
    ticks = [vmin, tick_mid, vmax]
    cbar.locator = FixedLocator(ticks)
    cbar.formatter = FuncFormatter(lambda x, pos: f"{x:.1e}")
    cbar.update_ticks()


def plot_error_vs_horizon(horizon_metrics: Dict[str, np.ndarray], figdir: str, logy: bool = True) -> None:
    horizons = horizon_metrics["horizons"]
    mse = horizon_metrics["horizon_mse"]
    rmse = horizon_metrics["horizon_rmse"]
    nrmse = horizon_metrics["horizon_nrmse"]

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
        sc = ax.scatter(starts[:, i], starts[:, j], c=np.maximum(errors, vmin), s=14, norm=norm)
        ax.set_xlabel(f"x{i + 1}")
        ax.set_ylabel(f"x{j + 1}")
        ax.set_title(f"Phase-space error map (h={h})")

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Terminal h-step MSE")
        _format_three_tick_colorbar(cbar, vmin, vmax)

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

        sc = ax.scatter(starts[:, i], starts[:, j], c=np.maximum(errors, vmin), s=12, norm=norm)
        if scatter_for_colorbar is None:
            scatter_for_colorbar = sc

        ax.set_xlabel(f"x{i + 1}")
        ax.set_ylabel(f"x{j + 1}")
        ax.set_title(f"Phase-space error map (h={h})")

    for ax in axes.flatten()[n:]:
        ax.axis("off")

    cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(scatter_for_colorbar, cax=cax)
    cbar.set_label("Terminal h-step MSE")
    _format_three_tick_colorbar(cbar, vmin, vmax)

    fig.savefig(os.path.join(figdir, "phase_space_error_maps.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_error_summary(
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
    plt.title("Full-rollout error across trajectories")
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
    sc = plt.scatter(starts[:, i], starts[:, j], c=np.clip(errors, vmin, vmax), s=18, norm=norm)
    plt.xlabel(f"x{i + 1}")
    plt.ylabel(f"x{j + 1}")
    plt.title(f"Initial-condition error map (h={horizon}, mode={mode})")

    cbar = plt.colorbar(sc, label="Terminal h-step MSE")
    _format_three_tick_colorbar(cbar, vmin, vmax)

    plt.tight_layout()
    plt.savefig(os.path.join(figdir, f"initial_condition_error_map_h{horizon}_{mode}.png"), dpi=200)
    plt.close()


def run_diagnostics(
    *,
    X: np.ndarray,
    split_idx: np.ndarray,
    traj_id: int,
    model_name: str,
    model,
    extras: Dict[str, np.ndarray],
    system: str,
    figdir: str,
    horizon_metrics: Dict[str, np.ndarray],
    rollout_metrics: Dict[str, np.ndarray],
    phase_horizons: List[int],
    heatmap_horizon: int,
    heatmap_mode: str,
    linear_error_scale: bool = False,
    rollout_cache: Dict[int, Dict[str, np.ndarray]] = None,
) -> None:
    phase_data = compute_phase_error_for_trajectory(
        X=X,
        traj_id=traj_id,
        horizons=phase_horizons,
        model_name=model_name,
        model=model,
        extras=extras,
        rollout_cache=rollout_cache,
    )

    heatmap_data = compute_initial_condition_heatmap_data(
        X=X,
        split_idx=split_idx,
        horizon=heatmap_horizon,
        model_name=model_name,
        model=model,
        extras=extras,
        mode=heatmap_mode,
        rollout_cache=rollout_cache,
    )

    plot_error_vs_horizon(horizon_metrics, figdir, logy=not linear_error_scale)
    plot_phase_space_colored_errors(phase_data, system, figdir)
    plot_rollout_error_summary(rollout_metrics, figdir)
    plot_initial_condition_heatmap(heatmap_data, system, figdir, heatmap_horizon, heatmap_mode)