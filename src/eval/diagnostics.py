import os
from typing import Dict, List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

from src.eval.model_io import predict_rollout_from_x0
from typing import Optional

from src.data_generation.data_simulation import (
    simulate,
    linear_system,
    vanderpol_system,
    lotka_volterra_system,
    pendulum_system,
    lorenz_system,
    duffing_system,
    koopman_poly_system,
    koopman_poly_system_large,
    koopman_poly_trig_system,
)

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

def _pretty_heatmap_mode(mode: str) -> str:
    special = {
        "traj_initials": "trajectory initials",
        "all_valid_starts": "all valid starts",
    }
    return special.get(mode, mode.replace("_", " "))

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


def _format_three_tick_colorbar(cbar, vmin: float, vmax: float, use_log: bool):
    if use_log:
        tick_mid = np.sqrt(vmin * vmax)
    else:
        tick_mid = 0.5 * (vmin + vmax)

    ticks = [vmin, tick_mid, vmax]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1e}" for t in ticks])
    cbar.minorticks_off()


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

    all_errors = np.concatenate([phase_data[h]["errors"] for h in horizons])
    norm, vmin, vmax, use_log = _make_error_norm(all_errors)

    if n == 1:
        h = horizons[0]
        starts = phase_data[h]["starts"]
        errors = phase_data[h]["errors"]

        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(
            starts[:, i],
            starts[:, j],
            c=np.clip(errors, vmin, vmax),
            s=14,
            norm=norm,
        )
        ax.set_xlabel(f"x{i + 1}")
        ax.set_ylabel(f"x{j + 1}")
        ax.set_title(f"{_pretty_system_name(system)} — phase-space error map (h={h})")

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Terminal h-step MSE")
        _format_three_tick_colorbar(cbar, vmin, vmax, use_log)

        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "phase_space_error_maps.png"), dpi=200)
        plt.close(fig)
        return

    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    fig.subplots_adjust(top=0.86, right=0.88, wspace=0.25, hspace=0.30)
    fig.suptitle(f"{_pretty_system_name(system)} — phase-space error maps", fontsize=18, y=0.96)

    scatter_for_colorbar = None

    for ax, h in zip(axes.flatten(), horizons):
        starts = phase_data[h]["starts"]
        errors = phase_data[h]["errors"]

        sc = ax.scatter(
            starts[:, i],
            starts[:, j],
            c=np.clip(errors, vmin, vmax),
            s=12,
            norm=norm,
        )
        if scatter_for_colorbar is None:
            scatter_for_colorbar = sc

        ax.set_xlabel(f"x{i + 1}")
        ax.set_ylabel(f"x{j + 1}")
        ax.set_title(f"h={h}")

    for ax in axes.flatten()[n:]:
        ax.axis("off")

    cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(scatter_for_colorbar, cax=cax)
    cbar.set_label("Terminal h-step MSE")
    _format_three_tick_colorbar(cbar, vmin, vmax, use_log)

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

    norm, vmin, vmax, use_log = _make_error_norm(errors)

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
    plt.title(
        f"{_pretty_system_name(system)} — sampled-start error map\n"
        f"(h={horizon}, mode={_pretty_heatmap_mode(mode)})"
    )

    cbar = plt.colorbar(sc, label="Terminal h-step MSE")
    _format_three_tick_colorbar(cbar, vmin, vmax, use_log)

    plt.tight_layout()
    plt.savefig(os.path.join(figdir, f"initial_condition_error_map_h{horizon}_{mode}.png"), dpi=200)
    plt.close()

def _np_scalar(data, key: str, default=None):
    if key not in data:
        if default is not None:
            return default
        raise KeyError(f"Missing key '{key}' in dataset.")
    arr = np.asarray(data[key])
    return arr.item() if arr.shape == () else arr


def build_true_dynamics_from_dataset(data_path: str):
    data = np.load(data_path, allow_pickle=True)
    system = str(_np_scalar(data, "system"))

    if system in {"linear", "inward_spiral", "harmonic_oscillator", "saddle_point", "degenerate_node"}:
        return linear_system(np.asarray(data["A"], dtype=float))

    if system == "vanderpol":
        return vanderpol_system(mu=float(_np_scalar(data, "mu")))

    if system == "lotka_volterra":
        return lotka_volterra_system(
            alpha=float(_np_scalar(data, "alpha")),
            beta=float(_np_scalar(data, "beta")),
            delta=float(_np_scalar(data, "delta")),
            gamma=float(_np_scalar(data, "gamma")),
        )

    if system == "pendulum":
        return pendulum_system(
            g=float(_np_scalar(data, "g")),
            L=float(_np_scalar(data, "L")),
        )

    if system == "lorenz":
        return lorenz_system(
            sigma=float(_np_scalar(data, "sigma")),
            rho=float(_np_scalar(data, "rho")),
            beta=float(_np_scalar(data, "beta")),
        )

    if system == "duffing":
        return duffing_system(
            alpha=float(_np_scalar(data, "alpha")),
            beta=float(_np_scalar(data, "beta")),
            delta=float(_np_scalar(data, "delta")),
            gamma=float(_np_scalar(data, "gamma")),
            omega=float(_np_scalar(data, "omega")),
        )

    if system == "koopman_poly":
        return koopman_poly_system(
            mu=float(_np_scalar(data, "mu")),
            alpha=float(_np_scalar(data, "alpha")),
        )

    if system == "koopman_poly_large":
        return koopman_poly_system_large(
            mu=float(_np_scalar(data, "mu")),
            alpha=float(_np_scalar(data, "alpha")),
            beta=float(_np_scalar(data, "beta")),
            gamma=float(_np_scalar(data, "gamma")),
            delta=float(_np_scalar(data, "delta")),
        )

    if system == "koopman_poly_trig":
        return koopman_poly_trig_system(
            omega=float(_np_scalar(data, "omega")),
            alpha=float(_np_scalar(data, "alpha")),
            beta_s1=float(_np_scalar(data, "beta_s1")),
            beta_c1=float(_np_scalar(data, "beta_c1")),
            beta_s2=float(_np_scalar(data, "beta_s2")),
            beta_c2=float(_np_scalar(data, "beta_c2")),
            beta_s3=float(_np_scalar(data, "beta_s3")),
            beta_c3=float(_np_scalar(data, "beta_c3")),
            beta_x=float(_np_scalar(data, "beta_x")),
            beta_x2=float(_np_scalar(data, "beta_x2")),
        )

    raise ValueError(f"Unsupported system '{system}' for true-grid diagnostics.")

def _pretty_system_name(system: str) -> str:
    special = {
        "vanderpol": "Van der Pol",
        "lotka_volterra": "Lotka–Volterra",
        "koopman_poly": "Koopman Poly",
        "koopman_poly_large": "Koopman Poly Large",
        "koopman_poly_trig": "Koopman Poly Trig",
    }
    return special.get(system, system.replace("_", " ").title())


def _default_grid_bounds_from_dataset(data, X: np.ndarray, i: int, j: int):
    system = str(_np_scalar(data, "system"))

    # Linear 2D annulus defaults
    if system in {"linear", "inward_spiral", "harmonic_oscillator", "degenerate_node"}:
        return (-1.5, 1.5), (-1.5, 1.5)

    # Saddle point: a=0.5, b=1.5, r_max=1.1
    if system == "saddle_point":
        return (-0.55, 0.55), (-1.65, 1.65)

    # Van der Pol: annulus r_max=3.5
    if system == "vanderpol":
        return (-3.5, 3.5), (-3.5, 3.5)

    # Pendulum: elliptic annulus a=2.8, b=3.5
    if system == "pendulum":
        return (-2.8, 2.8), (-3.5, 3.5)

    # Lotka–Volterra: annulus of radius 3 shifted to equilibrium, clipped positive
    if system == "lotka_volterra":
        alpha = float(_np_scalar(data, "alpha"))
        beta = float(_np_scalar(data, "beta"))
        delta = float(_np_scalar(data, "delta"))
        gamma = float(_np_scalar(data, "gamma"))
        x_star = gamma / delta
        y_star = alpha / beta
        return (max(0.3, x_star - 3.0), x_star + 3.0), (max(0.3, y_star - 3.0), y_star + 3.0)

    # Duffing: two blobs centered near ±sqrt(-alpha/beta), each with a≈0.6, b≈1.2
    if system == "duffing":
        alpha = float(_np_scalar(data, "alpha"))
        beta = float(_np_scalar(data, "beta"))
        if alpha < 0 and beta > 0:
            x_eq = np.sqrt(-alpha / beta)
            return (-(x_eq + 0.6), x_eq + 0.6), (-1.2, 1.2)

    # Closed-form Koopman test systems
    if system == "koopman_poly":
        return (-1.0, 1.0), (-1.0, 1.5)

    if system == "koopman_poly_large":
        return (-1.0, 1.0), (-1.0, 1.0)

    if system == "koopman_poly_trig":
        return (-2.0, 2.0), (-1.0, 1.0)

    # Lorenz or any future system: fallback to data-driven bounds
    return _auto_grid_bounds(X, i, j)

def _auto_grid_bounds(X: np.ndarray, i: int, j: int, pad_frac: float = 0.08):
    xi = X[..., i].reshape(-1)
    xj = X[..., j].reshape(-1)

    x_lo, x_hi = np.percentile(xi, [1.0, 99.0])
    y_lo, y_hi = np.percentile(xj, [1.0, 99.0])

    dx = max(x_hi - x_lo, 1e-8)
    dy = max(y_hi - y_lo, 1e-8)

    return (
        (x_lo - pad_frac * dx, x_hi + pad_frac * dx),
        (y_lo - pad_frac * dy, y_hi + pad_frac * dy),
    )


def compute_true_grid_heatmap_data(
    *,
    data_path: str,
    X: np.ndarray,
    horizon: int,
    model_name: str,
    model,
    extras: Dict[str, np.ndarray],
    grid_resolution: int = 100,
) -> Dict[str, np.ndarray]:
    data = np.load(data_path, allow_pickle=True)
    dt = float(_np_scalar(data, "dt"))
    method = str(_np_scalar(data, "method", "rk4"))
    system = str(_np_scalar(data, "system"))

    state_dim = X.shape[-1]
    i, j = get_phase_dims(system, state_dim)

    (xlim, ylim) = _default_grid_bounds_from_dataset(data, X, i, j)

    xs = np.linspace(xlim[0], xlim[1], grid_resolution)
    ys = np.linspace(ylim[0], ylim[1], grid_resolution)
    XX, YY = np.meshgrid(xs, ys)

    # For dimensions not shown in the plane, fix them at the dataset mean.
    fixed_state = X.reshape(-1, state_dim).mean(axis=0)

    grid_points = np.tile(fixed_state[None, :], (XX.size, 1))
    grid_points[:, i] = XX.ravel()
    grid_points[:, j] = YY.ravel()

    # True system rollout in batch
    f_true = build_true_dynamics_from_dataset(data_path)
    _, X_true_grid = simulate(
        f_true,
        x0=grid_points,
        dt=dt,
        T=horizon * dt,
        method=method,
    )
    true_terminal = X_true_grid[horizon]   # (Ngrid, d)

    # Model rollout (currently single-start loop)
    pred_terminal = np.empty_like(true_terminal)
    for k, x0 in enumerate(grid_points):
        rollout = predict_rollout_from_x0(
            x0=x0,
            steps=horizon,
            model_name=model_name,
            model=model,
            extras=extras,
        )
        pred_terminal[k] = rollout[horizon]

    errors = np.mean((pred_terminal - true_terminal) ** 2, axis=1).reshape(XX.shape)

    return {
        "XX": XX,
        "YY": YY,
        "errors": errors,
        "dims": np.array([i, j], dtype=int),
        "fixed_state": fixed_state,
        "xlim": np.array(xlim, dtype=float),
        "ylim": np.array(ylim, dtype=float),
    }

def _make_error_norm(errors: np.ndarray):
    positive_errors = errors[errors > 0]

    if positive_errors.size == 0:
        vmin, vmax = 1e-16, 1.0
        return mcolors.Normalize(vmin=vmin, vmax=vmax), vmin, vmax, False

    vmin = max(np.percentile(positive_errors, 1.0), 1e-16)
    vmax = np.percentile(errors, 99.0)

    if vmax <= vmin:
        vmax = positive_errors.max()
    if vmax <= vmin:
        vmax = vmin * 10.0

    ratio = vmax / vmin

    # Use log only when the spread is genuinely wide
    if ratio >= 50.0:
        return mcolors.LogNorm(vmin=vmin, vmax=vmax), vmin, vmax, True
    else:
        return mcolors.Normalize(vmin=vmin, vmax=vmax), vmin, vmax, False
    
def plot_true_grid_heatmap(
    grid_data: Dict[str, np.ndarray],
    system: str,
    figdir: str,
    horizon: int,
) -> None:
    XX = grid_data["XX"]
    YY = grid_data["YY"]
    errors = grid_data["errors"]
    i, j = grid_data["dims"]

    norm, vmin, vmax, use_log = _make_error_norm(errors)

    fig, ax = plt.subplots(figsize=(7, 6))
    mesh = ax.pcolormesh(
        XX,
        YY,
        np.clip(errors, vmin, vmax),
        shading="auto",
        norm=norm,
    )

    ax.set_xlabel(f"x{i + 1}")
    ax.set_ylabel(f"x{j + 1}")
    ax.set_xlim(grid_data["xlim"])
    ax.set_ylim(grid_data["ylim"])
    ax.set_title(f"{_pretty_system_name(system)} — true grid error heatmap (h={horizon})")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Terminal h-step MSE")
    _format_three_tick_colorbar(cbar, vmin, vmax, use_log)

    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"true_grid_error_heatmap_h{horizon}.png"), dpi=220)
    plt.close(fig)

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
    data_path: Optional[str] = None,
    run_true_grid_heatmap: bool = False,
    grid_resolution: int = 100,
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

    plot_error_vs_horizon(horizon_metrics, figdir, logy=not linear_error_scale)
    plot_phase_space_colored_errors(phase_data, system, figdir)
    plot_rollout_error_summary(rollout_metrics, figdir)

    # Old sampled-start map: only plot when we are NOT using the full true-grid map
    if not run_true_grid_heatmap:
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
        plot_initial_condition_heatmap(heatmap_data, system, figdir, heatmap_horizon, heatmap_mode)

    if run_true_grid_heatmap:
        if data_path is None:
            raise ValueError("data_path is required when run_true_grid_heatmap=True")

        grid_data = compute_true_grid_heatmap_data(
            data_path=data_path,
            X=X,
            horizon=heatmap_horizon,
            model_name=model_name,
            model=model,
            extras=extras,
            grid_resolution=grid_resolution,
        )
        plot_true_grid_heatmap(grid_data, system, figdir, heatmap_horizon)