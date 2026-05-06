import os
from typing import Dict, List, Tuple
import torch
from src.eval.model_io import predict_rollout_from_x0, supports_mode_subset_rollout
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

from typing import Optional

from src.data_generation.data_simulation import (
    simulate,
    linear_system,
    vanderpol_system,
    lotka_volterra_system,
    pendulum_system,
    lorenz_system,
    duffing_system,
    closed_small_system,
    closed_large_system,
    closed_trig_small_system,
    closed_trig_medium_system,
    closed_trig_large_system,
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

def compute_regression_mode_ranking_by_amplitude(
    *,
    X: np.ndarray,
    traj_id: int,
    model,
) -> Dict[str, np.ndarray]:
    X_traj = X[:, traj_id, :]
    x0 = torch.as_tensor(X_traj[0], dtype=torch.float64)

    if x0.ndim == 1:
        x0 = x0.unsqueeze(0)

    x0_n = model._normalize_x(x0)
    z0 = (model.expand(x0_n) / model.psi_scale)[0].to(torch.complex128)

    Phi = model.Phi_lift_fitted.to(torch.complex128)
    b0 = torch.linalg.pinv(Phi) @ z0
    scores = torch.abs(b0).detach().cpu().numpy()

    ranked_indices = np.argsort(-scores)

    return {
        "ranked_indices": ranked_indices,
        "scores": scores,
    }

def compute_ml_dmd_mode_ranking_by_amplitude(
    *,
    X: np.ndarray,
    traj_id: int,
    model,
) -> Dict[str, np.ndarray]:
    """Calculates the activation amplitude of each mode specifically for the ML_DMD model."""
    X_traj = X[:, traj_id, :]
    x0 = torch.as_tensor(X_traj[0], dtype=torch.float32)
    if x0.ndim == 1:
        x0 = x0.unsqueeze(0)

    # Replicate the ML-DMD lifting pipeline
    x0_scaled = model.scale_state(x0)
    z0 = model.expand(x0_scaled)
    z0_norm = z0 / model.z_scale

    # Project to modes
    Phi = model.Phi
    Phi_inv = model.Phi_inv if hasattr(model, "Phi_inv") else torch.linalg.pinv(Phi)
    b0 = (Phi_inv @ z0_norm.T).squeeze()
    
    # Rank by magnitude of activation
    scores = torch.abs(b0).detach().cpu().numpy()
    ranked_indices = np.argsort(-scores)

    return {
        "ranked_indices": ranked_indices,
        "scores": scores,
    }

def resolve_mode_subsets(
    *,
    model_name: str,
    model,
    extras: Dict[str, np.ndarray],
    X: np.ndarray,
    traj_id: int,
    subset_sizes: List[int],
    subset_strategy: str,
    manual_indices: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    if len(subset_sizes) == 0 and subset_strategy != "manual":
        return {}

    if not supports_mode_subset_rollout(model_name, model, extras):
        raise ValueError(
            f"Mode-subset heatmaps are not supported for model '{model_name}' "
            f"with rollout mode '{extras.get('rollout_mode', 'n/a')}'."
        )

    if subset_strategy == "manual":
        if manual_indices is None or len(manual_indices) == 0:
            raise ValueError("subset_strategy='manual' requires mode_subset_indices.")
        ranked_indices = np.asarray(manual_indices, dtype=int)
        return {"manual": ranked_indices}

    if subset_strategy == "amplitude":
            if model_name == "regression_dmd":
                info = compute_regression_mode_ranking_by_amplitude(X=X, traj_id=traj_id, model=model)
            elif model_name == "ml_dmd":
                info = compute_ml_dmd_mode_ranking_by_amplitude(X=X, traj_id=traj_id, model=model)
            else:
                raise ValueError(f"Amplitude ranking not implemented for {model_name}")
                
            ranked_indices = info["ranked_indices"]
            print(f"[diagnostics] Mode ranking strategy: amplitude ({model_name})")
            print("[diagnostics] Top ranked modes:", ranked_indices[: min(10, len(ranked_indices))].tolist())

            subsets = {}
            for k in subset_sizes:
                if k > 0:
                    subsets[f"top{k}_amplitude"] = ranked_indices[:k]
            return subsets

    raise ValueError(f"Unknown subset strategy: {subset_strategy}")

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
    mode_indices: Optional[np.ndarray] = None,
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
                    mode_indices=mode_indices,
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
    filename_suffix: str = "",
    title_suffix: str = "",
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
        f"(h={horizon}, mode={_pretty_heatmap_mode(mode)}){title_suffix}"
    )

    cbar = plt.colorbar(sc, label="Terminal h-step MSE")
    _format_three_tick_colorbar(cbar, vmin, vmax, use_log)

    plt.tight_layout()
    plt.savefig(
        os.path.join(figdir, f"initial_condition_error_map_h{horizon}_{mode}{filename_suffix}.png"),
        dpi=200,
    )
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
    
    if system in {"closed_small"}:
        return closed_small_system(
            mu=float(_np_scalar(data, "mu")),
            alpha=float(_np_scalar(data, "alpha")),
        )

    if system in {"closed_large"}:
        return closed_large_system(
            mu=float(_np_scalar(data, "mu")),
            alpha=float(_np_scalar(data, "alpha")),
            beta=float(_np_scalar(data, "beta")),
            gamma=float(_np_scalar(data, "gamma")),
            delta=float(_np_scalar(data, "delta")),
        )

    if system in {"closed_trig_small"}:
        return closed_trig_small_system(
            omega=float(_np_scalar(data, "omega")),
            alpha=float(_np_scalar(data, "alpha")),
            beta_s1=float(_np_scalar(data, "beta_s1")),
            beta_c1=float(_np_scalar(data, "beta_c1")),
            beta_x=float(_np_scalar(data, "beta_x")),
            beta_x2=float(_np_scalar(data, "beta_x2")),
        )
    
    if system in {"closed_trig_medium"}:
        return closed_trig_medium_system(
            omega=float(_np_scalar(data, "omega")),
            alpha=float(_np_scalar(data, "alpha")),
            beta_s1=float(_np_scalar(data, "beta_s1")),
            beta_c1=float(_np_scalar(data, "beta_c1")),
            beta_s2=float(_np_scalar(data, "beta_s2")),
            beta_c2=float(_np_scalar(data, "beta_c2")),
            beta_x=float(_np_scalar(data, "beta_x")),
            beta_x2=float(_np_scalar(data, "beta_x2")),
        )
    
    if system in {"closed_trig_large"}:
        return closed_trig_large_system(
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
        "koopman_poly": "Closed Small",
        "closed_small": "Closed Small",
        "koopman_poly_large": "Closed Large",
        "closed_large": "Closed Large",
        "koopman_poly_trig": "Closed Trig Small",
        "closed_trig": "Closed Trig Small",
        "closed_trig_small": "Closed Trig Small",
        "closed_trig_medium": "Closed Trig Medium",
        "closed_trig_large": "Closed Trig Large",
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
    if system in {"koopman_poly", "closed_small"}:
        return (-1.0, 1.0), (-1.0, 1.5)

    if system in {"koopman_poly_large", "closed_large"}:
        return (-1.0, 1.0), (-1.0, 1.0)

    if system in {"koopman_poly_trig", "closed_trig", "closed_trig_small", "closed_trig_medium", "closed_trig_large"}:
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
    mode_indices: Optional[np.ndarray] = None,
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
            mode_indices=mode_indices,
        )
        pred_terminal[k] = rollout[horizon]

    diff = pred_terminal - true_terminal

    # Mark invalid or extreme predictions before squaring
    invalid_mask = ~np.isfinite(diff).all(axis=1)

    # Optional: treat absurdly large values as unstable too
    large_mask = np.max(np.abs(diff), axis=1) > 1e150

    bad_mask = invalid_mask | large_mask

    errors_flat = np.empty(diff.shape[0], dtype=np.float64)
    errors_flat.fill(np.inf)

    good_mask = ~bad_mask
    if np.any(good_mask):
        diff_good = diff[good_mask]
        errors_flat[good_mask] = np.mean(diff_good * diff_good, axis=1)

    n_bad = int(np.sum(bad_mask))
    if n_bad > 0:
        print(
            f"[diagnostics] Warning: {n_bad}/{diff.shape[0]} grid points produced non-finite "
            f"or overflow-prone errors for horizon h={horizon}."
        )

    errors = errors_flat.reshape(XX.shape)

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
    finite_errors = errors[np.isfinite(errors)]
    positive_errors = finite_errors[finite_errors > 0]

    if positive_errors.size == 0:
        vmin, vmax = 1e-16, 1.0
        return mcolors.Normalize(vmin=vmin, vmax=vmax), vmin, vmax, False

    vmin = max(np.percentile(positive_errors, 1.0), 1e-16)
    vmax = np.percentile(finite_errors, 99.0) if finite_errors.size > 0 else 1.0
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
    trajectory_overlay: Optional[np.ndarray] = None,
    filename_suffix: str = "",
    title_suffix: str = "",
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

    # Optional overlay of true trajectory in the plotted phase plane
    if trajectory_overlay is not None:
        ax.plot(
            trajectory_overlay[:, i],
            trajectory_overlay[:, j],
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
            color="black",
            label="True trajectory",
        )
        ax.legend(loc="upper right")

    ax.set_xlabel(f"x{i + 1}")
    ax.set_ylabel(f"x{j + 1}")
    ax.set_xlim(grid_data["xlim"])
    ax.set_ylim(grid_data["ylim"])
    ax.set_title(f"{_pretty_system_name(system)} — true grid error heatmap (h={horizon}){title_suffix}")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Terminal h-step MSE")
    _format_three_tick_colorbar(cbar, vmin, vmax, use_log)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figdir, f"true_grid_error_heatmap_h{horizon}{filename_suffix}.png"),
        dpi=220,
    )
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
    phase_horizons: Optional[List[int]],
    heatmap_horizon: int,
    heatmap_mode: str,
    linear_error_scale: bool = False,
    rollout_cache: Dict[int, Dict[str, np.ndarray]] = None,
    data_path: Optional[str] = None,
    run_true_grid_heatmap: bool = False,
    grid_resolution: int = 100,
    true_grid_heatmap_horizons: Optional[List[int]] = None,
    run_phase_maps: bool = True,
    run_sampled_start_heatmap: bool = False,
    overlay_true_trajectory_on_grid: bool = True,
    mode_subset_sizes: Optional[List[int]] = None,
    mode_subset_strategy: str = "amplitude",
    mode_subset_indices: Optional[List[int]] = None,
) -> None:
    if mode_subset_sizes is None:
        mode_subset_sizes = []

    plot_error_vs_horizon(horizon_metrics, figdir, logy=not linear_error_scale)
    plot_rollout_error_summary(rollout_metrics, figdir)

    # Optional phase-space error maps
    if run_phase_maps and phase_horizons is not None and len(phase_horizons) > 0:
        phase_data = compute_phase_error_for_trajectory(
            X=X,
            traj_id=traj_id,
            horizons=phase_horizons,
            model_name=model_name,
            model=model,
            extras=extras,
            rollout_cache=rollout_cache,
        )
        plot_phase_space_colored_errors(phase_data, system, figdir)

    # Optional sampled-start heatmap
    if run_sampled_start_heatmap:
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

    # True-grid heatmaps, potentially for multiple horizons
    if run_true_grid_heatmap:
        if data_path is None:
            raise ValueError("data_path is required when run_true_grid_heatmap=True")

        horizons_to_plot = [heatmap_horizon] if true_grid_heatmap_horizons is None else true_grid_heatmap_horizons
        X_traj = X[:, traj_id, :]

        for h in horizons_to_plot:
            grid_data = compute_true_grid_heatmap_data(
                data_path=data_path,
                X=X,
                horizon=h,
                model_name=model_name,
                model=model,
                extras=extras,
                grid_resolution=grid_resolution,
            )

            plot_true_grid_heatmap(
                grid_data,
                system,
                figdir,
                h,
                trajectory_overlay=X_traj if overlay_true_trajectory_on_grid else None,
            )
    # --------------------------------------------------
    # Additional mode-subset heatmaps (optional)
    # --------------------------------------------------
    if len(mode_subset_sizes) > 0 or (mode_subset_strategy == "manual" and mode_subset_indices):
        print("[diagnostics] Computing additional mode-subset heatmaps...")

        subsets = resolve_mode_subsets(
            model_name=model_name,
            model=model,
            extras=extras,
            X=X,
            traj_id=traj_id,
            subset_sizes=mode_subset_sizes,
            subset_strategy=mode_subset_strategy,
            manual_indices=mode_subset_indices,
        )

        for subset_name, subset_idx in subsets.items():
            print(f"[diagnostics] Subset '{subset_name}' uses mode indices: {subset_idx.tolist()}")

            if not run_true_grid_heatmap:
                heatmap_data_subset = compute_initial_condition_heatmap_data(
                    X=X,
                    split_idx=split_idx,
                    horizon=heatmap_horizon,
                    model_name=model_name,
                    model=model,
                    extras=extras,
                    mode=heatmap_mode,
                    rollout_cache=None,
                    mode_indices=subset_idx,
                )
                plot_initial_condition_heatmap(
                    heatmap_data_subset,
                    system,
                    figdir,
                    heatmap_horizon,
                    heatmap_mode,
                    filename_suffix=f"__{subset_name}",
                    title_suffix=f" | {subset_name}",
                )

            if run_true_grid_heatmap:
                if data_path is None:
                    raise ValueError("data_path is required when run_true_grid_heatmap=True")

                horizons_to_plot = [heatmap_horizon] if true_grid_heatmap_horizons is None else true_grid_heatmap_horizons
                X_traj = X[:, traj_id, :]

                for h in horizons_to_plot:
                    grid_data_subset = compute_true_grid_heatmap_data(
                        data_path=data_path,
                        X=X,
                        horizon=h,
                        model_name=model_name,
                        model=model,
                        extras=extras,
                        grid_resolution=grid_resolution,
                        mode_indices=subset_idx,
                    )
                    plot_true_grid_heatmap(
                        grid_data_subset,
                        system,
                        figdir,
                        h,
                        trajectory_overlay=X_traj if overlay_true_trajectory_on_grid else None,
                        filename_suffix=f"__{subset_name}",
                        title_suffix=f" | {subset_name}",
                    )