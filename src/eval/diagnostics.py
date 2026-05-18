import os
from typing import Dict, List, Tuple
import torch
from src.eval.model_io import predict_rollout_from_x0, supports_mode_subset_rollout
from src.eval.delay_utils import (
    get_model_delay_depth,
    delay_start_index,
    make_rollout_initial_condition,
    make_backward_delay_x0_from_current_states,
)
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection 
from typing import Optional
from matplotlib.lines import Line2D

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
    max_samples: int = 5000,
) -> Dict[str, np.ndarray]:
    """
    Rank regression-DMD modes by dataset-level RMS modal amplitude.

    For lifted states z^(n), we compute modal coefficients
        b^(n) = Phi^\dagger z^(n),
    and then rank mode i by
        s_i = sqrt( (1/N) sum_n |b_i^(n)|^2 ).

    Notes
    -----
    - Uses many states from the dataset, not just one initial condition.
    - Optionally subsamples to keep runtime reasonable.
    """
    X_flat = X.reshape(-1, X.shape[-1])

    # Uniform subsampling over the flattened dataset for efficiency
    if max_samples is not None and X_flat.shape[0] > max_samples:
        idx = np.linspace(0, X_flat.shape[0] - 1, max_samples, dtype=int)
        X_eval = X_flat[idx]
    else:
        X_eval = X_flat

    x = torch.as_tensor(X_eval, dtype=torch.float64)

    # Normalize state the same way the model expects
    x_n = model._normalize_x(x)

    # Lift and apply the same lifted scaling used by the model
    z = model.expand(x_n) / model.psi_scale
    z = z.to(torch.complex128)  # shape: (N, m)

    # Modal coefficients for all samples:
    #   b^(n) = Phi^\dagger z^(n)
    Phi = model.Phi_lift_fitted.to(torch.complex128)           # shape: (m, r)
    Phi_pinv = torch.linalg.pinv(Phi)                          # shape: (r, m)
    B = (Phi_pinv @ z.T).T                                     # shape: (N, r)

    # Dataset-level RMS modal amplitude
    scores = torch.sqrt(torch.mean(torch.abs(B) ** 2, dim=0)).detach().cpu().numpy()

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
    max_samples: int = 5000,
) -> Dict[str, np.ndarray]:
    """
    Rank ML-DMD modes by dataset-level RMS modal amplitude.
    """
    X_flat = X.reshape(-1, X.shape[-1])

    # Uniform subsampling for efficiency
    if max_samples is not None and X_flat.shape[0] > max_samples:
        idx = np.linspace(0, X_flat.shape[0] - 1, max_samples, dtype=int)
        X_eval = X_flat[idx]
    else:
        X_eval = X_flat

    # Move to the correct device/dtype
    x = torch.as_tensor(
        X_eval, 
        dtype=next(model.parameters()).dtype, 
        device=next(model.parameters()).device
    )

    # Replicate the new ML-DMD lifting pipeline
    z = model.expander.expand(x)
    z_norm = model._normalize(z, update_stats=False) 

    # Project to modes
    B = model._get_modal_coords(z_norm)

    # Dataset-level RMS modal amplitude
    scores = torch.sqrt(torch.mean(torch.abs(B) ** 2, dim=0)).detach().cpu().numpy()

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
            elif model_name == "ml_dmd_free" or model_name == "ml_dmd_band":
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
    """
    Build sampled-start heatmap data.

    Important for delay models:
    - The model rollout needs a full delay history.
    - The heatmap plot only needs the current physical state x(t0).
    Therefore:
        rollout initial condition = full delay history
        plotted start location    = X_traj[t0]
    """
    starts = []
    errors = []

    delay_depth = get_model_delay_depth(model_name, model)
    first_valid_t0 = delay_start_index(delay_depth)

    for traj_id in split_idx:
        X_traj = X[:, traj_id, :]
        T = X_traj.shape[0]

        if rollout_cache is not None and traj_id in rollout_cache:
            cache_starts = np.asarray(rollout_cache[traj_id]["starts"], dtype=int)
            cache_rollouts = rollout_cache[traj_id]["rollouts"]

            if len(cache_starts) == 0:
                continue

            if mode == "traj_initials":
                # For non-delay models this is normally t0=0.
                # For delay models this is the first cached valid start,
                # usually delay_depth - 1.
                selected_indices = [0]

            elif mode == "all_valid_starts":
                selected_indices = range(len(cache_starts))

            else:
                raise ValueError(f"Unknown heatmap mode: {mode}")

            for idx in selected_indices:
                t0 = int(cache_starts[idx])
                rollout = cache_rollouts[idx]

                if horizon > rollout.shape[0] - 1:
                    continue
                if t0 + horizon >= T:
                    continue

                # Plot physical current state, not delay-history vector.
                starts.append(X_traj[t0].copy())

                err = rollout[horizon] - X_traj[t0 + horizon]
                errors.append(np.mean(err ** 2))

        else:
            if mode == "traj_initials":
                start_indices = [first_valid_t0]

            elif mode == "all_valid_starts":
                start_stop = T - horizon
                start_indices = range(first_valid_t0, start_stop)

            else:
                raise ValueError(f"Unknown heatmap mode: {mode}")

            for t0 in start_indices:
                if t0 + horizon >= T:
                    continue

                x0 = make_rollout_initial_condition(
                    X_traj=X_traj,
                    t0=int(t0),
                    model_name=model_name,
                    model=model,
                )

                rollout = predict_rollout_from_x0(
                    x0=x0,
                    steps=horizon,
                    model_name=model_name,
                    model=model,
                    extras=extras,
                    mode_indices=mode_indices,
                )

                # Plot physical current state, not delay-history vector.
                starts.append(X_traj[t0].copy())

                err = rollout[horizon] - X_traj[t0 + horizon]
                errors.append(np.mean(err ** 2))

    if len(starts) == 0:
        raise ValueError(
            "No sampled-start heatmap points were computed. "
            f"Check horizon={horizon}, delay_depth={delay_depth}, and heatmap mode='{mode}'."
        )

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

def _format_linear_three_tick_colorbar(cbar, vmin: float, vmax: float):
    """
    Format a linear colorbar with three readable ticks:
    bottom, middle, top.
    """
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

    n = len(horizons)

    use_markers = n <= 60
    marker = "o" if use_markers else None
    marker_size = 3.0 if n > 20 else 5.0
    line_width = 1.8

    plt.figure(figsize=(8, 5))

    plt.plot(
        horizons,
        mse,
        marker=marker,
        markersize=marker_size,
        markeredgewidth=0.0,
        linewidth=line_width,
        label="MSE",
    )
    plt.plot(
        horizons,
        rmse,
        marker=marker,
        markersize=marker_size,
        markeredgewidth=0.0,
        linewidth=line_width,
        label="RMSE",
    )
    plt.plot(
        horizons,
        nrmse,
        marker=marker,
        markersize=marker_size,
        markeredgewidth=0.0,
        linewidth=line_width,
        label="NRMSE",
    )

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
    n = len(horizons)

    use_markers = n <= 60
    marker = "o" if use_markers else None
    marker_size = 3.0 if n > 20 else 5.0
    line_width = 1.8

    plt.figure(figsize=(8, 5))

    plt.plot(
        horizons,
        rollout_metrics["rollout_mse"],
        marker=marker,
        markersize=marker_size,
        markeredgewidth=0.0,
        linewidth=line_width,
        label="MSE",
    )
    plt.plot(
        horizons,
        rollout_metrics["rollout_rmse"],
        marker=marker,
        markersize=marker_size,
        markeredgewidth=0.0,
        linewidth=line_width,
        label="RMSE",
    )
    plt.plot(
        horizons,
        rollout_metrics["rollout_nrmse"],
        marker=marker,
        markersize=marker_size,
        markeredgewidth=0.0,
        linewidth=line_width,
        label="NRMSE",
    )

    plt.xlabel("Prediction horizon")
    plt.ylabel("Metric value")
    plt.title("Full-rollout error across trajectories")
    plt.yscale("log")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "rollout_error_distribution.png"), dpi=200)
    plt.close()

def _compute_one_sided_psd(signal: np.ndarray, dt: float):
    """
    Simple one-sided PSD using the FFT.

    Parameters
    ----------
    signal : array, shape (T,)
        Real-valued time signal.
    dt : float
        Sampling interval in seconds.

    Returns
    -------
    freqs : array
        One-sided frequency axis in Hz.
    psd : array
        One-sided power spectral density.
    """
    x = np.asarray(signal, dtype=float).reshape(-1)
    if x.size < 2:
        raise ValueError("Signal must have length >= 2 for PSD computation.")

    x = x - np.mean(x)
    n = x.size

    freqs = np.fft.rfftfreq(n, d=dt)
    Xf = np.fft.rfft(x)

    # Standard one-sided PSD normalization
    psd = (1.0 / (n * (1.0 / dt))) * (np.abs(Xf) ** 2)
    if n > 2:
        psd[1:-1] *= 2.0

    return freqs, psd


def plot_rollout_error_spectrum(
    *,
    true_rollout: np.ndarray,
    pred_rollout: np.ndarray,
    dt: float,
    figdir: str,
    filename: str = None,
    title_prefix: str = "",
) -> None:
    """
    Plot frequency spectra for the first two state components.

    For each state component, plot:
        - PSD(True)
        - PSD(Pred)
        - PSD(Error)

    Layout:
        - top panel: state x1 (index 0)
        - bottom panel: state x2 (index 1)

    If the system is 1D, only one panel is produced.
    """
    true_rollout = np.asarray(true_rollout)
    pred_rollout = np.asarray(pred_rollout)

    if true_rollout.shape != pred_rollout.shape:
        raise ValueError(
            f"true_rollout and pred_rollout must have the same shape, got "
            f"{true_rollout.shape} and {pred_rollout.shape}"
        )

    if true_rollout.ndim != 2:
        raise ValueError(f"Expected rollout arrays of shape (T, d), got {true_rollout.shape}")

    T, d = true_rollout.shape
    if d < 1:
        raise ValueError("Rollout must contain at least one state dimension.")

    n_panels = min(2, d)

    if filename is None:
        filename = "rollout_frequency_spectra.png"

    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3.8 * n_panels), sharex=True)

    if n_panels == 1:
        axes = [axes]

    for state_dim in range(n_panels):
        ax = axes[state_dim]

        y_true = true_rollout[:, state_dim]
        y_pred = pred_rollout[:, state_dim]
        y_err = y_pred - y_true

        freqs_true, psd_true = _compute_one_sided_psd(y_true, dt)
        freqs_pred, psd_pred = _compute_one_sided_psd(y_pred, dt)
        freqs_err, psd_err = _compute_one_sided_psd(y_err, dt)

        mask_true = (freqs_true > 0) & np.isfinite(psd_true) & (psd_true > 0)
        mask_pred = (freqs_pred > 0) & np.isfinite(psd_pred) & (psd_pred > 0)
        mask_err = (freqs_err > 0) & np.isfinite(psd_err) & (psd_err > 0)

        ax.loglog(freqs_true[mask_true], psd_true[mask_true], label="PSD(True)", linewidth=2.0)
        ax.loglog(freqs_pred[mask_pred], psd_pred[mask_pred], label="PSD(Pred)", linestyle="--", linewidth=2.0)
        ax.loglog(freqs_err[mask_err], psd_err[mask_err], label="PSD(Error)", linewidth=1.8, alpha=0.9)

        ax.set_ylabel("PSD")
        ax.set_title(f"{title_prefix}state x{state_dim + 1}: frequency spectrum")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Frequency [Hz]")

    fig.tight_layout()
    fig.savefig(os.path.join(figdir, filename), dpi=200)
    plt.close(fig)

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

    # Model rollout (Batched / Vectorized)
    # For delay models, construct a physically consistent delay history
    # by integrating the true dynamics backwards from each grid point.
    delay_depth = get_model_delay_depth(model_name, model)

    model_grid_x0 = make_backward_delay_x0_from_current_states(
        current_states=grid_points,
        f_true=f_true,
        dt=dt,
        delay_depth=delay_depth,
    )

    try:
        rollout_batch = predict_rollout_from_x0(
            x0=model_grid_x0,
            steps=horizon,
            model_name=model_name,
            model=model,
            extras=extras,
            mode_indices=mode_indices,
        )

        pred_terminal = rollout_batch[horizon]

    except Exception as e:
        print(f"[diagnostics] Batched rollout failed ({e}), falling back to slow loop...")
        pred_terminal = np.empty_like(true_terminal)

        for k, x0 in enumerate(model_grid_x0):
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


def select_overlay_trajectories(
    *,
    X: np.ndarray,
    split_idx: np.ndarray,
    traj_id: int,
    n_trajs: int,
) -> List[np.ndarray]:
    if n_trajs <= 0:
        return []

    split_idx = list(split_idx)

    if traj_id not in split_idx:
        split_idx = [traj_id] + split_idx

    # Always include the chosen trajectory first
    selected = [traj_id]

    if n_trajs > 1:
        remaining = [tid for tid in split_idx if tid != traj_id]

        if len(remaining) > 0:
            k = min(n_trajs - 1, len(remaining))

            # Spread picks across the split instead of just taking the first few
            if k == len(remaining):
                extra_ids = remaining
            else:
                positions = np.linspace(0, len(remaining) - 1, k, dtype=int)
                extra_ids = [remaining[p] for p in positions]

            selected.extend(extra_ids)

    return [X[:, tid, :] for tid in selected]


def _build_reference_trajectory_color_info(
    trajectory: np.ndarray,
    dims: Tuple[int, int],
):
    """
    Build a colored overlay for the main trajectory based on per-step displacement
    in the plotted phase plane.

    The returned values color each line segment by:
        ||x_{t+1}^{(i,j)} - x_t^{(i,j)}||_2
    """
    i, j = dims
    pts = np.asarray(trajectory[:, [i, j]], dtype=float)

    if pts.shape[0] < 2:
        return None

    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    step_disp = np.linalg.norm(np.diff(pts, axis=0), axis=1)

    finite = step_disp[np.isfinite(step_disp)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
        values = np.zeros_like(step_disp)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        positive = finite[finite > 0]
        if positive.size > 0:
            vmin = np.percentile(positive, 5.0)
            vmax = np.percentile(positive, 95.0)
            if vmax <= vmin:
                vmax = max(positive.max(), vmin + 1e-12)
        else:
            vmin = 0.0
            vmax = max(finite.max(), 1e-12)

        values = np.clip(step_disp, vmin, vmax)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    return {
        "points": pts,
        "segments": segments,
        "values": values,
        "norm": norm,
        "vmin": vmin,
        "vmax": vmax,
    }
    
def build_true_grid_heatmap_specs(
    *,
    model_name: str,
    model,
    extras: Dict[str, np.ndarray],
    X: np.ndarray,
    traj_id: int,
    mode_subset_sizes: Optional[List[int]],
    mode_subset_strategy: str,
    mode_subset_indices: Optional[List[int]] = None,
) -> List[Dict[str, object]]:
    """
    Build the column specification for the combined true-grid heatmap figure.

    Returns a list of dicts with:
        - name: short identifier for filename / logic
        - title: subplot column title
        - mode_indices: None for full model, or ndarray of selected modes
    """
    specs: List[Dict[str, object]] = []

    if mode_subset_sizes is None:
        mode_subset_sizes = []

    if len(mode_subset_sizes) > 0 or (mode_subset_strategy == "manual" and mode_subset_indices):
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
            if subset_name.startswith("top") and subset_name.endswith("_amplitude"):
                k = subset_name.replace("top", "").replace("_amplitude", "")
                title = f"top {k} modes"
            else:
                title = subset_name.replace("_", " ")
            specs.append({
                "name": subset_name,
                "title": title,
                "mode_indices": subset_idx,
            })

    specs.append({"name": "all", "title": "all modes", "mode_indices": None})
    return specs

def compute_true_grid_heatmap_grid(
    *,
    data_path: str,
    X: np.ndarray,
    horizons: List[int],
    heatmap_specs: List[Dict[str, object]],
    model_name: str,
    model,
    extras: Dict[str, np.ndarray],
    grid_resolution: int = 100,
) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
    """
    Compute all grid heatmap data needed for the combined subplot figure.

    Returns
    -------
    grid_results[horizon][spec_name] = grid_data
    """
    grid_results: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}

    for h in horizons:
        grid_results[h] = {}
        for spec in heatmap_specs:
            spec_name = str(spec["name"])
            mode_indices = spec["mode_indices"]

            grid_results[h][spec_name] = compute_true_grid_heatmap_data(
                data_path=data_path,
                X=X,
                horizon=h,
                model_name=model_name,
                model=model,
                extras=extras,
                grid_resolution=grid_resolution,
                mode_indices=mode_indices,
            )

    return grid_results

def make_shared_heatmap_norm(
    grid_results: Dict[int, Dict[str, Dict[str, np.ndarray]]],
    force_linear: bool = False,
):
    """
    Build one shared normalization across all subplot heatmaps.

    Default behavior:
        - use log scale if the dynamic range is large
        - otherwise use linear scale

    If force_linear=True:
        - always use linear scaling
    """
    all_errors = []

    for _, spec_dict in grid_results.items():
        for _, grid_data in spec_dict.items():
            errs = np.asarray(grid_data["errors"], dtype=float)
            finite = errs[np.isfinite(errs)]
            if finite.size > 0:
                all_errors.append(finite)

    if len(all_errors) == 0:
        vmin, vmax = 1e-16, 1.0
        return mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True), vmin, vmax, False

    concat_errors = np.concatenate(all_errors)
    positive = concat_errors[concat_errors > 0]

    if positive.size == 0:
        vmin, vmax = 0.0, max(concat_errors.max(), 1.0)
        return mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True), vmin, vmax, False

    if force_linear:
        vmin = max(0.0, np.percentile(concat_errors, 1.0))
        vmax = np.percentile(concat_errors, 99.5)
        if vmax <= vmin:
            vmax = max(concat_errors.max(), vmin + 1e-12)
        return mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True), vmin, vmax, False

    # Default: auto-switch to log if spread is large
    vmin_log = max(np.percentile(positive, 1.0), 1e-16)
    vmax_log = np.percentile(concat_errors, 99.5)
    if vmax_log <= vmin_log:
        vmax_log = max(positive.max(), vmin_log * 10.0)

    ratio = vmax_log / vmin_log

    if ratio >= 50.0:
        return mcolors.LogNorm(vmin=vmin_log, vmax=vmax_log, clip=True), vmin_log, vmax_log, True

    vmin_lin = max(0.0, np.percentile(concat_errors, 1.0))
    vmax_lin = np.percentile(concat_errors, 99.5)
    if vmax_lin <= vmin_lin:
        vmax_lin = max(concat_errors.max(), vmin_lin + 1e-12)

    return mcolors.Normalize(vmin=vmin_lin, vmax=vmax_lin, clip=True), vmin_lin, vmax_lin, False

def plot_true_grid_heatmap_grid(
    *,
    grid_results: Dict[int, Dict[str, Dict[str, np.ndarray]]],
    horizons: List[int],
    heatmap_specs: List[Dict[str, object]],
    system: str,
    figdir: str,
    trajectory_overlay: Optional[np.ndarray] = None,
    trajectory_overlays: Optional[List[np.ndarray]] = None,
    filename: str = "true_grid_error_heatmap_grid.png",
    force_linear_error_scale: bool = False,
    data_path: Optional[str] = None,
) -> None:
    """
    Plot one combined subplot figure:
        rows = horizons
        cols = mode subsets
    with one shared heatmap colorbar and one trajectory-displacement colorbar.
    """
    n_rows = len(horizons)
    n_cols = len(heatmap_specs)

    norm, vmin, vmax, use_log = make_shared_heatmap_norm(
        grid_results,
        force_linear=force_linear_error_scale,
    )

    # --- FIXED: Dynamic Figure Sizing ---
    fig_width = max(5.5, 4.8 * n_cols)
    fig_height = max(5.0, 4.2 * n_rows)
    
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    # Dynamic top margin so titles don't overlap on 1x1 plots
    top_margin = 0.82 if n_rows == 1 else 0.88
    fig.subplots_adjust(left=0.10, right=0.82, bottom=0.12, top=top_margin, wspace=0.22, hspace=0.28)

    mesh_for_cbar = None
    traj_line_for_cbar = None

    first_spec_name = str(heatmap_specs[0]["name"])
    first_grid = grid_results[horizons[0]][first_spec_name]
    dims = tuple(first_grid["dims"].tolist())

    traj_color_info = None
    if trajectory_overlay is not None:
        traj_color_info = _build_reference_trajectory_color_info(trajectory_overlay, dims)

    traj_cmap = mcolors.LinearSegmentedColormap.from_list(
        "traj_overlay_pink",
        ["#f8d4ff", "#f39cf6", "#ec5be8", "#d81b9c", "#a0006d"],
        N=256,
    )

    for row, h in enumerate(horizons):
        dx_grid, dy_grid = None, None
        if data_path is not None:
            try:
                f_true = build_true_dynamics_from_dataset(data_path)
                g_data = grid_results[h][str(heatmap_specs[0]["name"])]
                XX_row, YY_row = g_data["XX"], g_data["YY"]
                i_idx, j_idx = g_data["dims"]
                
                grid_pts = np.tile(g_data["fixed_state"][None, :], (XX_row.size, 1))
                grid_pts[:, i_idx] = XX_row.ravel()
                grid_pts[:, j_idx] = YY_row.ravel()
                
                vf = f_true(0.0, grid_pts)
                dx_grid = vf[:, i_idx].reshape(XX_row.shape)
                dy_grid = vf[:, j_idx].reshape(XX_row.shape)
            except Exception as e:
                pass

        for col, spec in enumerate(heatmap_specs):
            ax = axes[row, col]
            spec_name = str(spec["name"])
            spec_title = str(spec["title"])

            grid_data = grid_results[h][spec_name]
            XX, YY = grid_data["XX"], grid_data["YY"]
            errors = np.asarray(grid_data["errors"], dtype=float)
            i, j = grid_data["dims"]

            if use_log:
                plot_errors = np.where(np.isfinite(errors), np.maximum(errors, vmin), np.nan)
            else:
                plot_errors = np.where(np.isfinite(errors), np.clip(errors, vmin, vmax), np.nan)

            mesh = ax.pcolormesh(XX, YY, plot_errors, shading="auto", cmap="viridis", norm=norm)
            if mesh_for_cbar is None:
                mesh_for_cbar = mesh

            if trajectory_overlays is not None:
                for traj in trajectory_overlays:
                    ax.plot(traj[:, i], traj[:, j], linestyle="-", linewidth=0.7, alpha=0.14, color="white", zorder=2)

            if traj_color_info is not None:
                ax.plot(traj_color_info["points"][:, 0], traj_color_info["points"][:, 1], color="white", linewidth=2.0, alpha=0.10, zorder=3)
                lc = LineCollection(traj_color_info["segments"], cmap=traj_cmap, norm=traj_color_info["norm"], linewidth=1.6, alpha=0.95, zorder=4)
                lc.set_array(traj_color_info["values"])
                ax.add_collection(lc)
                if traj_line_for_cbar is None:
                    traj_line_for_cbar = lc

            if dx_grid is not None and dy_grid is not None:
                ax.contour(XX, YY, dx_grid, levels=[0], colors='red', linewidths=1.5, linestyles='--', alpha=0.85, zorder=5)
                ax.contour(XX, YY, dy_grid, levels=[0], colors='orange', linewidths=1.5, linestyles='--', alpha=0.85, zorder=5)
                
                if row == 0 and col == 0:
                    legend_elements = [
                        Line2D([0], [0], color='red', lw=1.5, linestyle='--', label=r'$\dot{x}_1 = 0$'),
                        Line2D([0], [0], color='orange', lw=1.5, linestyle='--', label=r'$\dot{x}_2 = 0$')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize='small')

            ax.set_xlim(grid_data["xlim"])
            ax.set_ylim(grid_data["ylim"])

            if row == 0:
                ax.set_title(spec_title)

            if col == 0:
                ax.set_ylabel(f"horizon={h}\n\nx{i + 1}")
            else:
                ax.set_ylabel("")

            if row == n_rows - 1:
                ax.set_xlabel(f"x{j + 1}")
            else:
                ax.set_xlabel("")

    # Dynamically place the main title so it never crushes the subplots
    fig.suptitle(f"{_pretty_system_name(system)} — true-grid error heatmaps", fontsize=16, y=top_margin + 0.06)

    # --- FIXED: Colorbar Placement ---
    # Shifted to x=0.84 so they sit cleanly in the right margin regardless of grid size
    cax_err = fig.add_axes([0.84, 0.55, 0.02, 0.26])
    cbar_err = fig.colorbar(mesh_for_cbar, cax=cax_err)
    cbar_err.set_label("Terminal h-step MSE")
    _format_three_tick_colorbar(cbar_err, vmin, vmax, use_log)

    if traj_line_for_cbar is not None and traj_color_info is not None:
        cax_traj = fig.add_axes([0.84, 0.15, 0.02, 0.26])
        cbar_traj = fig.colorbar(traj_line_for_cbar, cax=cax_traj)
        cbar_traj.set_label("Reference trajectory\nper-step displacement")
        _format_linear_three_tick_colorbar(cbar_traj, traj_color_info["vmin"], traj_color_info["vmax"])

    fig.savefig(os.path.join(figdir, filename), dpi=240)
    plt.close(fig)

    # fig.suptitle(f"{_pretty_system_name(system)} — true-grid error heatmaps", fontsize=16)

    # # Error heatmap colorbar: always three ticks
    # cax_err = fig.add_axes([0.89, 0.54, 0.02, 0.30])
    # cbar_err = fig.colorbar(mesh_for_cbar, cax=cax_err)
    # cbar_err.set_label("Terminal h-step MSE")
    # _format_three_tick_colorbar(cbar_err, vmin, vmax, use_log)

    # # Trajectory displacement colorbar
    # if traj_line_for_cbar is not None and traj_color_info is not None:
    #     cax_traj = fig.add_axes([0.89, 0.14, 0.02, 0.26])
    #     cbar_traj = fig.colorbar(traj_line_for_cbar, cax=cax_traj)
    #     cbar_traj.set_label("Reference trajectory\nper-step displacement")
    #     _format_linear_three_tick_colorbar(
    #         cbar_traj,
    #         traj_color_info["vmin"],
    #         traj_color_info["vmax"],
    #     )

    # fig.savefig(os.path.join(figdir, filename), dpi=240)
    # plt.close(fig)

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
    grid_overlay_n_trajs: int = 1,
    mode_subset_sizes: Optional[List[int]] = None,
    mode_subset_strategy: str = "amplitude",
    mode_subset_indices: Optional[List[int]] = None,
    force_linear_true_grid_error_scale: bool = False,
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

    # --------------------------------------------------
    # Combined true-grid heatmap figure
    # --------------------------------------------------
    if run_true_grid_heatmap:
        if data_path is None:
            raise ValueError("data_path is required when run_true_grid_heatmap=True")

        horizons_to_plot = [heatmap_horizon] if true_grid_heatmap_horizons is None else true_grid_heatmap_horizons

        heatmap_specs = build_true_grid_heatmap_specs(
            model_name=model_name,
            model=model,
            extras=extras,
            X=X,
            traj_id=traj_id,
            mode_subset_sizes=mode_subset_sizes,
            mode_subset_strategy=mode_subset_strategy,
            mode_subset_indices=mode_subset_indices,
        )

        print("[diagnostics] Building combined true-grid heatmap figure...")
        print("[diagnostics] Horizons:", horizons_to_plot)
        print("[diagnostics] Columns:", [spec["title"] for spec in heatmap_specs])

        grid_results = compute_true_grid_heatmap_grid(
            data_path=data_path,
            X=X,
            horizons=horizons_to_plot,
            heatmap_specs=heatmap_specs,
            model_name=model_name,
            model=model,
            extras=extras,
            grid_resolution=grid_resolution,
        )

        X_traj = X[:, traj_id, :]
        overlay_trajs = (
            select_overlay_trajectories(
                X=X,
                split_idx=split_idx,
                traj_id=traj_id,
                n_trajs=grid_overlay_n_trajs,
            )
            if overlay_true_trajectory_on_grid and grid_overlay_n_trajs > 1
            else None
        )

        plot_true_grid_heatmap_grid(
            grid_results=grid_results,
            horizons=horizons_to_plot,
            heatmap_specs=heatmap_specs,
            system=system,
            figdir=figdir,
            trajectory_overlay=X_traj if overlay_true_trajectory_on_grid else None,
            trajectory_overlays=overlay_trajs,
            force_linear_error_scale=force_linear_true_grid_error_scale,
            data_path=data_path,
        )
 