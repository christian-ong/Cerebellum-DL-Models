
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.models.linear_baseline import rollout_linear_map
from src.models.dmd_baseline import rollout_dmd_eig
from src.models.ml_dmd import ML_DMD
from src.models.ml_eigen_dmd import MLEigenDMD
from src.models.manual_expansion_ml_dmd import ManualExpansion_MLDMD
from src.models.manual_expansion_manual_dmd import ManualExpansion_ManualDMD
from src.models.manual_expansion_eigen_dmd import ManualExpansion_EigenDMD


def parse_int_list(text: str) -> List[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("At least one horizon must be provided.")
    return sorted(set(values))


def infer_run_name(model_path: str, explicit_name: str = None) -> str:
    if explicit_name:
        return explicit_name
    return os.path.basename(os.path.dirname(model_path))


def get_phase_dims(system: str, state_dim: int) -> Tuple[int, int]:
    if system == "lorenz" and state_dim >= 3:
        return 0, 2
    if state_dim < 2:
        raise ValueError("Phase-space plots require state_dim >= 2.")
    return 0, 1


def predict_rollout_from_x0(
    *,
    x0: np.ndarray,
    steps: int,
    model_name: str,
    model,
    M=None,
    Lambda=None,
    Phi=None,
    K=None,
    C=None,
) -> np.ndarray:
    if model_name == "linear_baseline":
        return rollout_linear_map(M, x0=x0, steps=steps)

    if model_name == "dmd_baseline":
        return rollout_dmd_eig(Lambda, Phi, x0=x0, steps=steps)

    if model_name == "manual_expansion_manual_dmd":
        return model.rollout(K=K, C=C, x0=x0, steps=steps).detach().cpu().numpy()

    return model.rollout(x0=x0, steps=steps).detach().cpu().numpy()


def load_model(args, state_dim: int, system: str, device: str):
    model = None
    extras: Dict[str, np.ndarray] = {}

    if args.model == "linear_baseline":
        model_data = np.load(args.model_path)
        extras["M"] = model_data["M"]
        return model, extras

    if args.model == "dmd_baseline":
        model_data = np.load(args.model_path)
        extras["Lambda"] = model_data["Lambda"]
        extras["Phi"] = model_data["Phi"]
        return model, extras

    if args.model == "manual_expansion_manual_dmd":
        model_data = np.load(args.model_path, allow_pickle=True)
        extras["K"] = model_data["K"]

        if "C" not in model_data:
            raise ValueError(
                "Checkpoint is missing decoder matrix C. "
                "Please retrain manual_expansion_manual_dmd with the updated EDMD-style implementation."
            )
        extras["C"] = model_data["C"]

        degree = int(model_data["expansion_degree"]) if "expansion_degree" in model_data else 3

        if "constant_expansion" in model_data:
            constant_expansion = bool(np.asarray(model_data["constant_expansion"]).item())
        elif "include_bias" in model_data:
            constant_expansion = bool(np.asarray(model_data["include_bias"]).item())
        else:
            constant_expansion = True

        if "sine_cosine_expansion" in model_data:
            sine_cosine_expansion = bool(np.asarray(model_data["sine_cosine_expansion"]).item())
        else:
            sine_cosine_expansion = False

        expansion_type = str(model_data["expansion_type"]) if "expansion_type" in model_data else "general"

        if "system_basis" in model_data:
            system_basis = str(model_data["system_basis"])
            if system_basis == "":
                system_basis = None
        else:
            system_basis = system if expansion_type == "specific" else None

        model = ManualExpansion_ManualDMD(
            state_dim=state_dim,
            expansion_degree=degree,
            constant_expansion=constant_expansion,
            sine_cosine_expansion=sine_cosine_expansion,
            expansion_type=expansion_type,
            system=system_basis,
        ).to(device)
        model.eval()
        return model, extras

    if args.model == "ml_dmd":
        ckpt = torch.load(args.model_path, map_location=device)
        model = ML_DMD(state_dim=ckpt["state_dim"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, extras

    if args.model == "ml_eigen_dmd":
        ckpt = torch.load(args.model_path, map_location=device)
        model = MLEigenDMD(state_dim=ckpt["state_dim"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, extras

    if args.model == "manual_expansion_ml_dmd":
        ckpt = torch.load(args.model_path, map_location=device)
        train_args = ckpt["train_args"]
        model = ManualExpansion_MLDMD(
            state_dim=ckpt["state_dim"],
            expansion_degree=train_args["expansion_degree"],
            expansion_type=train_args["expansion_type"],
            system=ckpt["system"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, extras

    if args.model == "manual_expansion_eigen_dmd":
        ckpt = torch.load(args.model_path, map_location=device)
        train_args = ckpt["train_args"]
        model = ManualExpansion_EigenDMD(
            state_dim=ckpt["state_dim"],
            expansion_degree=train_args["expansion_degree"],
            expansion_type=train_args["expansion_type"],
            system=ckpt["system"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, extras

    raise ValueError(f"Unknown model: {args.model}")


def compute_error_vs_horizon(
    X: np.ndarray,
    val_idx: np.ndarray,
    horizons: List[int],
    model_name: str,
    model,
    extras: Dict[str, np.ndarray],
    start_stride: int = 1,
    max_starts_per_traj: int = None,
) -> Dict[str, np.ndarray]:
    """
    Terminal h-step error over all validation trajectories and valid starting times.

    Faster version:
    for each starting point, rollout once up to max(horizons), then read off
    the errors for all requested horizons from that single rollout.
    """
    T, _, state_dim = X.shape
    max_h = max(horizons)
    per_horizon = {h: [] for h in horizons}

    for traj_id in val_idx:
        X_traj = X[:, traj_id, :]
        n_valid_starts = T - max_h
        if n_valid_starts <= 0:
            raise ValueError(f"Trajectory length {T} is too short for max horizon {max_h}.")

        start_indices = np.arange(0, n_valid_starts, start_stride)

        if max_starts_per_traj is not None and len(start_indices) > max_starts_per_traj:
            keep = np.linspace(0, len(start_indices) - 1, max_starts_per_traj, dtype=int)
            start_indices = start_indices[keep]

        for t0 in start_indices:
            x0 = X_traj[t0]
            rollout = predict_rollout_from_x0(
                x0=x0,
                steps=max_h,
                model_name=model_name,
                model=model,
                **extras,
            )

            for h in horizons:
                err = np.mean((rollout[h] - X_traj[t0 + h]) ** 2)
                per_horizon[h].append(err)

    mean = np.array([np.mean(per_horizon[h]) for h in horizons])
    std = np.array([np.std(per_horizon[h]) for h in horizons])
    median = np.array([np.median(per_horizon[h]) for h in horizons])
    q25 = np.array([np.quantile(per_horizon[h], 0.25) for h in horizons])
    q75 = np.array([np.quantile(per_horizon[h], 0.75) for h in horizons])

    return {
        "horizons": np.array(horizons, dtype=int),
        "mean_mse": mean,
        "std_mse": std,
        "median_mse": median,
        "q25_mse": q25,
        "q75_mse": q75,
    }


def compute_phase_error_for_trajectory(
    X: np.ndarray,
    traj_id: int,
    horizons: List[int],
    model_name: str,
    model,
    extras: Dict[str, np.ndarray],
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    For one chosen trajectory, color each true starting point by the terminal h-step error.

    Faster version:
    for each starting point, rollout once up to max(horizons), then read off
    errors for all requested horizons.
    """
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
            **extras,
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


def compute_distribution_over_rollouts(
    X: np.ndarray,
    val_idx: np.ndarray,
    horizons: List[int],
    model_name: str,
    model,
    extras: Dict[str, np.ndarray],
) -> Dict[int, np.ndarray]:
    """
    Distribution of full-trajectory rollout MSE when starting from the first point of each validation trajectory.
    """
    out = {}
    for h in horizons:
        mse_list = []
        for traj_id in val_idx:
            X_true = X[: h + 1, traj_id, :]
            x0 = X_true[0]
            rollout = predict_rollout_from_x0(
                x0=x0,
                steps=h,
                model_name=model_name,
                model=model,
                **extras,
            )
            mse_list.append(np.mean((rollout - X_true) ** 2))
        out[h] = np.asarray(mse_list)
    return out


def compute_initial_condition_heatmap_data(
    X: np.ndarray,
    val_idx: np.ndarray,
    horizon: int,
    model_name: str,
    model,
    extras: Dict[str, np.ndarray],
    mode: str = "traj_initials",
) -> Dict[str, np.ndarray]:
    """
    mode='traj_initials': use x(0) from each validation trajectory.
    mode='all_valid_starts': use every valid starting point from each validation trajectory.
    """
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
                **extras,
            )
            starts.append(X_traj[t0].copy())
            errors.append(np.mean((rollout[horizon] - X_traj[t0 + horizon]) ** 2))

    return {
        "starts": np.asarray(starts),
        "errors": np.asarray(errors),
    }


def plot_error_vs_horizon(stats: Dict[str, np.ndarray], figdir: str, logy: bool = True) -> None:
    horizons = stats["horizons"]
    mean = stats["mean_mse"]
    q25 = stats["q25_mse"]
    q75 = stats["q75_mse"]
    median = stats["median_mse"]

    plt.figure(figsize=(7, 5))
    plt.plot(horizons, mean, label="Mean MSE")
    plt.plot(horizons, median, linestyle="--", label="Median MSE")
    plt.fill_between(horizons, q25, q75, alpha=0.25, label="25-75% quantile")

    plt.xlabel("Prediction horizon")
    plt.ylabel("Terminal h-step MSE")
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

    # Special case: only one horizon
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

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Terminal h-step MSE")

        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "phase_space_error_maps.png"), dpi=200)
        plt.close(fig)
        return

    # Multi-horizon case
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
    cbar = fig.colorbar(scatter_for_colorbar, cax=cax)
    cbar.set_label("Terminal h-step MSE")

    fig.savefig(os.path.join(figdir, "phase_space_error_maps.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_error_distribution(
    dist: Dict[int, np.ndarray],
    figdir: str,
) -> None:
    horizons = sorted(dist.keys())
    data = [dist[h] for h in horizons]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=[str(h) for h in horizons], showfliers=True)
    plt.xlabel("Prediction horizon")
    plt.ylabel("Full-rollout MSE per validation trajectory")
    plt.title("Distribution of rollout errors across validation trajectories")
    plt.yscale("log")
    plt.grid(True, axis="y", alpha=0.3)
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
    plt.colorbar(sc, label="Terminal h-step MSE")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, f"initial_condition_error_map_h{horizon}_{mode}.png"), dpi=200)
    plt.close()


def save_summary_npz(
    figdir: str,
    horizon_stats: Dict[str, np.ndarray],
    rollout_dist: Dict[int, np.ndarray],
    phase_data: Dict[int, Dict[str, np.ndarray]],
    heatmap_data: Dict[str, np.ndarray],
) -> None:
    payload = dict(horizon_stats)
    for h, arr in rollout_dist.items():
        payload[f"rollout_dist_h{h}"] = arr
    for h, d in phase_data.items():
        payload[f"phase_starts_h{h}"] = d["starts"]
        payload[f"phase_errors_h{h}"] = d["errors"]
    payload["heatmap_starts"] = heatmap_data["starts"]
    payload["heatmap_errors"] = heatmap_data["errors"]

    np.savez(os.path.join(figdir, "diagnostics_summary.npz"), **payload)


def main():
    parser = argparse.ArgumentParser(description="Diagnostic evaluation plots for dynamical-system prediction models.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "linear_baseline",
            "dmd_baseline",
            "ml_dmd",
            "ml_eigen_dmd",
            "manual_expansion_ml_dmd",
            "manual_expansion_manual_dmd",
            "manual_expansion_eigen_dmd",
        ],
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--name", type=str, help="Optional suffix for saved figure folder")
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,2,5,10,20,50,100",
        help="Comma-separated horizons for the error-vs-horizon curve and rollout distribution.",
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
        help="Which validation trajectory to use for the phase-space colored error maps.",
    )
    parser.add_argument(
        "--linear_error_scale",
        action="store_true",
        help="Use linear instead of log scale on the error-vs-horizon plot.",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(args.data_path)
    X = data["X"]
    if X.ndim != 3:
        raise ValueError("Expected X with shape (time, trajectories, state_dim).")
    if "val_idx" not in data:
        raise ValueError("Dataset does not contain val_idx. Please regenerate it using simulate_data.py.")
    val_idx = data["val_idx"]

    T, n_traj, state_dim = X.shape
    system = os.path.basename(args.data_path).replace("_trajectory.npz", "")
    run_name = infer_run_name(args.model_path, args.name)
    figdir = os.path.join("data", "figures", args.model, system, run_name, "diagnostics")
    os.makedirs(figdir, exist_ok=True)

    print(f"Loaded X with shape {X.shape}.")
    print(f"Using {len(val_idx)} validation trajectories.")
    print(f"Saving diagnostic figures to: {figdir}")

    model, extras = load_model(args, state_dim=state_dim, system=system, device=device)

    horizons = parse_int_list(args.horizons)
    phase_horizons = parse_int_list(args.phase_horizons)

    max_needed = max(max(horizons), max(phase_horizons), args.heatmap_horizon)
    if T <= max_needed:
        raise ValueError(
            f"Trajectory length T={T} is too short for requested max horizon {max_needed}. "
            "Use smaller horizons."
        )

    if args.traj_index >= len(val_idx):
        raise IndexError(f"traj_index={args.traj_index} but only {len(val_idx)} validation trajectories exist.")
    traj_id = val_idx[args.traj_index]

    print("Computing error-vs-horizon statistics...")
    horizon_stats = compute_error_vs_horizon(
        X=X,
        val_idx=val_idx,
        horizons=horizons,
        model_name=args.model,
        model=model,
        extras=extras,
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

    print("Computing rollout error distributions...")
    rollout_dist = compute_distribution_over_rollouts(
        X=X,
        val_idx=val_idx,
        horizons=horizons,
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
    plot_error_vs_horizon(horizon_stats, figdir, logy=not args.linear_error_scale)
    plot_phase_space_colored_errors(phase_data, system, figdir)
    plot_rollout_error_distribution(rollout_dist, figdir)
    plot_initial_condition_heatmap(heatmap_data, system, figdir, args.heatmap_horizon, args.heatmap_mode)
    save_summary_npz(figdir, horizon_stats, rollout_dist, phase_data, heatmap_data)

    print("\nDone.")
    print("Saved:")
    print(f"  - {os.path.join(figdir, 'error_vs_horizon.png')}")
    print(f"  - {os.path.join(figdir, 'phase_space_error_maps.png')}")
    print(f"  - {os.path.join(figdir, 'rollout_error_distribution.png')}")
    print(f"  - {os.path.join(figdir, f'initial_condition_error_map_h{args.heatmap_horizon}_{args.heatmap_mode}.png')}")
    print(f"  - {os.path.join(figdir, 'diagnostics_summary.npz')}")


if __name__ == "__main__":
    main()

