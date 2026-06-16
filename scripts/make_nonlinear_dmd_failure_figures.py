import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import torch
from src.models.ml_dmd import ML_DMD

# We reuse your diagnostics helper to reconstruct the true vector field
# from dataset metadata. This avoids duplicating all system definitions here.
from src.eval.diagnostics import build_true_dynamics_from_dataset
from src.data_generation.data_simulation import simulate


SYSTEMS_FIG3 = [
    ("pendulum", "Pendulum"),
    ("vanderpol", "Van der Pol"),
]


def ensure_3d(X):
    X = np.asarray(X)
    if X.ndim == 2:
        return X[:, None, :]
    if X.ndim == 3:
        return X
    raise ValueError(f"Expected X to be 2D or 3D, got shape {X.shape}")


def load_split(data_path, split):
    path = Path(data_path) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Could not find split file: {path}")
    data = np.load(path, allow_pickle=True)
    X = ensure_3d(data["X"])
    return X, data


def rollout_dmd(Lambda, Phi, x0, steps):
    """
    Plain DMD modal rollout:
        x_k = Phi Lambda^k b0
        b0 = pinv(Phi) x0
    """
    Lambda = np.asarray(Lambda, dtype=np.complex128)
    Phi = np.asarray(Phi, dtype=np.complex128)
    x0 = np.asarray(x0, dtype=np.complex128)

    Phi_pinv = np.linalg.pinv(Phi)
    b0 = Phi_pinv @ x0

    ks = np.arange(steps + 1)[:, None]
    coeffs = (Lambda[None, :] ** ks) * b0[None, :]
    X_hat = coeffs @ Phi.T

    X_hat = np.real_if_close(X_hat, tol=1e9)
    if np.iscomplexobj(X_hat):
        max_imag = float(np.max(np.abs(X_hat.imag)))
        if max_imag > 1e-6:
            print(f"[warning] complex DMD rollout, max imaginary part={max_imag:.3e}; taking real part.")
        X_hat = X_hat.real

    return np.asarray(X_hat, dtype=float)

def _to_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).lower() in {"true", "1", "yes", "y"}


def _to_optional_int(v):
    if v is None:
        return None
    if str(v).lower() in {"none", "null"}:
        return None
    return int(v)


def get_model_path(args, system):
    """
    Resolve model path depending on model type.

    dmd_baseline -> model.npz
    ml_dmd       -> model.pt
    """
    if args.model_type == "ml_dmd":
        return Path(args.model_root) / system / args.run_name / "model.pt"

    if args.model_type == "dmd_baseline":
        return Path(args.model_root) / system / args.run_name / "model.npz"

    raise ValueError(f"Unknown model_type={args.model_type}")


def load_ml_dmd_checkpoint(model_path, device="cpu"):
    ckpt = torch.load(model_path, map_location=device)
    train_args = ckpt.get("train_args", {})

    model = ML_DMD(
        state_dim=int(ckpt["state_dim"]),
        expansion_degree=int(train_args.get("expansion_degree", 1)),
        bias=_to_bool(train_args.get("bias", "false"), default=False),
        sine_cosine_expansion=_to_bool(train_args.get("sine_cosine_expansion", "false"), default=False),
        expansion_type=str(train_args.get("expansion_type", "general")),
        system=ckpt.get("system", None) if str(train_args.get("expansion_type", "general")) == "specific" else None,
        delay_depth=int(train_args.get("delay_depth", 1)),
        rbf_n_centers=int(train_args.get("rbf_n_centers", 50)),
        rbf_center_selection=str(train_args.get("rbf_center_selection", "farthest")),
        rbf_bandwidth_mode=str(train_args.get("rbf_bandwidth_mode", "knn")),
        rbf_knn_k=int(train_args.get("rbf_knn_k", 5)),
        hankel_rank=_to_optional_int(train_args.get("hankel_rank", None)),
        l1_weight=float(train_args.get("l1_weight", 1e-6)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt


def load_predictor(args, system):
    model_path = get_model_path(args, system)

    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model file: {model_path}")

    if args.model_type == "dmd_baseline":
        model_npz = np.load(model_path, allow_pickle=True)
        return {
            "type": "dmd_baseline",
            "Lambda": model_npz["Lambda"],
            "Phi": model_npz["Phi"],
            "path": model_path,
        }

    if args.model_type == "ml_dmd":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, ckpt = load_ml_dmd_checkpoint(model_path, device=device)
        return {
            "type": "ml_dmd",
            "model": model,
            "ckpt": ckpt,
            "device": device,
            "path": model_path,
        }

    raise ValueError(f"Unknown model_type={args.model_type}")


def rollout_predictor(predictor, x0, steps):
    """
    Generic rollout wrapper for dmd_baseline and ml_dmd.
    Returns array with shape (steps+1, state_dim).
    """
    if predictor["type"] == "dmd_baseline":
        return rollout_dmd(
            predictor["Lambda"],
            predictor["Phi"],
            x0,
            steps,
        )

    if predictor["type"] == "ml_dmd":
        model = predictor["model"]
        device = predictor["device"]

        x0_tensor = torch.as_tensor(
            x0,
            dtype=next(model.parameters()).dtype,
            device=device,
        )

        with torch.no_grad():
            pred = model.rollout(x0_tensor, steps)

        return pred.detach().cpu().numpy()

    raise ValueError(f"Unknown predictor type={predictor['type']}")


def predictor_one_step_operator(predictor):
    """
    Return the learned one-step state-space operator.

    For ml_dmd this is only directly meaningful for no-expansion models,
    i.e. expansion_degree=1, bias=false, no trig, so latent_dim == state_dim.
    """
    if predictor["type"] == "dmd_baseline":
        return reconstruct_dmd_operator(
            predictor["Lambda"],
            predictor["Phi"],
        )

    if predictor["type"] == "ml_dmd":
        model = predictor["model"]

        with torch.no_grad():
            K = model.get_K().detach().cpu().numpy()

        state_dim = int(model.state_dim)

        if K.shape != (state_dim, state_dim):
            raise ValueError(
                "NN-DMD operator is not a raw state-space matrix. "
                f"Got K.shape={K.shape}, state_dim={state_dim}. "
                "For Figure 2, train with no expansion: "
                "--expansion_type general --expansion_degree 1 "
                "--bias false --sine_cosine_expansion false."
            )

        return np.asarray(K, dtype=float)

    raise ValueError(f"Unknown predictor type={predictor['type']}")


def predictor_label(args):
    if args.model_type == "ml_dmd":
        return "NN-DMD"
    if args.model_type == "dmd_baseline":
        return "Plain DMD"
    return args.model_type

def rmse_over_time(X_true, X_hat):
    n = min(len(X_true), len(X_hat))
    X_true = np.asarray(X_true[:n], dtype=float)
    X_hat = np.asarray(X_hat[:n], dtype=float)
    diff = X_hat - X_true
    return np.sqrt(np.mean(diff * diff, axis=1))


def set_phase_limits(ax, trajectories, pad_frac=0.08):
    pts = []
    for traj in trajectories:
        traj = np.asarray(traj)
        if traj.ndim == 2 and traj.shape[1] >= 2:
            finite = np.isfinite(traj).all(axis=1)
            if np.any(finite):
                pts.append(traj[finite, :2])

    if not pts:
        return

    pts = np.vstack(pts)
    x_min, x_max = np.percentile(pts[:, 0], [1, 99])
    y_min, y_max = np.percentile(pts[:, 1], [1, 99])

    dx = max(x_max - x_min, 1e-8)
    dy = max(y_max - y_min, 1e-8)

    ax.set_xlim(x_min - pad_frac * dx, x_max + pad_frac * dx)
    ax.set_ylim(y_min - pad_frac * dy, y_max + pad_frac * dy)

def get_display_bounds_from_trajectories(X, n_trajs, steps, pad_frac=0.08):
    """
    Bounds from the actually displayed trajectories.
    """
    n_trajs = min(n_trajs, X.shape[1])
    steps = min(steps, X.shape[0] - 1)

    pts = X[: steps + 1, :n_trajs, :2].reshape(-1, 2)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]

    if len(pts) == 0:
        return (-1, 1), (-1, 1)

    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

    dx = max(x_max - x_min, 1e-8)
    dy = max(y_max - y_min, 1e-8)

    return (
        (x_min - pad_frac * dx, x_max + pad_frac * dx),
        (y_min - pad_frac * dy, y_max + pad_frac * dy),
    )

def choose_phase_traj_ids(X, n_trajs, anchor_idx=0):
    """
    Pick a small set of representative trajectories for the phase-space panel.

    The middle and right panels still use args.traj_index only, but the first
    column can show several rollouts to demonstrate that the DMD failure is
    not just one unlucky initial condition.
    """
    X = ensure_3d(X)
    x0 = np.asarray(X[0, :, :2], dtype=float)

    finite = np.isfinite(x0).all(axis=1)
    valid_ids = np.where(finite)[0]
    x0_valid = x0[finite]

    if len(valid_ids) == 0:
        return [0]

    n_pick = min(int(n_trajs), len(valid_ids))
    anchor_idx = int(np.clip(anchor_idx, 0, X.shape[1] - 1))

    # Start with the trajectory used in the time-series/RMSE panels.
    chosen_global = [anchor_idx] if anchor_idx in valid_ids else [int(valid_ids[0])]

    # Farthest-point selection for the remaining phase-space trajectories.
    while len(chosen_global) < n_pick:
        chosen_pts = x0[chosen_global]
        dist_to_chosen = np.min(
            np.linalg.norm(x0_valid[:, None, :] - chosen_pts[None, :, :], axis=2),
            axis=1,
        )

        # Do not reselect already chosen trajectories.
        for gid in chosen_global:
            local_match = np.where(valid_ids == gid)[0]
            if len(local_match) > 0:
                dist_to_chosen[local_match[0]] = -np.inf

        next_local = int(np.argmax(dist_to_chosen))
        chosen_global.append(int(valid_ids[next_local]))

    return chosen_global

def mean_trajectory_rollout_rmse_summary_over_test_rollouts(predictor, X, steps, max_trajs=0):
    """
    Compute per-trajectory rollout RMSE curves over test trajectories.

    For each trajectory i and horizon h:

        r_i(1:h) = sqrt(mean over steps 1..h and state dimensions of error^2)

    Then summarize r_i(1:h) across trajectories using median, 25--75% range,
    and mean.

    This is not terminal RMSE. It measures average rollout error over the
    whole trajectory segment from step 1 to h.
    """
    X = ensure_3d(X)
    steps = min(int(steps), X.shape[0] - 1)

    n_total = X.shape[1]
    if max_trajs is None or int(max_trajs) <= 0:
        traj_ids = np.arange(n_total)
    else:
        traj_ids = np.arange(min(int(max_trajs), n_total))

    per_traj_curves = []

    for traj_id in traj_ids:
        X_true = X[: steps + 1, traj_id, :]
        X_hat = rollout_predictor(predictor, X_true[0], steps)

        n = min(len(X_true), len(X_hat))

        # Exclude h=0 later, but keep index 0 as zero for alignment.
        diff = X_hat[:n] - X_true[:n]

        # Mean squared error over state dimensions at each time step.
        mse_t = np.mean(diff * diff, axis=1)

        # Running rollout RMSE over steps 1..h.
        rmse_curve = np.zeros(n, dtype=float)
        if n > 1:
            running_mse = np.cumsum(mse_t[1:]) / np.arange(1, n)
            rmse_curve[1:] = np.sqrt(running_mse)

        per_traj_curves.append(rmse_curve)

    min_len = min(len(curve) for curve in per_traj_curves)
    per_traj_curves = np.asarray(
        [curve[:min_len] for curve in per_traj_curves],
        dtype=float,
    )

    median_rmse = np.percentile(per_traj_curves, 50, axis=0)
    q25_rmse = np.percentile(per_traj_curves, 25, axis=0)
    q75_rmse = np.percentile(per_traj_curves, 75, axis=0)
    mean_rmse = np.mean(per_traj_curves, axis=0)

    horizons = np.arange(min_len)

    return horizons, median_rmse, q25_rmse, q75_rmse, mean_rmse

def make_figure3(args):
    """
    Figure 3:
    Plain DMD rollout failure on nonlinear systems.

    Final report layout:
        rows    = nonlinear systems
        col 1   = several phase-space rollouts
        col 2   = median per-trajectory rollout RMSE over test trajectories

    This avoids mixing one-trajectory time-series diagnostics with
    multi-trajectory phase-space plots.
    """
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12.0, 8.2),
        squeeze=False,
        constrained_layout=True,
    )

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    panel_idx = 0

    for row, (system, label) in enumerate(SYSTEMS_FIG3):
        data_path = Path(args.data_root) / system
        X, data = load_split(data_path, args.split)
        predictor = load_predictor(args, system)

        # Print data support once so we know whether the heatmap domain extends
        # beyond the sampled trajectory region.
        print_state_support(f"{system} {args.split}", X)

        train_path = data_path / "train.npz"
        if train_path.exists() and args.split != "train":
            X_train, _ = load_split(data_path, "train")
            print_state_support(f"{system} train", X_train)

        steps = min(args.steps, X.shape[0] - 1)
        traj_id = min(args.traj_index, X.shape[1] - 1)

        ax_phase = axes[row, 0]
        ax_rmse = axes[row, 1]

        # ------------------------------------------------------------
        # Left column: several representative phase-space rollouts
        # ------------------------------------------------------------
        phase_traj_ids = choose_phase_traj_ids(
            X,
            n_trajs=args.n_trajs,
            anchor_idx=traj_id,
        )

        phase_plotted = []

        for k, phase_id in enumerate(phase_traj_ids):
            X_phase_true = X[: steps + 1, phase_id, :]
            X_phase_hat = rollout_predictor(predictor, X_phase_true[0], steps)

            phase_plotted.extend([X_phase_true, X_phase_hat])

            ax_phase.plot(
                X_phase_true[:, 0],
                X_phase_true[:, 1],
                color="C0",
                linewidth=1.8,
                alpha=0.75,
                label="True" if k == 0 else None,
            )
            ax_phase.plot(
                X_phase_hat[:, 0],
                X_phase_hat[:, 1],
                "--",
                color="C1",
                linewidth=1.7,
                alpha=0.9,
                label=f"{predictor_label(args)} rollout" if k == 0 else None,
            )
            ax_phase.scatter(
                X_phase_true[0, 0],
                X_phase_true[0, 1],
                s=24,
                color="black",
                alpha=0.55,
                zorder=5,
                label="Initial states" if k == 0 else None,
            )

        ax_phase.set_title(
            f"{panel_labels[panel_idx]} {label}: phase-space rollouts",
            fontsize=12,
        )
        panel_idx += 1

        ax_phase.set_xlabel("$x_1$")
        ax_phase.set_ylabel("$x_2$")
        ax_phase.grid(True, alpha=0.25)
        set_phase_limits(ax_phase, phase_plotted)
        ax_phase.set_box_aspect(1)

        if row == 0:
            ax_phase.legend(loc="best", framealpha=0.95, fontsize=8)

        # ------------------------------------------------------------
        # Right column: per trajectory rollout RMSE over test set
        # ------------------------------------------------------------
        horizons, median_rmse, q25_rmse, q75_rmse, mean_rmse = mean_trajectory_rollout_rmse_summary_over_test_rollouts(
            predictor,
            X,
            steps=steps,
            max_trajs=args.rmse_cap,
        )

        # Skip h=0 on a log plot because the initial error is exactly/near zero.
        h_plot = horizons[1:]
        median_plot = median_rmse[1:]
        q25_plot = q25_rmse[1:]
        q75_plot = q75_rmse[1:]

        ax_rmse.plot(
            h_plot,
            median_plot,
            color="C0",
            linewidth=2.2,
            label=r"Median traj. RMSE$_{1:h}$",
        )

        ax_rmse.fill_between(
            h_plot,
            q25_plot,
            q75_plot,
            color="C0",
            alpha=0.18,
            linewidth=0,
            label="25–75% range",
        )

        ax_rmse.set_yscale("log")
        ax_rmse.set_title(
            f"{panel_labels[panel_idx]} {label}: median trajectory RMSE$_{{1:h}}$",
            fontsize=12,
        )
        panel_idx += 1

        ax_rmse.set_xlabel("Prediction horizon")
        ax_rmse.set_ylabel(r"Trajectory RMSE$_{1:h}$")
        ax_rmse.grid(True, alpha=0.25)

        finite_positive = median_plot[np.isfinite(median_plot) & (median_plot > 0)]
        if finite_positive.size > 0:
            y_min = max(float(np.min(finite_positive)) * 0.8, 1e-8)
            y_max = float(np.max(q75_plot)) * 1.3
            if y_max > y_min:
                ax_rmse.set_ylim(y_min, y_max)

        if row == 0:
            ax_rmse.legend(loc="best", framealpha=0.95, fontsize=8)

    fig.suptitle(
        f"{predictor_label(args)} rollout on nonlinear systems",
        fontsize=16,
    )

    out_path = Path(args.outdir) / "figure3_nonlinear_dmd_rollout_failure.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 3: {out_path}")

def make_figure3_rmse_comparison(args):
    """
    Extra comparison plot:
    Put Pendulum and Van der Pol mean-trajectory rollout RMSE curves on the same axes.

    This makes it easier to compare whether one nonlinear system has larger
    rollout error than the other, since the original Figure 3 uses separate
    RMSE subplots.
    """
    fig, ax = plt.subplots(
        figsize=(7.2, 4.6),
        constrained_layout=True,
    )

    line_handles = []

    for idx, (system, label) in enumerate(SYSTEMS_FIG3):
        data_path = Path(args.data_root) / system
        X, data = load_split(data_path, args.split)
        predictor = load_predictor(args, system)

        steps = min(args.steps, X.shape[0] - 1)

        horizons, median_rmse, q25_rmse, q75_rmse, mean_rmse = (
            mean_trajectory_rollout_rmse_summary_over_test_rollouts(
                predictor,
                X,
                steps=steps,
                max_trajs=args.rmse_cap,
            )
        )

        # Skip h=0 because the initial condition is identical.
        h_plot = horizons[1:]
        median_plot = median_rmse[1:]
        q25_plot = q25_rmse[1:]
        q75_plot = q75_rmse[1:]

        color = f"C{idx}"

        ax.plot(
            h_plot,
            median_plot,
            color=color,
            linewidth=2.4,
            label=label,
        )

        ax.fill_between(
            h_plot,
            q25_plot,
            q75_plot,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

        line_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.4,
                label=label,
            )
        )

    # Add a generic legend entry for the shaded interquartile range.
    band_handle = Line2D(
        [0],
        [0],
        color="gray",
        linewidth=8,
        alpha=0.25,
        label="25--75% range",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel(r"Trajectory RMSE$_{1:h}$")
    ax.set_title(
        f"{predictor_label(args)} trajectory rollout RMSE comparison",
        fontsize=14,
    )
    ax.grid(True, alpha=0.25)

    ax.legend(
        handles=line_handles + [band_handle],
        loc="best",
        framealpha=0.95,
        fontsize=9,
    )

    out_path = Path(args.outdir) / "figure3b_nonlinear_rmse_comparison.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Figure 3b RMSE comparison: {out_path}")

def make_figure3_normalized_rmse_comparison(args):
    """
    Compare nonlinear rollout errors after normalizing by the test-set state scale.

    This is useful because absolute RMSE can be misleading when systems have
    different state ranges. For each system, we divide the terminal RMSE curve
    by a characteristic test-set state scale.
    """
    fig, ax = plt.subplots(
        figsize=(7.2, 4.6),
        constrained_layout=True,
    )

    line_handles = []

    for idx, (system, label) in enumerate(SYSTEMS_FIG3):
        data_path = Path(args.data_root) / system
        X, data = load_split(data_path, args.split)
        predictor = load_predictor(args, system)

        steps = min(args.steps, X.shape[0] - 1)

        horizons, median_rmse, q25_rmse, q75_rmse, mean_rmse = (
            mean_trajectory_rollout_rmse_summary_over_test_rollouts(
                predictor,
                X,
                steps=steps,
                max_trajs=args.rmse_cap,
            )
        )

        # Characteristic state scale over the full test split.
        # This uses both state dimensions.
        X_flat = X[: steps + 1, :, :2].reshape(-1, 2)
        state_scale = np.sqrt(np.mean(np.var(X_flat, axis=0)))
        state_scale = max(float(state_scale), 1e-12)

        h_plot = horizons[1:]
        median_plot = median_rmse[1:] / state_scale
        q25_plot = q25_rmse[1:] / state_scale
        q75_plot = q75_rmse[1:] / state_scale

        color = f"C{idx}"

        ax.plot(
            h_plot,
            median_plot,
            color=color,
            linewidth=2.4,
            label=f"{label}",
        )

        ax.fill_between(
            h_plot,
            q25_plot,
            q75_plot,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

        print(
            f"[normalized RMSE] {label}: "
            f"state_scale={state_scale:.4e}, "
            f"median terminal RMSE h={steps}: {median_rmse[steps]:.4e}, "
            f"normalized={median_rmse[steps] / state_scale:.4e}"
        )

        line_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.4,
                label=label,
            )
        )

    band_handle = Line2D(
        [0],
        [0],
        color="gray",
        linewidth=8,
        alpha=0.25,
        label="25--75% range",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Normalized terminal RMSE")
    ax.set_title(
        f"{predictor_label(args)} normalized terminal rollout RMSE comparison",
        fontsize=14,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(
        handles=line_handles + [band_handle],
        loc="best",
        framealpha=0.95,
        fontsize=9,
    )

    out_path = Path(args.outdir) / "figure3c_nonlinear_normalized_rmse_comparison.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Figure 3c normalized RMSE comparison: {out_path}")

def default_grid_bounds(system, X):
    """
    Heatmap bounds from the actual loaded data split.

    This keeps the dense-grid heatmap inside the state-space region covered
    by the train/test trajectories instead of forcing a hardcoded domain.
    """
    X = ensure_3d(X)

    x = X[..., 0].reshape(-1)
    y = X[..., 1].reshape(-1)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    # Use min/max support with a tiny visual padding.
    # This stays inside the actual sampled regime up to a small margin.
    xlo, xhi = float(np.min(x)), float(np.max(x))
    ylo, yhi = float(np.min(y)), float(np.max(y))

    dx = max(xhi - xlo, 1e-8)
    dy = max(yhi - ylo, 1e-8)

    pad = 0.02

    return (
        (xlo - pad * dx, xhi + pad * dx),
        (ylo - pad * dy, yhi + pad * dy),
    )

def terminal_rmse_grid(Lambda, Phi, X_true_grid, grid_points, horizons):
    max_h = max(horizons)
    n_grid = grid_points.shape[0]

    # Roll out every grid point independently.
    # This is simple and robust for report-size grids.
    pred = np.empty((max_h + 1, n_grid, grid_points.shape[1]), dtype=float)

    for idx in range(n_grid):
        pred[:, idx, :] = rollout_dmd(Lambda, Phi, grid_points[idx], max_h)

    errors = {}
    for h in horizons:
        diff = pred[h] - X_true_grid[h]
        errors[h] = np.sqrt(np.mean(diff * diff, axis=1))

    return errors


def make_error_norm(errors_list):
    all_err = np.concatenate([e[np.isfinite(e)] for e in errors_list])
    positive = all_err[all_err > 0]

    if positive.size == 0:
        return mcolors.Normalize(vmin=0.0, vmax=max(float(np.max(all_err)), 1.0)), False

    vmin = max(float(np.percentile(positive, 1.0)), 1e-14)
    vmax = float(np.percentile(all_err, 99.5))

    if vmax <= vmin:
        vmax = max(float(np.max(positive)), vmin * 10.0)

    if vmax / vmin > 50:
        return mcolors.LogNorm(vmin=vmin, vmax=vmax), True

    return mcolors.Normalize(vmin=0.0, vmax=vmax), False

def numerical_jacobian(f, x_star, eps=1e-6):
    """
    Numerically compute Jacobian of f at x_star.
    """
    x_star = np.asarray(x_star, dtype=float)
    d = x_star.size
    J = np.zeros((d, d), dtype=float)

    for j in range(d):
        e = np.zeros(d)
        e[j] = eps

        fp = np.asarray(f(0.0, (x_star + e)[None, :]))[0]
        fm = np.asarray(f(0.0, (x_star - e)[None, :]))[0]

        J[:, j] = (fp - fm) / (2.0 * eps)

    return J


def rk4_linear_step_matrix(J, dt):
    """
    One-step RK4 map for linear system xdot = J x.
    This matches the RK4 integrator used for the nonlinear simulation.
    """
    I = np.eye(J.shape[0])
    J2 = J @ J
    J3 = J2 @ J
    J4 = J3 @ J

    return (
        I
        + dt * J
        + (dt ** 2 / 2.0) * J2
        + (dt ** 3 / 6.0) * J3
        + (dt ** 4 / 24.0) * J4
    )


def rollout_linear_map(K, x0, steps):
    """
    Roll out x_{k+1} = K x_k.
    """
    x0 = np.asarray(x0, dtype=float)
    out = np.empty((steps + 1, x0.size), dtype=float)
    out[0] = x0

    for k in range(steps):
        out[k + 1] = K @ out[k]

    return out


def grid_error_with_rollout_function(rollout_fn, X_true_grid, grid_points, horizons, mode="terminal"):
    """
    Compute dense-grid error using an arbitrary rollout function.

    mode="terminal":
        error at exactly horizon h.

    mode="mean_rollout":
        mean RMSE over steps 1..h.
    """
    max_h = max(horizons)
    n_grid = grid_points.shape[0]
    d = grid_points.shape[1]

    pred = np.empty((max_h + 1, n_grid, d), dtype=float)

    for idx in range(n_grid):
        pred[:, idx, :] = rollout_fn(grid_points[idx], max_h)

    errors = {}

    for h in horizons:
        if mode == "terminal":
            diff = pred[h] - X_true_grid[h]
            errors[h] = np.sqrt(np.mean(diff * diff, axis=1))

        elif mode == "mean_rollout":
            diff = pred[1 : h + 1] - X_true_grid[1 : h + 1]
            errors[h] = np.sqrt(np.mean(diff * diff, axis=(0, 2)))

        else:
            raise ValueError(f"Unknown mode={mode}")

    return errors

def make_figure4(args):
    system = args.heatmap_system
    label = system.replace("_", " ").title()
    if system == "vanderpol":
        label = "Van der Pol"

    data_path = Path(args.data_root) / system
    X, data = load_split(data_path, args.split)
    predictor = load_predictor(args, system)

    dt = float(np.asarray(data["dt"]).item()) if "dt" in data.files else 0.01
    method = str(np.asarray(data["method"]).item()) if "method" in data.files else "rk4"

    if "dt" in data.files:
        print(f"[Figure 4] dt loaded from dataset: {dt}")
    else:
        print(f"[Figure 4] dt not found in dataset. Using fallback dt: {dt}")

    print(f"[Figure 4] integration method: {method}")

    try:
        A_model = predictor_one_step_operator(predictor)
        print(f"\n[Figure 4] {predictor_label(args)} one-step operator:")
        print(A_model)

        if system == "pendulum":
            effective_stiffness = -A_model[1, 0] / dt
            effective_damping = (1.0 - A_model[1, 1]) / dt
            print(f"[Figure 4] Approx. learned stiffness k ≈ {effective_stiffness:.4f}")
            print(f"[Figure 4] Approx. learned damping   d ≈ {effective_damping:.4f}")

    except Exception as exc:
        print(f"[Figure 4] Skipping one-step operator print: {exc}")

    # Three heatmap horizons
    horizons = parse_int_list(args.heatmap_horizons)
    if len(horizons) != 3:
        raise ValueError("Figure 4 currently expects exactly 3 heatmap horizons, e.g. --heatmap_horizons 1,10,25")

    max_h = max(horizons)

    base_xlim, base_ylim = default_grid_bounds(system, X)

    # Use the actual data-support bounds directly for the heatmap.
    # Do not enlarge the heatmap based only on the displayed overlay trajectories.
    xlim = base_xlim
    ylim = base_ylim

    print(f"[Figure 4] heatmap x1 domain: [{xlim[0]:.4f}, {xlim[1]:.4f}]")
    print(f"[Figure 4] heatmap x2 domain: [{ylim[0]:.4f}, {ylim[1]:.4f}]")
    print(f"[Figure 4] grid resolution: {args.grid_resolution} x {args.grid_resolution}")

    xs = np.linspace(xlim[0], xlim[1], args.grid_resolution)
    ys = np.linspace(ylim[0], ylim[1], args.grid_resolution)
    XX, YY = np.meshgrid(xs, ys)

    grid_points = np.zeros((XX.size, X.shape[-1]), dtype=float)
    fixed_state = X.reshape(-1, X.shape[-1]).mean(axis=0)
    grid_points[:] = fixed_state[None, :]
    grid_points[:, 0] = XX.ravel()
    grid_points[:, 1] = YY.ravel()

    split_npz_path = data_path / f"{args.split}.npz"
    f_true = build_true_dynamics_from_dataset(str(split_npz_path))

    _, X_true_grid = simulate(
        f_true,
        x0=grid_points,
        dt=dt,
        T=max_h * dt,
        method=method,
    )

    vf = f_true(0.0, grid_points)
    dx1_grid = vf[:, 0].reshape(XX.shape)
    dx2_grid = vf[:, 1].reshape(XX.shape)

    if args.heatmap_model == "dmd":
        rollout_fn = lambda x0, steps: rollout_predictor(predictor, x0, steps)
        heatmap_model_label = predictor_label(args)

    elif args.heatmap_model == "local_jacobian":
        x_star = np.zeros(X.shape[-1], dtype=float)
        J = numerical_jacobian(f_true, x_star)
        K_local = rk4_linear_step_matrix(J, dt)

        print("\n[Figure 4] Local Jacobian at origin:")
        print(J)
        print("[Figure 4] Local one-step linear map:")
        print(K_local)

        rollout_fn = lambda x0, steps: rollout_linear_map(K_local, x0, steps)
        heatmap_model_label = "Local Jacobian linearization"

    else:
        raise ValueError(f"Unknown heatmap_model={args.heatmap_model}")

    errors = grid_error_with_rollout_function(
        rollout_fn,
        X_true_grid,
        grid_points,
        horizons,
        mode=args.heatmap_error_type,
    )
    norm, use_log = make_error_norm([errors[h] for h in horizons])

    fig = plt.figure(figsize=(15.0, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=4,
        width_ratios=[1.0, 1.0, 1.0, 0.045],
        wspace=0.12,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
    ]
    cax = fig.add_subplot(gs[0, 3])

    overlay_n = args.n_trajs if args.heatmap_overlay_n_trajs < 0 else args.heatmap_overlay_n_trajs
    n_traj = min(int(overlay_n), X.shape[1])
    steps = min(args.steps, X.shape[0] - 1)

    mesh = None
    for ax, h in zip(axes, horizons):
        plot_err = errors[h].reshape(XX.shape)
        if use_log:
            plot_err = np.maximum(plot_err, norm.vmin)

        mesh = ax.pcolormesh(
            XX,
            YY,
            plot_err,
            shading="auto",
            cmap="viridis",
            norm=norm,
        )

        if args.heatmap_error_type == "terminal":
            err_label = "terminal error"
        else:
            err_label = "mean rollout error"

        ax.set_title(f"h = {h}", fontsize=13)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_box_aspect(1)

        # Overlay a few true trajectories for context
        overlay_ids = choose_phase_traj_ids(
            X,
            n_trajs=n_traj,
            anchor_idx=args.traj_index,
        )

        for j in overlay_ids:
            traj = X[: steps + 1, j, :]
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color="white",
                linewidth=0.9,
                alpha=0.35,
            )

        # Nullclines
        ax.contour(
            XX, YY, dx1_grid,
            levels=[0],
            colors="red",
            linestyles="--",
            linewidths=1.4,
            alpha=0.85,
        )
        ax.contour(
            XX, YY, dx2_grid,
            levels=[0],
            colors="orange",
            linestyles="--",
            linewidths=1.4,
            alpha=0.85,
        )
        # # Small-amplitude / low-energy region for pendulum
        # if system == "pendulum":
        #     E = 0.5 * YY**2 + (1.0 - np.cos(XX))

        #     ax.contour(
        #         XX,
        #         YY,
        #         E,
        #         levels=[0.5],
        #         colors="white",
        #         linestyles="-",
        #         linewidths=2.0,
        #         alpha=0.9,
        #     )

    legend_elements = [
        Line2D([0], [0], color="red", lw=1.5, linestyle="--", label=r"$\dot{x}_1 = 0$"),
        Line2D([0], [0], color="orange", lw=1.5, linestyle="--", label=r"$\dot{x}_2 = 0$"),
        # Line2D([0], [0], color="white", lw=2.0, linestyle="-", label="low-energy region"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", framealpha=0.95, fontsize="small")

    cbar = fig.colorbar(mesh, cax=cax)
    if args.heatmap_error_type == "terminal":
        err_label_full = "terminal h-step RMSE"
    else:
        err_label_full = "mean rollout RMSE over steps 1..h"

    fig.suptitle(
        f"{heatmap_model_label} error on {label}\n"
        f"{err_label_full}",
        fontsize=16,
        y=1.03,
    )

    out_path = Path(args.outdir) / "figure4_plain_dmd_dense_error_map.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 4: {out_path}")

def parse_int_list(s):
    return [int(v.strip()) for v in str(s).split(",") if v.strip()]

def print_state_support(tag, trajs):
    """
    trajs: array of shape (n_traj, T, state_dim)
    assumes x1 = state[..., 0], x2 = state[..., 1]
    """
    states = trajs[..., :2].reshape(-1, 2)

    x1 = states[:, 0]
    x2 = states[:, 1]

    print(f"[{tag}] x1 range: [{x1.min():.4f}, {x1.max():.4f}]")
    print(f"[{tag}] x2 range: [{x2.min():.4f}, {x2.max():.4f}]")
    print(f"[{tag}] x1 1-99 pct: [{np.percentile(x1, 1):.4f}, {np.percentile(x1, 99):.4f}]")
    print(f"[{tag}] x2 1-99 pct: [{np.percentile(x2, 1):.4f}, {np.percentile(x2, 99):.4f}]")

def main():
    parser = argparse.ArgumentParser(description="Make Figure 3 and Figure 4 for nonlinear plain-DMD limitations.")

    parser.add_argument("--data_root", type=str, default="data/trajectories/nonlinear")
    parser.add_argument("--model_root", type=str, default="data/models/dmd_baseline")
    parser.add_argument("--outdir", type=str, default="data/figures/dmd_baseline/nonlinear_failure_section")
    parser.add_argument("--run_name", type=str, default="fullrank")
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--traj_index", type=int, default=0)
    parser.add_argument("--n_trajs", type=int, default=4)
    parser.add_argument(
        "--heatmap_overlay_n_trajs",
        type=int,
        default=10,
        help="Number of true trajectories overlaid on dense heatmaps. Use -1 to reuse --n_trajs, 0 for none.",
    )
    parser.add_argument(
        "--rmse_cap",
        type=int,
        default=0,
        help="Maximum number of test trajectories used for the mean RMSE curve. Use 0 for all.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="ml_dmd",
        choices=["dmd_baseline", "ml_dmd"],
        help="Which trained model type to plot.",
    )
    parser.add_argument("--heatmap_system", type=str, default="pendulum", choices=["pendulum", "vanderpol", "duffing"])
    parser.add_argument("--grid_resolution", type=int, default=120)

    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--heatmap_model",
        type=str,
        default="dmd",
        choices=["dmd", "local_jacobian"],
        help="Model used for Figure 4 heatmap: global trained DMD or local Jacobian linearization.",
    )

    parser.add_argument(
        "--heatmap_error_type",
        type=str,
        default="terminal",
        choices=["terminal", "mean_rollout"],
        help="Heatmap error type: terminal h-step error or mean rollout error over steps 1..h.",
    )

    parser.add_argument(
        "--heatmap_horizons",
        type=str,
        default="1,25,100",
        help="Comma-separated horizons for Figure 4, e.g. 1,10,25.",
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    make_figure3(args)
    make_figure4(args)

    print("\nDone.")
    print(f"Outputs written to: {args.outdir}")


if __name__ == "__main__":
    main()
