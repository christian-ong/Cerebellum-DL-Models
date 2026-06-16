import argparse
import csv
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import torch
from src.models.ml_dmd import ML_DMD


SYSTEMS = [
    ("saddle_point", "Saddle point", "Captured unstable/stable directions"),
    ("degenerate_node", "Degenerate node", "Captured monotone decay"),
    ("inward_spiral", "Inward spiral", "Captured spiral decay"),
    ("harmonic_oscillator", "Harmonic oscillator", "Captured periodic orbit"),
]

system_colors = {
    "Saddle point": "C0",
    "Degenerate node": "C1",
    "Inward spiral": "C2",
    "Harmonic oscillator": "C3",
}

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
    DMD eigenvalue rollout:
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


def reconstruct_dmd_operator(Lambda, Phi):
    """
    Reconstruct the learned one-step DMD operator from modes/eigenvalues.

        A_dmd = Phi diag(Lambda) Phi^+

    This is the discrete-time map learned by the DMD baseline.
    """
    Lambda = np.asarray(Lambda, dtype=np.complex128)
    Phi = np.asarray(Phi, dtype=np.complex128)

    A = Phi @ np.diag(Lambda) @ np.linalg.pinv(Phi)
    A = np.real_if_close(A, tol=1e9)

    if np.iscomplexobj(A):
        max_imag = float(np.max(np.abs(A.imag)))
        if max_imag > 1e-6:
            print(f"[warning] learned DMD operator has max imaginary part={max_imag:.3e}; taking real part.")
        A = A.real

    return np.asarray(A, dtype=float)


def true_discrete_linear_map(A_cont, dt, method="rk4"):
    """
    Compute the analytical one-step map corresponding to the data-generation
    integrator for a continuous linear system:

        dx/dt = A_cont x

    If the dataset was generated by RK4, then the exact discrete map used by
    one RK4 step is the fourth-order matrix polynomial:

        I + dt A + dt^2 A^2/2 + dt^3 A^3/6 + dt^4 A^4/24

    This is better than comparing against exp(A dt) if the data were generated
    with RK4 rather than an exact matrix exponential.
    """
    A = np.asarray(A_cont, dtype=float)
    I = np.eye(A.shape[0])
    method = str(method).lower()

    if method == "euler":
        return I + dt * A

    if method == "rk4":
        A2 = A @ A
        A3 = A2 @ A
        A4 = A3 @ A
        return I + dt * A + (dt ** 2 / 2.0) * A2 + (dt ** 3 / 6.0) * A3 + (dt ** 4 / 24.0) * A4

    # Safe fallback. For these systems you are probably using RK4 anyway.
    try:
        from scipy.linalg import expm
        return expm(A * dt)
    except Exception:
        print(f"[warning] Unknown method={method}; falling back to Euler approximation.")
        return I + dt * A


def choose_representative_trajectories(X, n_trajs):
    """
    Pick representative trajectories instead of just the first random ones.
    We spread choices across initial radius so each panel shows the structure.
    """
    X = ensure_3d(X)
    x0 = X[0, :, :]

    radii = np.linalg.norm(x0[:, :2], axis=1)
    order = np.argsort(radii)

    n_available = X.shape[1]
    n_pick = min(n_trajs, n_available)

    if n_pick <= 1:
        return [int(order[-1])]

    positions = np.linspace(0, n_available - 1, n_pick, dtype=int)
    return [int(order[p]) for p in positions]


def set_common_square_limits(ax, trajectories, pad_frac=0.08):
    """
    Make each subplot square-looking and avoid weird panel resizing.
    Limits are based on the trajectories shown in that panel.
    """
    pts = []

    for traj in trajectories:
        traj = np.asarray(traj)
        if traj.ndim == 2 and traj.shape[1] >= 2:
            finite = np.isfinite(traj[:, :2]).all(axis=1)
            if np.any(finite):
                pts.append(traj[finite, :2])

    if not pts:
        ax.set_box_aspect(1)
        return

    pts = np.vstack(pts)

    x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
    y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    half_width = 0.5 * max(x_max - x_min, y_max - y_min)
    half_width = max(half_width, 1e-8)
    half_width *= 1.0 + pad_frac

    ax.set_xlim(cx - half_width, cx + half_width)
    ax.set_ylim(cy - half_width, cy + half_width)
    ax.set_box_aspect(1)


def choose_representative_trajectories(X, n_trajs):
    """
    Pick diverse initial conditions using farthest-point selection in x0-space.
    This avoids simply taking the first random trajectories.
    """
    X = ensure_3d(X)
    x0 = np.asarray(X[0, :, :2], dtype=float)

    finite = np.isfinite(x0).all(axis=1)
    valid_ids = np.where(finite)[0]
    x0_valid = x0[finite]

    if len(valid_ids) == 0:
        return [0]

    n_pick = min(n_trajs, len(valid_ids))

    # Start from the largest-radius initial condition.
    radii = np.linalg.norm(x0_valid, axis=1)
    first = int(np.argmax(radii))

    chosen_local = [first]

    while len(chosen_local) < n_pick:
        chosen_pts = x0_valid[chosen_local]
        dist_to_chosen = np.min(
            np.linalg.norm(x0_valid[:, None, :] - chosen_pts[None, :, :], axis=2),
            axis=1,
        )
        dist_to_chosen[chosen_local] = -np.inf
        chosen_local.append(int(np.argmax(dist_to_chosen)))

    return [int(valid_ids[i]) for i in chosen_local]


def set_square_limits_from_trajs(ax, trajectories, pad_frac=0.08):
    pts = []

    for traj in trajectories:
        traj = np.asarray(traj)
        if traj.ndim == 2 and traj.shape[1] >= 2:
            finite = np.isfinite(traj[:, :2]).all(axis=1)
            if np.any(finite):
                pts.append(traj[finite, :2])

    if not pts:
        ax.set_box_aspect(1)
        return

    pts = np.vstack(pts)

    x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
    y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    half_width = 0.5 * max(x_max - x_min, y_max - y_min)
    half_width = max(half_width, 1e-8)
    half_width *= 1.0 + pad_frac

    ax.set_xlim(cx - half_width, cx + half_width)
    ax.set_ylim(cy - half_width, cy + half_width)
    ax.set_box_aspect(1)

def mean_trajectory_rollout_rmse_over_split(predictor, X, steps):
    """
    Mean per-trajectory rollout RMSE over the loaded split.

    For each trajectory i:
        r_i(1:h) = sqrt(mean over steps 1..h and state dimensions of error^2)

    Then:
        MeanTrajRMSE(1:h) = mean_i r_i(1:h)

    This excludes t=0 because the model starts from the true initial condition.
    """
    X = ensure_3d(X)
    steps = min(int(steps), X.shape[0] - 1)

    per_traj_rmse = []

    for traj_id in range(X.shape[1]):
        true = X[: steps + 1, traj_id, :]
        pred = rollout_predictor(predictor, true[0], steps)

        n = min(len(true), len(pred))

        # Exclude t=0 since initial condition is identical.
        diff = pred[1:n] - true[1:n]

        rmse_i = float(np.sqrt(np.mean(diff * diff)))
        per_traj_rmse.append(rmse_i)

    per_traj_rmse = np.asarray(per_traj_rmse, dtype=float)

    return {
        "mean": float(np.mean(per_traj_rmse)),
        "std": float(np.std(per_traj_rmse)),
        "median": float(np.median(per_traj_rmse)),
        "q25": float(np.percentile(per_traj_rmse, 25)),
        "q75": float(np.percentile(per_traj_rmse, 75)),
        "per_traj": per_traj_rmse,
    }

def make_figure1_rollouts(args):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.5, 8.2),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.ravel()

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, panel, (system, label, _) in zip(axes, panel_labels, SYSTEMS):
        data_path = Path(args.data_root) / system
        X, _ = load_split(data_path, args.split)
        predictor = load_predictor(args, system)

        steps = min(args.steps, X.shape[0] - 1)
        traj_ids = choose_representative_trajectories(X, args.n_trajs)

        plotted = []

        for k, traj_id in enumerate(traj_ids):
            x0 = X[0, traj_id, :]
            true = X[: steps + 1, traj_id, :]
            pred = rollout_predictor(predictor, x0, steps)

            plotted.extend([true, pred])

            ax.plot(
                true[:, 0],
                true[:, 1],
                color="C0",
                linewidth=1.8,
                alpha=0.85,
                label="True" if k == 0 else None,
            )
            ax.plot(
                pred[:, 0],
                pred[:, 1],
                "--",
                color="C1",
                linewidth=1.7,
                alpha=0.95,
                label=f"{predictor_label(args)} rollout" if k == 0 else None,
            )
            ax.scatter(
                true[0, 0],
                true[0, 1],
                s=22,
                color="black",
                alpha=0.55,
                zorder=5,
                label="Initial state" if k == 0 else None,
            )

        rmse_stats = mean_trajectory_rollout_rmse_over_split(
            predictor,
            X,
            steps,
        )

        ax.set_title(
            f"{panel} {label}\n"
            rf"Mean traj. RMSE$_{{1:{steps}}}$ = {rmse_stats['mean']:.2e}",
            fontsize=11,
        )
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.grid(True, alpha=0.22)
        set_square_limits_from_trajs(ax, plotted)

    axes[0].legend(loc="best", framealpha=0.95, fontsize=9)
    fig.suptitle("NN-DMD rollouts on linear systems", fontsize=14)

    out_path = Path(args.outdir) / "figure1_linear_dmd_rollouts.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1: {out_path}")

def reconstruct_dmd_operator(Lambda, Phi):
    Lambda = np.asarray(Lambda, dtype=np.complex128)
    Phi = np.asarray(Phi, dtype=np.complex128)

    A = Phi @ np.diag(Lambda) @ np.linalg.pinv(Phi)
    A = np.real_if_close(A, tol=1e9)

    if np.iscomplexobj(A):
        max_imag = float(np.max(np.abs(A.imag)))
        if max_imag > 1e-6:
            print(f"[warning] learned DMD operator has max imaginary part={max_imag:.3e}; taking real part.")
        A = A.real

    return np.asarray(A, dtype=float)


def true_discrete_linear_map(A_cont, dt, method="rk4"):
    A = np.asarray(A_cont, dtype=float)
    I = np.eye(A.shape[0])
    method = str(method).lower()

    if method == "euler":
        return I + dt * A

    if method == "rk4":
        A2 = A @ A
        A3 = A2 @ A
        A4 = A3 @ A
        return (
            I
            + dt * A
            + (dt ** 2 / 2.0) * A2
            + (dt ** 3 / 6.0) * A3
            + (dt ** 4 / 24.0) * A4
        )

    try:
        from scipy.linalg import expm
        return expm(A * dt)
    except Exception:
        print(f"[warning] Unknown method={method}; falling back to Euler approximation.")
        return I + dt * A


def discrete_to_continuous_eigs(lam, dt):
    """
    Convert discrete-time eigenvalues to continuous-time eigenvalues.

        mu = log(lambda) / dt

    For these small-dt linear systems this makes growth rates and frequencies
    visible instead of clustering everything near lambda = 1.
    """
    lam = np.asarray(lam, dtype=np.complex128)
    return np.log(lam) / dt


def match_eigenvalues(eig_true, eig_learned):
    """
    Greedy nearest-neighbour matching for tiny 2D spectra.
    """
    remaining = list(range(len(eig_learned)))
    errors = []

    for lam_true in eig_true:
        j = min(remaining, key=lambda idx: abs(eig_learned[idx] - lam_true))
        errors.append(abs(eig_learned[j] - lam_true))
        remaining.remove(j)

    return np.asarray(errors, dtype=float)

def match_eigendecomposition(A_true_dt, A_learned_dt, dt):
    """
    Compare learned modal structure against the analytical modal structure.

    Returns:
        phi_rel_err:
            Relative eigenvector/mode error after matching eigenvalues and
            fixing arbitrary complex phase/sign.

        lambda_rel_err:
            Relative continuous-time eigenvalue error after matching.

    Important:
        This compares the eigendecomposition of the recovered operator K/A,
        not the raw trainable NN-DMD Phi/Lambda parameters.

        That is intentional: Phi/Lambda are not uniquely identifiable because
        columns can be permuted/scaled/phase-shifted while representing the
        same operator.
    """
    A_true_dt = np.asarray(A_true_dt, dtype=np.complex128)
    A_learned_dt = np.asarray(A_learned_dt, dtype=np.complex128)

    lam_true_dt, Phi_true = np.linalg.eig(A_true_dt)
    lam_learned_dt, Phi_learned = np.linalg.eig(A_learned_dt)

    eig_true = lam_true_dt
    eig_learned = lam_learned_dt

    # ------------------------------------------------------------
    # Match learned modes to true modes by eigenvalue distance.
    # For your 2D systems, brute-force permutations are simplest.
    # ------------------------------------------------------------
    n = len(eig_true)

    import itertools

    best_perm = None
    best_cost = np.inf

    for perm in itertools.permutations(range(n)):
        cost = np.sum(np.abs(eig_learned[list(perm)] - eig_true))
        if cost < best_cost:
            best_cost = cost
            best_perm = perm

    best_perm = list(best_perm)

    eig_learned = eig_learned[best_perm]
    Phi_learned = Phi_learned[:, best_perm]

    # ------------------------------------------------------------
    # Eigenvalue / Lambda error
    # ------------------------------------------------------------
    lambda_rel_err = np.linalg.norm(eig_learned - eig_true) / max(
        np.linalg.norm(eig_true),
        1e-12,
    )

    # ------------------------------------------------------------
    # Eigenvector / Phi error
    #
    # Eigenvectors have arbitrary scale and complex phase/sign:
    #     v and c*v represent the same eigenvector.
    #
    # Therefore:
    #   1. normalize each vector,
    #   2. rotate learned vector phase to best match true vector.
    # ------------------------------------------------------------
    def normalize_columns(V):
        V = np.asarray(V, dtype=np.complex128).copy()
        norms = np.linalg.norm(V, axis=0)
        norms = np.maximum(norms, 1e-12)
        return V / norms[None, :]

    Phi_true_n = normalize_columns(Phi_true)
    Phi_learned_n = normalize_columns(Phi_learned)

    for j in range(n):
        # Choose complex phase/sign that best aligns learned vector to true vector.
        phase = np.vdot(Phi_learned_n[:, j], Phi_true_n[:, j])
        if abs(phase) > 1e-12:
            Phi_learned_n[:, j] *= phase / abs(phase)

    phi_rel_err = np.linalg.norm(Phi_learned_n - Phi_true_n, ord="fro") / max(
        np.linalg.norm(Phi_true_n, ord="fro"),
        1e-12,
    )

    return float(phi_rel_err), float(lambda_rel_err)

def make_figure2_operator_spectral_recovery(args):
    """
    Figure 2:
    DMD recovers the linear solution.

    Panel (a):
        Relative Frobenius error between analytical one-step map and learned DMD map.

    Panel (b):
        Discrete-time eigenvalues, true vs learned.
        This is more readable than discrete eigenvalues because discrete eigenvalues
        cluster near lambda = 1 when dt is small.
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.8),
        gridspec_kw={"width_ratios": [1.45, 1.10]},
        constrained_layout=True,
    )

    ax_err, ax_spec = axes

    labels = []
    op_errors = []
    phi_errors = []
    lambda_errors = []
    diagnostics_rows = []

    for system, label, _ in SYSTEMS:
        data_path = Path(args.data_root) / system
        _, data = load_split(data_path, args.split)
        predictor = load_predictor(args, system)

        A_dmd = predictor_one_step_operator(predictor)

        if "A" not in data.files:
            raise KeyError(
                f"Dataset for {system} does not contain matrix A. "
                "Cannot compute analytical linear reference."
            )

        A_cont = np.asarray(data["A"], dtype=float)
        dt = float(np.asarray(data["dt"]).item()) if "dt" in data.files else 0.01
        method = str(np.asarray(data["method"]).item()) if "method" in data.files else "rk4"

        A_true_dt = true_discrete_linear_map(A_cont, dt, method=method)

        rel_op_err = np.linalg.norm(A_dmd - A_true_dt, ord="fro") / max(
            np.linalg.norm(A_true_dt, ord="fro"),
            1e-12,
        )

        lambda_true = np.linalg.eigvals(A_true_dt)
        lambda_learned = np.linalg.eigvals(A_dmd)

        # Discrete eigenvalue errors for reporting/debugging.
        lambda_abs_errors = match_eigenvalues(lambda_true, lambda_learned)
        max_lambda_abs_err = float(np.max(lambda_abs_errors))

        phi_rel_err, lambda_rel_err = match_eigendecomposition(
            A_true_dt,
            A_dmd,
            dt,
        )

        labels.append(label)
        op_errors.append(rel_op_err)
        phi_errors.append(phi_rel_err)
        lambda_errors.append(lambda_rel_err)

        diagnostics_rows.append({
            "system": system,
            "label": label,
            "operator_rel_fro_error": rel_op_err,
            "phi_mode_rel_error": phi_rel_err,
            "lambda_discrete_rel_error": lambda_rel_err,
            "max_lambda_abs_err": max_lambda_abs_err,
        })

        print(
            f"{label:20s} | "
            f"A_err={rel_op_err:.3e} | "
            f"Phi_err={phi_rel_err:.3e} | "
            f"Lambda_err={lambda_rel_err:.3e} | "
            f"abs_lambda_err={max_lambda_abs_err:.3e}"
        )

        color = system_colors[label]

        # Hollow circle = analytical discrete eigenvalues
        ax_spec.scatter(
            lambda_true.real,
            lambda_true.imag,
            marker="o",
            s=110,
            facecolors="none",
            edgecolors=color,
            linewidths=1.8,
            alpha=0.95,
            zorder=3,
        )

        # Cross = learned model discrete eigenvalues
        ax_spec.scatter(
            lambda_learned.real,
            lambda_learned.imag,
            marker="x",
            s=110,
            linewidths=2.2,
            color=color,
            alpha=0.95,
            zorder=4,
        )

    # ------------------------------------------------------------
    # Panel A: A / Phi / Lambda recovery errors
    # ------------------------------------------------------------
    x = np.arange(len(labels))
    width = 0.24

    op_errors_arr = np.asarray(op_errors, dtype=float)
    phi_errors_arr = np.asarray(phi_errors, dtype=float)
    lambda_errors_arr = np.asarray(lambda_errors, dtype=float)

    bars_A = ax_err.bar(
        x - width,
        op_errors_arr,
        width,
        label=r"$A$ / $K$ operator",
    )

    bars_Phi = ax_err.bar(
        x,
        phi_errors_arr,
        width,
        label=r"$\Phi$ modes",
    )

    bars_Lambda = ax_err.bar(
        x + width,
        lambda_errors_arr,
        width,
        label=r"$\Lambda$ eigenvalues",
    )

    ax_err.set_yscale("log")
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(labels, rotation=20, ha="right")
    ax_err.set_ylabel("Relative recovery error")
    ax_err.set_title(r"(a) Relative recovery of $A$, $\Phi$, and $\Lambda$")
    ax_err.grid(True, axis="y", alpha=0.25)
    ax_err.legend(framealpha=0.95, fontsize=8, ncols=1)

    # # Numerical labels above bars.
    # def annotate_bars(bars):
    #     for bar in bars:
    #         value = bar.get_height()
    #         if not np.isfinite(value) or value <= 0:
    #             continue
    #         ax_err.text(
    #             bar.get_x() + bar.get_width() / 2,
    #             value * 1.08,
    #             f"{value:.1e}",
    #             ha="center",
    #             va="bottom",
    #             fontsize=7,
    #             rotation=90,
    #         )

    # annotate_bars(bars_A)
    # annotate_bars(bars_Phi)
    # annotate_bars(bars_Lambda)

    all_errs = np.concatenate([op_errors_arr, phi_errors_arr, lambda_errors_arr])
    all_errs = all_errs[np.isfinite(all_errs) & (all_errs > 0)]

    if all_errs.size > 0:
        ax_err.set_ylim(
            max(np.min(all_errs) * 0.5, 1e-12),
            np.max(all_errs) * 20.0,
        )

    # Show y-axis ticks as actual scientific numbers, not as a scaling in the axis label.
    def sci_tick(y, _):
        if abs(y) < 1e-15:
            return "0"
        return f"{y:.1e}"

    ax_err.yaxis.set_major_formatter(FuncFormatter(sci_tick))

    # Panel B: discrete-time spectrum

    theta = np.linspace(0, 2 * np.pi, 400)
    ax_spec.plot(
        np.cos(theta),
        np.sin(theta),
        "--",
        color="gray",
        linewidth=1.2,
        alpha=0.75,
        label="Unit circle",
    )

    ax_spec.axhline(0.0, color="black", linewidth=0.7)
    ax_spec.axvline(0.0, color="black", linewidth=0.7)

    ax_spec.set_xlabel(r"Re($\lambda$)")
    ax_spec.set_ylabel(r"Im($\lambda$)")
    ax_spec.set_title("(b) Modal spectrum recovery")
    ax_spec.grid(True, alpha=0.25)
    ax_spec.set_aspect("equal", adjustable="box")
    ax_spec.set_xlim(0.958, 1.022)
    ax_spec.set_ylim(-0.035, 0.035)
    # Legend 1: marker meaning
    meaning_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.6,
            markersize=8,
            label="Analytical",
        ),
        Line2D(
            [0], [0],
            marker="x",
            linestyle="None",
            color="black",
            markeredgewidth=2.0,
            markersize=8,
            label=f"Learned model",
        ),
    ]

    legend_meaning = ax_spec.legend(
        handles=meaning_handles,
        loc="upper right",
        framealpha=0.95,
        fontsize=8,
        title="Marker",
    )
    ax_spec.add_artist(legend_meaning)

    # Legend 2: system identity
    system_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=system_colors[label],
            markeredgecolor=system_colors[label],
            markersize=8,
            label=label,
        )
        for label in system_colors
    ]

    ax_spec.legend(
        handles=system_handles,
        loc="lower left",
        bbox_to_anchor=(0.03, 0.03),
        borderaxespad=0.0,
        framealpha=0.95,
        fontsize=8,
        title="System",
    )

    fig.suptitle(
        "NN-DMD recovers the linear operator and modal structure",
        fontsize=16,
    )

    out_path = Path(args.outdir) / "figure2_linear_dmd_operator_spectral_recovery.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 2: {out_path}")

    csv_path = Path(args.outdir) / "figure2_operator_modal_errors.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "system,label,"
            "operator_relative_frobenius_error,"
            "phi_mode_relative_error,"
            "lambda_discrete_relative_error\n"
        )

        for system_tuple, label, op_err, phi_err, lambda_err in zip(
            SYSTEMS,
            labels,
            op_errors,
            phi_errors,
            lambda_errors,
        ):
            system = system_tuple[0]
            f.write(
                f"{system},{label},"
                f"{op_err:.8e},"
                f"{phi_err:.8e},"
                f"{lambda_err:.8e}\n"
            )

    print(f"Saved Figure 2 diagnostics: {csv_path}")

def scalar(x):
    arr = np.asarray(x)
    return float(arr.reshape(-1)[0])


def make_table1_metrics(args):
    rows = []

    for system, label, comment in SYSTEMS:
        summary_path = Path(args.figure_root) / system / args.run_name / f"{args.split}_summary.npz"

        if not summary_path.exists():
            raise FileNotFoundError(
                f"Missing summary file: {summary_path}\n"
                "Run scripts.eval first for this system/run_name."
            )

        summary = np.load(summary_path, allow_pickle=True)

        one_step_rmse = scalar(summary["one_step_rmse"])

        rollout_horizons = np.asarray(summary["rollout_horizons"], dtype=int)
        rollout_rmse = np.asarray(summary["rollout_rmse"], dtype=float)

        if args.rollout_table_horizon in rollout_horizons:
            idx = int(np.where(rollout_horizons == args.rollout_table_horizon)[0][0])
            rollout_value = float(rollout_rmse[idx])
            rollout_name = f"Mean traj. RMSE, $1:{args.rollout_table_horizon}$"
        else:
            rollout_value = float(np.mean(rollout_rmse))
            rollout_name = "Mean traj. rollout RMSE"

        rows.append({
            "System": label,
            "One-step RMSE": one_step_rmse,
            rollout_name: rollout_value,
            "Qualitative behaviour": comment,
        })

    csv_path = Path(args.outdir) / "table1_linear_dmd_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rollout_candidates = [
        k for k in rows[0].keys()
        if (
            k.startswith("Rollout")
            or k.startswith("Mean rollout")
            or k.startswith("Mean traj.")
        )
    ]

    if len(rollout_candidates) == 0:
        raise KeyError(
            f"Could not identify rollout metric column. Available columns: {list(rows[0].keys())}"
        )

    rollout_col = rollout_candidates[0]

    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\begin{tabular}{lccc}")
    latex_lines.append("\\hline")
    latex_lines.append(f"System & One-step RMSE & {rollout_col} & Qualitative behaviour \\\\")
    latex_lines.append("\\hline")

    for row in rows:
        latex_lines.append(
            f"{row['System']} & "
            f"{row['One-step RMSE']:.2e} & "
            f"{row[rollout_col]:.2e} & "
            f"{row['Qualitative behaviour']} \\\\"
        )

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{DMD baseline performance on linear systems.}")
    latex_lines.append("\\label{tab:dmd_linear_systems}")
    latex_lines.append("\\end{table}")

    tex_path = Path(args.outdir) / "table1_linear_dmd_metrics.tex"
    tex_path.write_text("\n".join(latex_lines), encoding="utf-8")

    print(f"Saved Table 1 CSV  : {csv_path}")
    print(f"Saved Table 1 LaTeX: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Make Figure 1, Figure 2, and Table 1 for the linear DMD baseline section.")

    parser.add_argument("--data_root", type=str, default="data/trajectories/linear")
    parser.add_argument("--model_root", type=str, default="data/models/dmd_baseline")
    parser.add_argument("--figure_root", type=str, default="data/figures/dmd_baseline")
    parser.add_argument("--outdir", type=str, default="data/figures/dmd_baseline/linear_section")
    parser.add_argument("--run_name", type=str, default="fullrank")
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--n_trajs", type=int, default=10)
    parser.add_argument("--rollout_table_horizon", type=int, default=50)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--model_type",
        type=str,
        default="ml_dmd",
        choices=["dmd_baseline", "ml_dmd"],
        help="Which model type to plot.",
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    make_figure1_rollouts(args)
    make_figure2_operator_spectral_recovery(args)
    make_table1_metrics(args)

    print("\nDone.")
    print(f"Outputs written to: {args.outdir}")


if __name__ == "__main__":
    main()
