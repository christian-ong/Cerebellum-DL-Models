"""Modal noise projection experiment.

Generates white noise in the latent modal space, filters it to isolate 
the governing modes (parallel) vs junk modes (orthogonal), projects it 
back to physical space, and applies fixed-magnitude physical shoves.
"""

from __future__ import annotations
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_generation.load_data import resolve_split_npz_path
from src.eval.model_io import load_model, predict_rollout_from_x0

def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

def _get_expanded_indices(mode_indices, lambdas):
    """Ensures complex conjugate pairs are kept together."""
    expanded_idx = set(mode_indices)
    if np.iscomplexobj(lambdas):
        for i in mode_indices:
            if i < len(lambdas) and abs(lambdas[i].imag) > 1e-6:
                diffs = np.abs(lambdas - lambdas[i].conj())
                diffs[i] = np.inf
                conj_idx = int(np.argmin(diffs))
                if diffs[conj_idx] < 1e-4:
                    expanded_idx.add(conj_idx)
    return sorted(list(expanded_idx))

def _get_b_and_phi(model_name, model, x0_full):
    """Lifts a physical state into Koopman modal coordinates (b)."""
    if model_name == "regression_dmd":
        Phi = model.Phi_state_fitted.to(torch.complex128).detach().cpu().numpy()
        W = np.linalg.pinv(Phi)
        return W @ x0_full, Phi
    elif model_name in {"ml_dmd", "ml_dmd_drop"}:
        dev = next(model.parameters()).device
        x_t = torch.as_tensor(x0_full, dtype=torch.float32, device=dev).unsqueeze(0)
        z = model._normalize(model.expander.expand(x_t))
        Phi = model.Phi
        i_eps = 1e-6 * torch.eye(model.latent_dim, device=Phi.device, dtype=Phi.dtype)
        W = torch.linalg.solve(Phi + i_eps, torch.eye(model.latent_dim, device=Phi.device, dtype=Phi.dtype)).transpose(-2, -1)
        b = (W @ z.T).T
        return b.detach().cpu().numpy().squeeze(0), Phi.detach().cpu().numpy()
    raise ValueError(f"Unsupported model for modal projection: {model_name}")

def _recon_x(model_name, model, b, Phi):
    """Maps modal coordinates (b) back down to a physical state."""
    if model_name == "regression_dmd":
        return np.real(Phi @ b)
    elif model_name in {"ml_dmd", "ml_dmd_drop"}:
        dev = next(model.parameters()).device
        b_t = torch.as_tensor(b, dtype=torch.complex64, device=dev).unsqueeze(0)
        Phi_t = torch.as_tensor(Phi, dtype=torch.complex64, device=dev)
        z_recon = (Phi_t @ b_t.T).T
        x_recon = model.expander.de_expand(model._unnormalize(z_recon))
        return x_recon.detach().cpu().numpy().squeeze(0).real
    raise ValueError("Unsupported model")

def run_experiment(*, model_name: str, model, extras: dict, X: np.ndarray, traj_index: int, noise_scales: list[float], steps: int, outdir: str, seed: int):
    rng = np.random.default_rng(seed)
    state_dim = X.shape[-1]
    delay_depth = int(getattr(model, "delay_depth", getattr(getattr(model, "expander", model), "delay_depth", 1)))
    x0_full = np.asarray(X[:delay_depth, traj_index, :], dtype=float).reshape(-1) if delay_depth > 1 else np.asarray(X[0, traj_index, :], dtype=float)

    clean_rollout = predict_rollout_from_x0(x0=x0_full, steps=steps, model_name=model_name, model=model, extras=extras)

    # 1. Project to modes
    b_base, Phi = _get_b_and_phi(model_name, model, x0_full)
    n_modes = len(b_base)

    # 2. Identify Governing vs Junk modes
    lambdas = model.get_eigenvalues().detach().cpu().numpy() if hasattr(model, "get_eigenvalues") else model.Lambda_fitted.detach().cpu().numpy()
    
    try:
        from src.eval.noise_robustness import compute_mode_diagnostics
        diag = compute_mode_diagnostics(model, X)
        top_mode = np.asarray(diag["order_contrib"], dtype=int)[0]
    except Exception:
        top_mode = np.argsort(np.linalg.norm(Phi[:state_dim, :], axis=0))[::-1][0]
    
    gov_modes = _get_expanded_indices([top_mode], lambdas)
    junk_modes = [i for i in range(n_modes) if i not in gov_modes]

    # 3. Generate White Noise strictly in the Governing Modes (Parallel)
    b_par = b_base.copy()
    for i in gov_modes: b_par[i] += (rng.standard_normal() + 1j * rng.standard_normal()) * 1e-3
    x_par_full = _recon_x(model_name, model, b_par, Phi)
    v_par = x_par_full[-state_dim:] - x0_full[-state_dim:]
    u_par = v_par / np.linalg.norm(v_par)

    # 4. Generate White Noise strictly in the Junk Modes (Orthogonal)
    if not junk_modes:
        print("\nWARNING: Model has no expansion dimensions! Cannot perform modal noise filtering. Defaulting to random orthogonal noise.\n")
        rand_vec = rng.standard_normal(state_dim)
        v_orth = rand_vec - np.dot(rand_vec, u_par) * u_par
        u_orth = v_orth / np.linalg.norm(v_orth)
    else:
        b_orth = b_base.copy()
        for i in junk_modes: b_orth[i] += (rng.standard_normal() + 1j * rng.standard_normal()) * 1e-3
        x_orth_full = _recon_x(model_name, model, b_orth, Phi)
        v_orth = x_orth_full[-state_dim:] - x0_full[-state_dim:]
        u_orth = v_orth / np.linalg.norm(v_orth)

    # 5. Apply strictly scaled physical shoves
    rows = []
    trajectories = {"parallel": [], "orthogonal": []}

    for ptype, direction_vec in [("parallel", u_par), ("orthogonal", u_orth)]:
        for scale in noise_scales:
            physical_jump = direction_vec * scale
            x0_pert = x0_full.copy()
            if delay_depth > 1: x0_pert[-state_dim:] += physical_jump
            else: x0_pert += physical_jump

            pert_rollout = predict_rollout_from_x0(x0=x0_pert, steps=steps, model_name=model_name, model=model, extras=extras)
            trajectories[ptype].append((scale, x0_pert[-state_dim:] if delay_depth > 1 else x0_pert, pert_rollout))
            rows.append({"perturbation_type": ptype, "noise_scale": scale, "terminal_rmse_vs_clean": _rmse(clean_rollout[-1], pert_rollout[-1])})

    # 6. Plotting
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    for ptype in ["orthogonal", "parallel"]:
        sub = [r for r in rows if r["perturbation_type"] == ptype]
        xs = [r["noise_scale"] for r in sub]
        ax.plot(xs, [r["terminal_rmse_vs_clean"] for r in sub], marker="o", linewidth=2, label=f"{ptype} noise")
    ax.set_xlabel("Physical Noise Scale")
    ax.set_ylabel("Terminal RMSE vs clean rollout")
    ax.set_title("Modal Projection Noise Robustness")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "modal_noise_rmse.png"), dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("Rollouts Under Modal Projected Perturbations", fontsize=16, y=0.95)
    for ax, ptype in zip(axes, ["orthogonal", "parallel"]):
        ax.plot(clean_rollout[:, 0], clean_rollout[:, 1], 'k-', linewidth=3, label="Clean Rollout", zorder=10)
        ax.scatter([clean_rollout[0, 0]], [clean_rollout[0, 1]], color='k', s=60, zorder=15)
        colors = plt.cm.plasma(np.linspace(0.8, 0.1, len(trajectories[ptype])))
        for (scale, x0_p, traj), color in zip(trajectories[ptype], colors):
            ax.plot(traj[:, 0], traj[:, 1], '--', color=color, linewidth=1.5, alpha=0.8, label=f"Noise: {scale}")
            ax.scatter([x0_p[0]], [x0_p[1]], color=color, s=50, marker='X', zorder=12)
        ax.set_title(f"Projected {ptype.capitalize()} Noise", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc='best', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(outdir, "modal_noise_trajectories.png"), dpi=200)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--traj_index", type=int, default=0)
    parser.add_argument("--noise_scales", type=str, default="0.05,0.1,0.2,0.5")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="experiments/modal_noise")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split_path = resolve_split_npz_path(args.data_path, args.split)
    data = np.load(split_path, allow_pickle=True)
    X = data["X"] if data["X"].ndim == 3 else data["X"][:, None, :]

    model, extras = load_model(
        model_name=args.model, model_path=args.model_path, data_path=split_path, 
        state_dim=X.shape[-1], system=str(data["system"]), device=device
    )
    
    run_experiment(
        model_name=args.model, model=model, extras=extras, X=X, traj_index=args.traj_index,
        noise_scales=parse_float_list(args.noise_scales), steps=args.steps, outdir=args.outdir, seed=args.seed
    )
    print(f"✅ Modal projection complete. Plots saved to: {args.outdir}")

if __name__ == "__main__":
    main()