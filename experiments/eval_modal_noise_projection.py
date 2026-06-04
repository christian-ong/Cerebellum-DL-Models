"""Modal noise projection experiment.

This is a focused experiment for DMD-style models. It perturbs an initial
condition along or orthogonal to a selected modal direction, rolls the model
forward, and measures how much the prediction changes relative to the clean
rollout.

The first intended use case is a 2D linear system or another small modal model
where the basis is easy to inspect.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data_generation.load_data import resolve_split_npz_path
from src.eval.model_io import infer_run_name, load_model, predict_rollout_from_x0


SUPPORTED_MODELS = {"ml_dmd", "regression_dmd"}
DEFAULT_SYSTEMS = ["inward_spiral", "closed_large", "vanderpol"]


def parse_float_list(text: str) -> List[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one float must be provided.")
    return values


def parse_int_list(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one integer must be provided.")
    return values


def _write_rows(csv_path: str, rows: Sequence[dict]) -> None:
    if not rows:
        return

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.exists(csv_path)
    fieldnames = list(rows[0].keys())

    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _resolve_model_checkpoint(model_name: str, system: str, run_name: str) -> Optional[str]:
    if model_name == "regression_dmd":
        file_candidates = ["model.npz"]
    else:
        file_candidates = ["model_best.pt", "model.pt", "model.npz"]

    base_dir = Path("data/models") / model_name / system / run_name
    for file_name in file_candidates:
        candidate = base_dir / file_name
        if candidate.exists():
            return str(candidate)
    return None


def _infer_metric_key(data_path: str) -> str:
    data_path = str(data_path)
    if "dt_0.05" in data_path:
        return "best_val_rollout_rmse_h20"
    return "best_val_rollout_rmse_h100"


def _load_best_wandb_runs(
    wandb_csvs: Sequence[str],
    *,
    models: Sequence[str],
    systems: Sequence[str],
) -> List[dict]:
    frames: List[pd.DataFrame] = []
    for wandb_csv in wandb_csvs:
        frame = pd.read_csv(wandb_csv, low_memory=False)
        if "model_name" not in frame.columns or "system_name" not in frame.columns:
            raise ValueError(f"wandb CSV {wandb_csv} must contain model_name and system_name columns")
        frames.append(frame)

    if not frames:
        return []

    df = pd.concat(frames, ignore_index=True, sort=False)

    selected_rows: List[dict] = []
    wanted_models = {str(m).strip() for m in models if str(m).strip()}
    wanted_systems = {str(s).strip() for s in systems if str(s).strip()}

    subset = df[df["model_name"].astype(str).isin(wanted_models) & df["system_name"].astype(str).isin(wanted_systems)].copy()
    if subset.empty:
        return []

    if "regression_rollout_mode" in subset.columns:
        regression_mask = subset["model_name"].astype(str) == "regression_dmd"
        mode_values = subset.get("regression_rollout_mode", subset.get("rollout_mode", None))
        if mode_values is not None:
            subset = subset[~regression_mask | mode_values.astype(str).isin({"DMD", "projected_DMD"})].copy()

    for (model_name, system_name), group in subset.groupby(["model_name", "system_name"], dropna=False):
        group = group.copy()
        metric_key = None

        if "data_path" in group.columns and len(group) > 0:
            metric_key = _infer_metric_key(str(group.iloc[0]["data_path"]))
        if metric_key is None or metric_key not in group.columns:
            for candidate in ["best_val_rollout_rmse_h100", "best_val_rollout_rmse_h20", "best_val_rollout_rmse_h10"]:
                if candidate in group.columns:
                    metric_key = candidate
                    break

        if metric_key is None or metric_key not in group.columns:
            continue

        scores = pd.to_numeric(group[metric_key], errors="coerce")
        if scores.notna().any():
            best_idx = scores.idxmin()
            selected_rows.append(group.loc[best_idx].to_dict())

    return selected_rows


def _load_split(data_path: str, split: str):
    split_path = resolve_split_npz_path(data_path, split)
    data = np.load(split_path, allow_pickle=True)
    X = data["X"]
    if X.ndim == 2:
        X = X[:, None, :]
    if X.ndim != 3:
        raise ValueError(f"Expected X to be 2D or 3D, got shape {X.shape}")
    return split_path, data, X


def _build_rollout_initial_state(X: np.ndarray, traj_index: int, model) -> np.ndarray:
    delay_depth = int(getattr(model, "delay_depth", getattr(getattr(model, "expander", model), "delay_depth", 1)))
    if delay_depth <= 1:
        return np.asarray(X[0, traj_index, :], dtype=float).reshape(-1)

    if X.shape[0] < delay_depth:
        raise ValueError(
            f"Trajectory length {X.shape[0]} is shorter than required delay_depth={delay_depth}."
        )

    history = np.asarray(X[:delay_depth, traj_index, :], dtype=float)
    return history.reshape(-1)


def _extract_basis(model_name: str, model, extras: dict) -> np.ndarray:
    if model_name == "ml_dmd":
        if hasattr(model, "Phi"):
            basis = model.Phi.detach().cpu().numpy()
        elif hasattr(model, "Phi_fitted"):
            basis = model.Phi_fitted.detach().cpu().numpy()
        elif hasattr(model, "get_Phi"):
            basis_obj = model.get_Phi()
            basis = basis_obj.detach().cpu().numpy() if hasattr(basis_obj, "detach") else np.asarray(basis_obj)
        else:
            raise ValueError("ml_dmd model does not expose a modal basis on the loaded model object.")
    elif model_name == "regression_dmd":
        if not hasattr(model, "Phi_state_fitted"):
            raise ValueError("regression_dmd model does not expose Phi_state_fitted.")
        basis = model.Phi_state_fitted.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported model for this experiment: {model_name}")

    basis = np.real_if_close(np.asarray(basis))
    if np.iscomplexobj(basis):
        basis = np.real(basis)

    if basis.ndim != 2:
        raise ValueError(f"Expected a 2D basis matrix, got shape {basis.shape}")
    return np.asarray(basis, dtype=float)


def _normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(vec, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def _orthogonal_noise(direction: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    u = _normalize(direction)
    candidate = rng.standard_normal(size=u.shape)
    candidate = candidate - np.dot(candidate, u) * u

    if np.linalg.norm(candidate) < 1e-10:
        candidate = np.roll(u, 1)
        candidate = candidate - np.dot(candidate, u) * u

    return scale * _normalize(candidate)


def _parallel_noise(direction: np.ndarray, scale: float) -> np.ndarray:
    return scale * _normalize(direction)


def _ml_dmd_modal_coords_safe(model, z_norm: torch.Tensor) -> torch.Tensor:
    i_eps = 1e-6 * torch.eye(model.latent_dim, device=model.Phi.device, dtype=model.Phi.dtype)
    return torch.linalg.solve(model.Phi + i_eps, z_norm.transpose(-2, -1)).transpose(-2, -1)


def _ml_dmd_modal_to_latent_safe(model, b: torch.Tensor) -> torch.Tensor:
    return b @ model.Phi.transpose(-2, -1)


def _ml_dmd_step_modal_safe(model, b: torch.Tensor) -> torch.Tensor:
    return b @ model.Lambda.transpose(-2, -1)


def _predict_rollout_safe(*, x0: np.ndarray, steps: int, model_name: str, model, extras: dict) -> np.ndarray:
    if model_name != "ml_dmd":
        return predict_rollout_from_x0(
            x0=x0,
            steps=steps,
            model_name=model_name,
            model=model,
            extras=extras,
        )

    with torch.inference_mode():
        dev = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
        x_t = torch.as_tensor(x0, dtype=torch.float32, device=dev)
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)

        delay_depth = int(getattr(model, "delay_depth", getattr(getattr(model, "expander", model), "delay_depth", 1)))
        expected_width = int(model.state_dim) * delay_depth
        if delay_depth > 1 and x_t.shape[1] != expected_width:
            raise ValueError(
                f"ml_dmd expected delay-state width {expected_width}, got {x_t.shape[1]}."
            )

        x_curr0 = x_t[:, : int(model.state_dim)] if delay_depth > 1 else x_t
        traj = [x_curr0.squeeze(0)]

        z_raw = model.expander.expand(x_t)
        z_norm = model._normalize(z_raw)
        b = _ml_dmd_modal_coords_safe(model, z_norm)

        for _ in range(int(steps)):
            b = _ml_dmd_step_modal_safe(model, b)
            z_next_norm = _ml_dmd_modal_to_latent_safe(model, b)
            z_next = model._unnormalize(z_next_norm)
            x_next = model.expander.de_expand(z_next)
            traj.append(x_next.squeeze(0))

        return torch.stack(traj).detach().cpu().numpy()


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _step_rmse(clean: np.ndarray, perturbed: np.ndarray) -> float:
    steps = min(clean.shape[0], perturbed.shape[0])
    if steps == 0:
        return float("nan")
    return _rmse(clean[:steps], perturbed[:steps])


def run_experiment(
    *,
    model_name: str,
    model,
    extras: dict,
    X: np.ndarray,
    traj_index: int,
    mode_indices: Sequence[int],
    noise_scales: Sequence[float],
    perturbation_types: Sequence[str],
    steps: int,
    seed: int,
    outdir: str,
    baseline_name: Optional[str] = None,
    baseline_model: Optional[object] = None,
    baseline_extras: Optional[dict] = None,
) -> List[dict]:
    basis = _extract_basis(model_name, model, extras)
    n_modes = basis.shape[1]

    if traj_index < 0 or traj_index >= X.shape[1]:
        raise IndexError(f"traj_index={traj_index} out of range for {X.shape[1]} trajectories")

    selected_modes = [int(i) for i in mode_indices if 0 <= int(i) < n_modes]
    if not selected_modes:
        raise ValueError(f"No valid mode indices provided. Available modes: 0..{n_modes - 1}")

    x0 = _build_rollout_initial_state(X, traj_index, model)
    clean_rollout = _predict_rollout_safe(
        x0=x0,
        steps=steps,
        model_name=model_name,
        model=model,
        extras=extras,
    )

    rng = np.random.default_rng(seed)
    rows: List[dict] = []

    def _make_coeff_delta(mode_index: int, perturbation_type: str, scale: float) -> np.ndarray:
        direction = np.zeros(n_modes, dtype=float)
        direction[mode_index] = 1.0
        if perturbation_type == "parallel":
            return _parallel_noise(direction, scale)
        return _orthogonal_noise(direction, scale, rng)

    def _reconstruct_state_from_coeff_delta(delta_coeff: np.ndarray) -> np.ndarray:
        delta_coeff = np.asarray(delta_coeff, dtype=float).reshape(1, -1)

        if model_name == "ml_dmd":
            dev = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
            x_t = torch.as_tensor(x0, dtype=torch.float32, device=dev).reshape(1, -1)
            coeff_t = torch.as_tensor(delta_coeff, dtype=torch.float32, device=dev)

            z_raw = model.expander.expand(x_t)
            z_norm = model._normalize(z_raw)
            b_clean = _ml_dmd_modal_coords_safe(model, z_norm)
            b_pert = b_clean + coeff_t
            z_norm_pert = _ml_dmd_modal_to_latent_safe(model, b_pert)
            z_pert = model._unnormalize(z_norm_pert)
            x_pert = model.expander.de_expand(z_pert)
            return x_pert.detach().cpu().numpy().reshape(-1)

        if model_name == "regression_dmd" and hasattr(model, "Phi_state_fitted"):
            Phi = model.Phi_state_fitted.to(torch.complex128)
            coeff_t = torch.as_tensor(delta_coeff, dtype=torch.complex128, device=Phi.device)
            x0_arr = np.asarray(x0, dtype=float).reshape(-1)
            x_state = x0_arr[: int(model.state_dim)]
            x_state_t = torch.as_tensor(x_state, dtype=torch.complex128, device=Phi.device).reshape(1, -1)
            b_clean = (torch.linalg.pinv(Phi) @ x_state_t.T).T
            b_pert = b_clean + coeff_t
            x_state_pert = (Phi @ b_pert.T).T.real.detach().cpu().numpy().reshape(-1)

            if x0_arr.size == int(model.state_dim):
                return x_state_pert

            x_out = x0_arr.copy()
            x_out[: int(model.state_dim)] = x_state_pert
            return x_out

        return np.asarray(x0, dtype=float).reshape(-1) + np.asarray(delta_coeff, dtype=float).reshape(-1)

    for mode_index in selected_modes:
        for perturbation_type in perturbation_types:
            perturbation_type = str(perturbation_type).strip().lower()
            if perturbation_type not in {"parallel", "orthogonal"}:
                raise ValueError(f"Unsupported perturbation_type: {perturbation_type}")

            for scale in noise_scales:
                scale = float(scale)
                delta_coeff = _make_coeff_delta(mode_index, perturbation_type, scale)
                x0_pert = _reconstruct_state_from_coeff_delta(delta_coeff)
                perturbed_rollout = _predict_rollout_safe(
                    x0=x0_pert,
                    steps=steps,
                    model_name=model_name,
                    model=model,
                    extras=extras,
                )

                rollout_rmse = _step_rmse(clean_rollout, perturbed_rollout)
                terminal_rmse = _rmse(clean_rollout[-1], perturbed_rollout[-1])

                row = {
                    "model_name": model_name,
                    "traj_index": traj_index,
                    "mode_index": int(mode_index),
                    "perturbation_type": perturbation_type,
                    "noise_scale": scale,
                    "rollout_rmse_vs_clean": rollout_rmse,
                    "terminal_rmse_vs_clean": terminal_rmse,
                    "clean_x0_norm": float(np.linalg.norm(x0)),
                    "perturbation_norm": float(np.linalg.norm(np.asarray(x0_pert, dtype=float) - np.asarray(x0, dtype=float))),
                    "perturbation_ratio": float(np.linalg.norm(np.asarray(x0_pert, dtype=float) - np.asarray(x0, dtype=float)) / max(np.linalg.norm(x0), 1e-12)),
                    "x0_clean": np.array2string(x0, separator=", "),
                    "x0_perturbed": np.array2string(x0_pert, separator=", "),
                }

                # If a baseline (non-modal) model is provided, compute its response
                if baseline_model is not None and baseline_name is not None:
                    try:
                        baseline_clean = _predict_rollout_safe(
                            x0=x0,
                            steps=steps,
                            model_name=baseline_name,
                            model=baseline_model,
                            extras=baseline_extras or {},
                        )
                    except Exception:
                        baseline_clean = None

                    try:
                        baseline_pert = _predict_rollout_safe(
                            x0=x0_pert,
                            steps=steps,
                            model_name=baseline_name,
                            model=baseline_model,
                            extras=baseline_extras or {},
                        )
                    except Exception:
                        baseline_pert = None

                    if baseline_clean is None or baseline_pert is None:
                        row.update({
                            "baseline_rollout_rmse_vs_clean": float("nan"),
                            "baseline_terminal_rmse_vs_clean": float("nan"),
                        })
                    else:
                        row.update({
                            "baseline_rollout_rmse_vs_clean": _step_rmse(baseline_clean, baseline_pert),
                            "baseline_terminal_rmse_vs_clean": _rmse(baseline_clean[-1], baseline_pert[-1]),
                        })

                rows.append(row)

    os.makedirs(outdir, exist_ok=True)

    if len(rows) > 0:
        plot_path = os.path.join(outdir, "modal_noise_projection.png")
        fig, ax = plt.subplots(figsize=(7.5, 4.8))

        baseline_present = any("baseline_terminal_rmse_vs_clean" in row for row in rows)
        for perturbation_type in sorted({row["perturbation_type"] for row in rows}):
            subset = [row for row in rows if row["perturbation_type"] == perturbation_type]
            xs = [row["noise_scale"] for row in subset]
            ys = [row["terminal_rmse_vs_clean"] for row in subset]
            ax.plot(xs, ys, marker="o", label=f"{perturbation_type} (modal)")
            if baseline_present:
                ys_b = [row.get("baseline_terminal_rmse_vs_clean", float("nan")) for row in subset]
                ax.plot(xs, ys_b, marker="x", linestyle="--", label=f"{perturbation_type} (baseline)")

        ax.set_xlabel("Noise scale")
        ax.set_ylabel("Terminal RMSE vs clean rollout")
        ax.set_title("Modal noise projection sensitivity")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=220)
        plt.close(fig)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a modal noise projection experiment.")
    parser.add_argument("--model", type=str, choices=sorted(SUPPORTED_MODELS), help="Single model to run. Omit when using --wandb_csv.")
    parser.add_argument("--model_path", type=str, default=None, help="Single checkpoint to run. Omit when using --wandb_csv.")
    parser.add_argument("--data_path", type=str, default=None, help="Base dataset path without split suffix. Required for single-run mode.")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--traj_index", type=int, default=0)
    parser.add_argument("--mode_indices", type=str, default="0,1", help="Comma-separated mode indices to probe.")
    parser.add_argument("--noise_scales", type=str, default="0.0,0.01,0.05,0.1")
    parser.add_argument(
        "--perturbation_types",
        type=str,
        default="parallel,orthogonal",
        help="Comma-separated perturbation types: parallel, orthogonal.",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--wandb_csv", type=str, default=None, help="Optional wandb CSV for selecting best runs automatically.")
    parser.add_argument(
        "--wandb_csvs",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of wandb CSVs to merge before selecting best runs.",
    )
    parser.add_argument("--systems", type=str, default=",".join(DEFAULT_SYSTEMS), help="Comma-separated systems to evaluate in wandb mode.")
    parser.add_argument("--models", type=str, default="regression_dmd", help="Comma-separated model families to select in wandb mode (ml_dmd is opt-in).")
    parser.add_argument("--baseline_model", type=str, default=None, help="Optional baseline model family name for single-run comparison (e.g. ml_linear_dynamics).")
    parser.add_argument("--baseline_model_path", type=str, default=None, help="Optional baseline checkpoint path for single-run comparison.")

    args = parser.parse_args()

    wandb_csvs: List[str] = []
    if args.wandb_csvs:
        wandb_csvs.extend([str(path) for path in args.wandb_csvs])
    if args.wandb_csv:
        wandb_csvs.append(str(args.wandb_csv))

    wandb_mode = len(wandb_csvs) > 0
    if not wandb_mode and (args.model is None or args.model_path is None):
        raise ValueError("Provide --model and --model_path for single-run mode, or --wandb_csv for batch selection mode.")
    if not wandb_mode and args.data_path is None:
        raise ValueError("Provide --data_path for single-run mode.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    selected_jobs: List[dict] = []
    if wandb_mode:
        selected_jobs = _load_best_wandb_runs(
            wandb_csvs,
            models=[item.strip() for item in args.models.split(",") if item.strip()],
            systems=[item.strip() for item in args.systems.split(",") if item.strip()],
        )
        if not selected_jobs:
            raise ValueError(f"No matching rows found in {wandb_csvs} for the requested systems and model families.")
        if args.baseline_model_path or args.baseline_model:
            print("Warning: baseline comparison is only supported in single-run mode (--wandb_csv skipped baseline).")
    else:
        selected_jobs = [
            {
                "model_name": args.model,
                "system_name": None,
                "model_path": args.model_path,
                "run_name": args.name or infer_run_name(args.model_path),
                "data_path": args.data_path,
            }
        ]

    all_rows: List[dict] = []
    for job in selected_jobs:
        model_name = str(job["model_name"])
        if model_name not in SUPPORTED_MODELS:
            continue

        model_path = job.get("model_path")
        if not model_path:
            system_name = str(job.get("system_name", ""))
            run_name = str(job.get("run_name", ""))
            model_path = _resolve_model_checkpoint(model_name, system_name, run_name)
            if model_path is None:
                print(f"Skipping {model_name}/{system_name}/{run_name}: checkpoint not found.")
                continue

        job_data_path = str(job.get("data_path") or args.data_path or "").strip()
        if not job_data_path:
            raise ValueError(f"No data_path available for {model_name}/{job.get('system_name', '')}/{job.get('run_name', '')}")

        split_path, data, X = _load_split(job_data_path, args.split)
        state_dim = int(X.shape[-1])
        system = str(data["system"])

        model, extras = load_model(
            model_name=model_name,
            model_path=model_path,
            data_path=split_path,
            state_dim=state_dim,
            system=system,
            device=device,
        )

        run_name = str(job.get("run_name") or infer_run_name(model_path))
        outdir = args.outdir or os.path.join(
            "experiments",
            "noise_projection",
            model_name,
            system,
            "modal_noise_projection",
            run_name,
            f"split_{args.split}",
        )

        # load optional baseline model (single-run only)
        baseline_name = None
        baseline_model = None
        baseline_extras = None
        if not wandb_mode and args.baseline_model_path:
            if not args.baseline_model:
                raise ValueError("Provide --baseline_model when using --baseline_model_path")
            baseline_name = args.baseline_model
            baseline_model, baseline_extras = load_model(
                model_name=baseline_name,
                model_path=args.baseline_model_path,
                data_path=split_path,
                state_dim=state_dim,
                system=system,
                device=device,
            )

        rows = run_experiment(
            model_name=model_name,
            model=model,
            extras=extras,
            X=X,
            traj_index=args.traj_index,
            mode_indices=parse_int_list(args.mode_indices),
            noise_scales=parse_float_list(args.noise_scales),
            perturbation_types=[item.strip() for item in args.perturbation_types.split(",") if item.strip()],
            steps=args.steps,
            seed=args.seed,
            outdir=outdir,
            baseline_name=baseline_name,
            baseline_model=baseline_model,
            baseline_extras=baseline_extras,
        )

        csv_path = args.csv_path or os.path.join(outdir, "summary.csv")
        extra = {
            "run_name": run_name,
            "system": system,
            "split": args.split,
            "traj_index": args.traj_index,
            "steps": args.steps,
            "model_path": model_path,
            "data_path": job_data_path,
            "seed": args.seed,
        }

        if rows:
            enriched_rows = []
            for row in rows:
                merged = dict(extra)
                merged.update(row)
                enriched_rows.append(merged)
            _write_rows(csv_path, enriched_rows)
            all_rows.extend(enriched_rows)

        print(f"Saved modal noise projection outputs to: {outdir}")
        print(f"Appended summary CSV      : {csv_path}")

    if wandb_mode and all_rows:
        combined_csv = args.csv_path or os.path.join(
            "experiments",
            "noise_projection",
            "summary.csv",
        )
        _write_rows(combined_csv, all_rows)
        print(f"Appended combined summary CSV : {combined_csv}")


if __name__ == "__main__":
    main()