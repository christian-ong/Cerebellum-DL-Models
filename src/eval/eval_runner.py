import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch

from src.eval.model_io import load_model, infer_run_name
from src.eval.metrics import (
    compute_one_step_metrics,
    compute_horizon_metrics,
    compute_full_rollout_metrics,
    compute_composite_validation_score,
    get_state_scale_from_train_split,
    save_summary_npz,
    build_rollout_cache,
)
from src.data_generation.load_data import resolve_split_npz_path


MODEL_CHOICES = [
    "linear_baseline",
    "dmd_baseline",
    "regression_dmd",
    "ml_dmd",
    "ml_dmd_drop",
    "ml_linear_dynamics",
    "ml_lineardynamics",
    "ml_dmd_free",
    "ml_dmd_band",
    "mlp_baseline",
    "sindy_baseline",
]


def parse_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one integer must be provided.")
    return sorted(set(values))


def ensure_3d_trajectories(X: np.ndarray) -> np.ndarray:
    if X.ndim == 2:
        return X[:, None, :]
    if X.ndim != 3:
        raise ValueError(f"Expected X to be 2D or 3D, got {X.ndim}D")
    return X


def load_split_data(data_path: str, split: str):
    split_data_path = resolve_split_npz_path(data_path, split)
    data = np.load(split_data_path, allow_pickle=True)
    X = ensure_3d_trajectories(data["X"])
    if X.shape[1] == 0:
        raise ValueError(f"No trajectories found in split '{split}'.")
    system = str(data["system"])
    state_dim = X.shape[-1]
    return split_data_path, data, X, system, state_dim


@dataclass
class EvalContext:
    args: Any
    device: str
    split: str
    split_data_path: str
    data: Any
    X: np.ndarray
    system: str
    state_dim: int
    traj_indices: np.ndarray
    model: Any
    extras: Dict[str, Any]
    run_name: str
    base_figdir: str
    figdir: str
    scales: Dict[str, np.ndarray]
    scale_std: np.ndarray
    rollout_cache: Optional[Dict[int, Dict[str, np.ndarray]]] = None


def prepare_eval_context(
    *,
    args,
    split: str,
    subdir: Optional[str] = None,
    need_cache: bool = False,
    max_horizon_for_cache: Optional[int] = None,
    save_run_metadata: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    split_data_path, data, X, system, state_dim = load_split_data(args.data_path, split)
    traj_indices = np.arange(X.shape[1])

    model, extras = load_model(
        model_name=args.model,
        model_path=args.model_path,
        data_path=split_data_path,
        state_dim=state_dim,
        system=system,
        device=device,
    )

    run_name = infer_run_name(args.model_path, explicit_name=getattr(args, "name", None))
    outdir_arg = getattr(args, "outdir", None)
    if outdir_arg:
        figdir = outdir_arg
        norm_outdir = os.path.normpath(outdir_arg)
        # Keep one canonical run-root even when scripts write into nested folders
        # like .../run_name/data (formerly 'eval') or .../run_name/behavior/final.
        if norm_outdir.endswith(os.path.join("behavior", "final")):
            base_figdir = os.path.dirname(os.path.dirname(norm_outdir))
        else:
            tail = os.path.basename(norm_outdir)
            if tail in {"data", "rollout", "noise_robustness", "modes", "behavior"}:
                base_figdir = os.path.dirname(norm_outdir)
            else:
                base_figdir = norm_outdir
    else:
        base_figdir = os.path.join(os.environ.get("EVAL_BASE_DIR", "data/figures"), args.model, system)
        expansion_type = getattr(model, "expansion_type", None)

        expansion_folder = str(expansion_type) if expansion_type is not None else "none"
        if expansion_type == "rbf":
            train_args = extras.get("train_args", {}) if isinstance(extras, dict) else {}
            bandwidth = getattr(model, "rbf_bandwidth_mode", None) or train_args.get("rbf_bandwidth_mode", None)
            bw = str(bandwidth).strip().lower() if bandwidth is not None and not (isinstance(bandwidth, float) and np.isnan(bandwidth)) else "global"
            expansion_folder = os.path.join("rbf", "global" if bw == "global" else "knn")
        if expansion_type in {"hankel", "hankel_svd"}:
            expansion_folder = "hankel_svd"

        base_figdir = os.path.join(base_figdir, expansion_folder)

        if args.model in {"ml_dmd", "ml_dmd_drop"}: # <-- CHANGE TO INCLUDE ml_dmd_drop
            l1_weight = None
            try:
                l1_weight = getattr(model, "l1_weight", None)
            except Exception:
                l1_weight = None
            if l1_weight is None and isinstance(extras, dict):
                l1_weight = extras.get("train_args", {}).get("l1_weight")
            if l1_weight is not None:
                try:
                    l1_value = float(l1_weight)
                    if l1_value == 0.0:
                        base_figdir = os.path.join(base_figdir, "l1_0.0")
                    else:
                        base_figdir = os.path.join(base_figdir, f"l1_{l1_value:.0e}")
                except Exception:
                    base_figdir = os.path.join(base_figdir, str(l1_weight))

        base_figdir = os.path.join(base_figdir, run_name)
        figdir = base_figdir if subdir is None else os.path.join(base_figdir, subdir)

    if outdir_arg:
        figdir = outdir_arg

    os.makedirs(figdir, exist_ok=True)
    os.makedirs(base_figdir, exist_ok=True)
    # Save training metadata in the final output directory chosen for this run.
    if save_run_metadata:
        try:
            # Delay import of torch in case environment is minimal; torch is already imported above.
            def _extract_train_args(model_path, extras):
                # 1) Check cached ckpt in extras (used for many torch models)
                if isinstance(extras, dict) and "ckpt" in extras and isinstance(extras["ckpt"], dict):
                    return extras["ckpt"].get("train_args", {}) or {}

                # 2) Try loading torch checkpoint if file looks like a torch checkpoint
                if os.path.exists(model_path):
                    try:
                        if model_path.endswith(".pt") or model_path.endswith(".pth"):
                            ck = torch.load(model_path, map_location="cpu")
                            if isinstance(ck, dict):
                                return ck.get("train_args", {}) or {}
                    except Exception:
                        pass

                    # 3) Try numpy checkpoint (baselines, SINDy, DMD)
                    try:
                        data = np.load(model_path, allow_pickle=True)
                        if "train_args" in data:
                            return dict(data["train_args"].item()) if hasattr(data["train_args"], "item") else data["train_args"]
                        # Some older checkpoints may store args as individual keys; gather common fields
                        keys = [k for k in data.files if k.startswith("exp") or k in {"alpha", "threshold", "poly_order", "rbf_centers"}]
                        if keys:
                            out = {k: (data[k].item() if getattr(data[k], "shape", ()) == () else data[k].tolist()) for k in keys}
                            return out
                    except Exception:
                        pass

                return {}

            train_args = _extract_train_args(args.model_path, extras)
            # Minimal payload with provenance
            payload = {
                "train_args": train_args,
                "model_path": args.model_path,
                "data_path": args.data_path,
                "model": args.model,
                "run_name": infer_run_name(args.model_path, explicit_name=getattr(args, "name", None)),
            }

            # Write JSON and plain-text versions
            try:
                json_path = os.path.join(figdir, "train_args.json")
                with open(json_path, "w", encoding="utf-8") as _f:
                    json.dump(payload, _f, indent=2)

                txt_path = os.path.join(figdir, "train_args.txt")
                with open(txt_path, "w", encoding="utf-8") as _f:
                    for k, v in payload.items():
                        _f.write(f"{k}: {v}\n")

                # Optional: save a small PNG summary if matplotlib is available
                try:
                    import matplotlib.pyplot as plt

                    lines = [f"{k}: {train_args[k]}" for k in sorted(train_args.keys())]
                    if not lines:
                        lines = ["(no train_args found)"]
                    # Scale height with line count so long train_args dumps do not clip at the bottom.
                    fig_height = min(max(3.2, 0.16 * len(lines) + 1.3), 12.0)
                    fig, ax = plt.subplots(figsize=(7.8, fig_height))
                    ax.axis("off")
                    ax.text(0.01, 0.99, "\n".join(lines), fontsize=8, family="monospace", va="top", clip_on=False)
                    fig.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.04)
                    png_path = os.path.join(figdir, "train_args.png")
                    fig.savefig(png_path, dpi=150)
                    plt.close(fig)
                except Exception:
                    # matplotlib not available or failed to render; ignore silently
                    pass
            except Exception as e:
                print(f"[eval_runner] Warning: failed to save train_args payload: {e}")
        except Exception:
            # Best-effort only; do not fail eval on metadata saving issues
            pass

    scales = get_state_scale_from_train_split(args.data_path)
    scale_std = scales["std"]

    rollout_cache = None
    if need_cache:
        if max_horizon_for_cache is None:
            raise ValueError("need_cache=True requires max_horizon_for_cache")
        metric_cap = None if getattr(args, "metric_cap", 0) == 0 else getattr(args, "metric_cap", 0)
        print(f"[eval_runner] Building in-memory rollout cache up to horizon {max_horizon_for_cache}...")
        rollout_cache = build_rollout_cache(
            X=X,
            traj_indices=traj_indices,
            model_name=args.model,
            model=model,
            extras=extras,
            max_horizon=max_horizon_for_cache,
            start_stride=1,
            max_starts_per_traj=metric_cap,
        )

    return EvalContext(
        args=args,
        device=device,
        split=split,
        split_data_path=split_data_path,
        data=data,
        X=X,
        system=system,
        state_dim=state_dim,
        traj_indices=traj_indices,
        model=model,
        extras=extras,
        run_name=run_name,
        base_figdir=base_figdir,
        figdir=figdir,
        scales=scales,
        scale_std=scale_std,
        rollout_cache=rollout_cache,
    )


def get_core_summary_path(ctx: EvalContext) -> str:
    return os.path.join(ctx.base_figdir, f"{ctx.split}_summary.npz")


def get_rollout_example_path(ctx: EvalContext, traj_index: int) -> str:
    return os.path.join(ctx.base_figdir, f"rollout_example_idx{traj_index}.npz")


def maybe_load_npz(path: str, *, description: str):
    if os.path.exists(path):
        print(f"[eval_runner] Found existing {description}: {path}")
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    print(f"[eval_runner] No existing {description} found at: {path}")
    return None


def maybe_load_core_summary(ctx: EvalContext):
    candidates = [
        get_core_summary_path(ctx),
        os.path.join(ctx.base_figdir, "data", f"{ctx.split}_summary.npz"),
        os.path.join(ctx.figdir, f"{ctx.split}_summary.npz"),
    ]

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        data = maybe_load_npz(path, description="core summary")
        if data is not None:
            return data
    return None


def maybe_load_rollout_example(ctx: EvalContext, traj_index: int):
    return maybe_load_npz(
        get_rollout_example_path(ctx, traj_index),
        description=f"rollout example for traj_index={traj_index}",
    )


def compute_core_metrics_bundle(
    ctx: EvalContext,
    *,
    horizons: List[int],
    rollout_horizons: List[int],
):
    max_needed = max(max(horizons), max(rollout_horizons))
    if ctx.X.shape[0] <= max_needed:
        raise ValueError(
            f"Trajectory length T={ctx.X.shape[0]} is too short for requested max horizon {max_needed}."
        )

    metric_cap = None if getattr(ctx.args, "metric_cap", 0) == 0 else getattr(ctx.args, "metric_cap", 0)

    rollout_cache = ctx.rollout_cache
    if rollout_cache is None and getattr(ctx.args, "use_cache", False):
        print(f"[eval_runner] Cache requested but not prebuilt; building now up to horizon {max_needed}...")
        rollout_cache = build_rollout_cache(
            X=ctx.X,
            traj_indices=ctx.traj_indices,
            model_name=ctx.args.model,
            model=ctx.model,
            extras=ctx.extras,
            max_horizon=max_needed,
            start_stride=1,
            max_starts_per_traj=metric_cap,
        )

    one_step_metrics = compute_one_step_metrics(
        X=ctx.X,
        traj_indices=ctx.traj_indices,
        model_name=ctx.args.model,
        model=ctx.model,
        extras=ctx.extras,
        scale_std=ctx.scale_std,
        max_pairs_per_traj=metric_cap,
        rollout_cache=rollout_cache,
    )

    horizon_metrics = compute_horizon_metrics(
        X=ctx.X,
        traj_indices=ctx.traj_indices,
        horizons=horizons,
        model_name=ctx.args.model,
        model=ctx.model,
        extras=ctx.extras,
        scale_std=ctx.scale_std,
        max_starts_per_traj=metric_cap,
        rollout_cache=rollout_cache,
    )

    rollout_metrics = compute_full_rollout_metrics(
        X=ctx.X,
        traj_indices=ctx.traj_indices,
        rollout_horizons=rollout_horizons,
        model_name=ctx.args.model,
        model=ctx.model,
        extras=ctx.extras,
        scale_std=ctx.scale_std,
        rollout_cache=rollout_cache,
    )

    composite = compute_composite_validation_score(
        one_step_rmse=float(one_step_metrics["one_step_rmse"]),
        horizon_rmse=np.asarray(horizon_metrics["horizon_rmse"]),
        rollout_rmse=np.asarray(rollout_metrics["rollout_rmse"]),
    )

    summary = {}
    summary.update(one_step_metrics)
    summary.update(horizon_metrics)
    summary.update(rollout_metrics)
    summary["composite_score"] = np.array(composite)

    return summary, rollout_cache


def save_summary(summary: Dict[str, np.ndarray], out_path: str):
    save_summary_npz(out_path, summary)


def save_metadata_json(ctx: EvalContext, out_path: str, extra: Optional[Dict[str, Any]] = None):
    payload = {
        "model": ctx.args.model,
        "system": ctx.system,
        "run_name": ctx.run_name,
        "data_path": ctx.args.data_path,
        "model_path": ctx.args.model_path,
        "split": ctx.split,
    }
    if extra:
        payload.update(extra)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)