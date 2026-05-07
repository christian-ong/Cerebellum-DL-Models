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
    "ml_lineardynamics",
    "ml_dmd_free",
    "ml_dmd_band",
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
    base_figdir = os.path.join("data", "figures", args.model, system, run_name)
    figdir = base_figdir if subdir is None else os.path.join(base_figdir, subdir)
    os.makedirs(figdir, exist_ok=True)

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
    return maybe_load_npz(get_core_summary_path(ctx), description="core summary")


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
        one_step_nrmse=float(one_step_metrics["one_step_nrmse"]),
        horizon_nrmse=np.asarray(horizon_metrics["horizon_nrmse"]),
        rollout_nrmse=np.asarray(rollout_metrics["rollout_nrmse"]),
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