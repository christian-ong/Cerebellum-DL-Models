import os
from typing import Dict, List, Optional

import numpy as np
import warnings
from src.data_generation.load_data import resolve_split_npz_path

from src.eval.model_io import predict_rollout_from_x0
from src.eval.delay_utils import (
    get_model_delay_depth,
    delay_start_index,
    make_rollout_initial_condition,
    valid_start_indices,
)

"""
Shared metric functions for dynamical-system evaluation.

Metric meaning:
- One-step metrics:
  Start from many valid states x_t and compare the predicted next state to the true next state x_{t+1}.
  These measure local one-step prediction quality.

- Horizon metrics:
  For a chosen horizon h, start from many valid states x_t, roll the model forward h steps,
  and compare only the final predicted state x_{t+h} to the true x_{t+h}.
  Example: horizon-5 RMSE means the error at the 5th predicted point only, not the sum of errors from steps 1 to 5.

- Full-rollout metrics:
  Start from x_0 of each selected trajectory, roll forward up to horizon h,
  and compare the full predicted rollout [x_0, ..., x_h] to the full true rollout [x_0, ..., x_h].
  These measure accumulated forecasting quality over the whole rollout.

- NRMSE:
  RMSE normalized by the state-wise standard deviation computed from the training split.
  This makes errors more comparable across state dimensions and across models evaluated on the same dataset.

- Composite score:
  Weighted combination of one-step NRMSE, mean horizon NRMSE, and mean rollout NRMSE.
  Lower is better.
"""

EPS = 1e-12


def get_state_scale_from_train_split(data_path: str) -> Dict[str, np.ndarray]:
    """
    Compute state-wise scale statistics from the TRAIN trajectories only.
    
    data_path should be the base path (without _train, _val, _test suffix).
    The function will load from {data_path}_train.npz
    """
    train_data_path = resolve_split_npz_path(data_path, "train")
    data = np.load(train_data_path)
    X = data["X"]

    # In the new format, X contains only the train trajectories
    # All trajectories in this file are training data
    flat = X.reshape(-1, X.shape[-1])  # (T*N, d)

    std = np.std(flat, axis=0)
    data_min = np.min(flat, axis=0)
    data_max = np.max(flat, axis=0)
    value_range = data_max - data_min

    std_safe = np.maximum(std, EPS)
    range_safe = np.maximum(value_range, EPS)

    return {
        "std": std_safe,
        "range": range_safe,
        "min": data_min,
        "max": data_max,
    }


def _mse_rmse_nrmse_from_errors(errors: np.ndarray, scale: np.ndarray) -> Dict[str, np.ndarray]:
    """
    errors shape: (..., d)
    scale shape: (d,)
    """
    sq = errors ** 2
    if sq.size == 0:
        d = scale.shape[0]
        mse_per_dim = np.full((d,), np.nan)
        rmse_per_dim = np.full((d,), np.nan)
        nrmse_per_dim = np.full((d,), np.nan)
        mse = float(np.nan)
        rmse = float(np.nan)
        nrmse = float(np.nan)
        return {
            "mse": mse,
            "rmse": rmse,
            "nrmse": nrmse,
            "mse_per_dim": mse_per_dim,
            "rmse_per_dim": rmse_per_dim,
            "nrmse_per_dim": nrmse_per_dim,
        }

    # Use nan-safe means to avoid warnings when slices are empty
    mse_per_dim = np.nanmean(sq, axis=0)
    rmse_per_dim = np.sqrt(mse_per_dim)
    nrmse_per_dim = rmse_per_dim / np.maximum(scale, EPS)

    mse = float(np.nanmean(mse_per_dim))
    rmse = float(np.sqrt(mse))
    nrmse = float(np.sqrt(np.nanmean((rmse_per_dim / np.maximum(scale, EPS)) ** 2)))

    return {
        "mse": mse,
        "rmse": rmse,
        "nrmse": nrmse,
        "mse_per_dim": mse_per_dim,
        "rmse_per_dim": rmse_per_dim,
        "nrmse_per_dim": nrmse_per_dim,
    }

def build_rollout_cache(
    *,
    X: np.ndarray,
    traj_indices: np.ndarray,
    model_name: str,
    model,
    extras: Dict,
    max_horizon: int,
    start_stride: int = 1,
    max_starts_per_traj: Optional[int] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Re-initialized rollout cache.

    For ordinary models:
        x0 = X_traj[t0]

    For delay models:
        x0 = [x(t0), x(t0-1), ..., x(t0-delay_depth+1)]

    This prevents invalid fake histories such as [x(t0), x(t0), ..., x(t0)].
    """
    T, _, _ = X.shape
    cache = {}

    for traj_id in traj_indices:
        X_traj = X[:, traj_id, :]

        starts = valid_start_indices(
            T=T,
            horizon=max_horizon,
            model_name=model_name,
            model=model,
            start_stride=start_stride,
            max_starts_per_traj=max_starts_per_traj,
        )

        if len(starts) == 0:
            raise ValueError(
                f"Trajectory length T={T} is too short for max_horizon={max_horizon} "
                f"and delay_depth={get_model_delay_depth(model_name, model)}."
            )

        rollouts = []

        for t0 in starts:
            x0 = make_rollout_initial_condition(
                X_traj=X_traj,
                t0=int(t0),
                model_name=model_name,
                model=model,
            )

            rollout = predict_rollout_from_x0(
                x0=x0,
                steps=max_horizon,
                model_name=model_name,
                model=model,
                extras=extras,
            )

            # Skip rollouts that produce NaN/Inf values to avoid downstream overflows
            if not np.all(np.isfinite(rollout)):
                warnings.warn(
                    f"Skipping rollout starting at t0={t0} for traj_id={traj_id} due to NaN/Inf in predictions",
                    RuntimeWarning,
                )
                continue

            rollouts.append(rollout)

        cache[traj_id] = {
            "starts": starts,
            "rollouts": rollouts,
        }

    return cache


def compute_one_step_metrics(
    *,
    X: np.ndarray,
    traj_indices: np.ndarray,
    model_name: str,
    model,
    extras: Dict,
    scale_std: np.ndarray,
    max_pairs_per_traj: Optional[int] = None,
    rollout_cache: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Evaluate one-step prediction.

    For delay models, the first valid t0 is delay_depth - 1.
    """
    T, _, _ = X.shape
    err_list = []
    per_traj_mse = []

    for traj_id in traj_indices:
        X_traj = X[:, traj_id, :]
        traj_sq_err = []

        if rollout_cache is not None and traj_id in rollout_cache:
            starts = rollout_cache[traj_id]["starts"]
            rollouts = rollout_cache[traj_id]["rollouts"]

            for t0, rollout in zip(starts, rollouts):
                if t0 + 1 >= X_traj.shape[0]:
                    continue

                err = rollout[1] - X_traj[t0 + 1]
                err_list.append(err)
                traj_sq_err.append(np.mean(err ** 2))

        else:
            starts = valid_start_indices(
                T=T,
                horizon=1,
                model_name=model_name,
                model=model,
                start_stride=1,
                max_starts_per_traj=max_pairs_per_traj,
            )

            for t0 in starts:
                x0 = make_rollout_initial_condition(
                    X_traj=X_traj,
                    t0=int(t0),
                    model_name=model_name,
                    model=model,
                )

                rollout = predict_rollout_from_x0(
                    x0=x0,
                    steps=1,
                    model_name=model_name,
                    model=model,
                    extras=extras,
                )

                # Skip one-step rollouts that produce NaN/Inf
                if not np.all(np.isfinite(rollout)):
                    warnings.warn(
                        f"Skipping one-step rollout at t0={t0} for traj_id={traj_id} due to NaN/Inf in predictions",
                        RuntimeWarning,
                    )
                    continue

                err = rollout[1] - X_traj[t0 + 1]
                err_list.append(err)
                traj_sq_err.append(np.mean(err ** 2))

        if len(traj_sq_err) > 0:
            per_traj_mse.append(np.mean(traj_sq_err))

    if len(err_list) == 0:
        raise ValueError("No valid one-step prediction errors were computed.")

    errors = np.asarray(err_list)
    stats = _mse_rmse_nrmse_from_errors(errors, scale_std)

    return {
        "one_step_mse": np.array(stats["mse"]),
        "one_step_rmse": np.array(stats["rmse"]),
        "one_step_nrmse": np.array(stats["nrmse"]),
        "one_step_mse_per_dim": stats["mse_per_dim"],
        "one_step_rmse_per_dim": stats["rmse_per_dim"],
        "one_step_nrmse_per_dim": stats["nrmse_per_dim"],
        "one_step_traj_mse_mean": np.array(float(np.mean(per_traj_mse))),
        "one_step_traj_mse_std": np.array(float(np.std(per_traj_mse))),
    }


def compute_horizon_metrics(
    *,
    X: np.ndarray,
    traj_indices: np.ndarray,
    horizons: List[int],
    model_name: str,
    model,
    extras: Dict,
    scale_std: np.ndarray,
    start_stride: int = 1,
    max_starts_per_traj: Optional[int] = None,
    rollout_cache: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Terminal h-step metrics over selected trajectories and valid starting points.

    For delay models, starts begin at delay_depth - 1.
    """
    T, _, _ = X.shape
    max_h = max(horizons)

    per_h_errors = {h: [] for h in horizons}
    per_h_traj_mse = {h: [] for h in horizons}

    for traj_id in traj_indices:
        X_traj = X[:, traj_id, :]
        traj_h_sq = {h: [] for h in horizons}

        if rollout_cache is not None and traj_id in rollout_cache:
            starts = rollout_cache[traj_id]["starts"]
            rollouts = rollout_cache[traj_id]["rollouts"]

            for t0, rollout in zip(starts, rollouts):
                for h in horizons:
                    if t0 + h >= X_traj.shape[0]:
                        continue

                    err = rollout[h] - X_traj[t0 + h]
                    per_h_errors[h].append(err)
                    traj_h_sq[h].append(np.mean(err ** 2))

        else:
            starts = valid_start_indices(
                T=T,
                horizon=max_h,
                model_name=model_name,
                model=model,
                start_stride=start_stride,
                max_starts_per_traj=max_starts_per_traj,
            )

            if len(starts) == 0:
                raise ValueError(
                    f"Trajectory length T={T} is too short for max horizon {max_h} "
                    f"and delay_depth={get_model_delay_depth(model_name, model)}."
                )

            for t0 in starts:
                x0 = make_rollout_initial_condition(
                    X_traj=X_traj,
                    t0=int(t0),
                    model_name=model_name,
                    model=model,
                )

                rollout = predict_rollout_from_x0(
                    x0=x0,
                    steps=max_h,
                    model_name=model_name,
                    model=model,
                    extras=extras,
                )

                # Skip rollouts that produce NaN/Inf values to avoid downstream overflows
                if not np.all(np.isfinite(rollout)):
                    warnings.warn(
                        f"Skipping rollout starting at t0={t0} for traj_id={traj_id} due to NaN/Inf in predictions",
                        RuntimeWarning,
                    )
                    continue

                for h in horizons:
                    err = rollout[h] - X_traj[t0 + h]
                    per_h_errors[h].append(err)
                    traj_h_sq[h].append(np.mean(err ** 2))

        for h in horizons:
            if len(traj_h_sq[h]) > 0:
                per_h_traj_mse[h].append(np.mean(traj_h_sq[h]))

    horizon_mse = []
    horizon_rmse = []
    horizon_nrmse = []
    horizon_q25_mse = []
    horizon_q50_mse = []
    horizon_q75_mse = []
    horizon_traj_mse_mean = []
    horizon_traj_mse_std = []

    for h in horizons:
        if len(per_h_errors[h]) == 0:
            raise ValueError(f"No valid horizon errors computed for h={h}.")

        errors = np.asarray(per_h_errors[h])
        stats = _mse_rmse_nrmse_from_errors(errors, scale_std)

        horizon_mse.append(stats["mse"])
        horizon_rmse.append(stats["rmse"])
        horizon_nrmse.append(stats["nrmse"])

        all_sq = np.mean(errors ** 2, axis=1)
        horizon_q25_mse.append(float(np.quantile(all_sq, 0.25)))
        horizon_q50_mse.append(float(np.quantile(all_sq, 0.50)))
        horizon_q75_mse.append(float(np.quantile(all_sq, 0.75)))

        horizon_traj_mse_mean.append(float(np.mean(per_h_traj_mse[h])))
        horizon_traj_mse_std.append(float(np.std(per_h_traj_mse[h])))

    return {
        "horizons": np.asarray(horizons, dtype=int),
        "horizon_mse": np.asarray(horizon_mse, dtype=float),
        "horizon_rmse": np.asarray(horizon_rmse, dtype=float),
        "horizon_nrmse": np.asarray(horizon_nrmse, dtype=float),
        "horizon_q25_mse": np.asarray(horizon_q25_mse, dtype=float),
        "horizon_q50_mse": np.asarray(horizon_q50_mse, dtype=float),
        "horizon_q75_mse": np.asarray(horizon_q75_mse, dtype=float),
        "horizon_traj_mse_mean": np.asarray(horizon_traj_mse_mean, dtype=float),
        "horizon_traj_mse_std": np.asarray(horizon_traj_mse_std, dtype=float),
    }


def compute_full_rollout_metrics(
    *,
    X: np.ndarray,
    traj_indices: np.ndarray,
    rollout_horizons: List[int],
    model_name: str,
    model,
    extras: Dict,
    scale_std: np.ndarray,
    rollout_cache: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Full-rollout metrics.

    For ordinary models, rollout starts at t0=0.
    For delay models, rollout starts at t0=delay_depth-1.
    """
    T, _, _ = X.shape

    delay_depth = get_model_delay_depth(model_name, model)
    start0 = delay_start_index(delay_depth)

    out = {
        "rollout_horizons": np.asarray(rollout_horizons, dtype=int),
        "rollout_mse": [],
        "rollout_rmse": [],
        "rollout_nrmse": [],
        "rollout_traj_mse_mean": [],
        "rollout_traj_mse_std": [],
    }

    for h in rollout_horizons:
        traj_mse = []
        all_errors = []

        if start0 + h >= T:
            raise ValueError(
                f"Trajectory length T={T} is too short for rollout horizon h={h} "
                f"and delay_depth={delay_depth}."
            )

        for traj_id in traj_indices:
            X_traj = X[:, traj_id, :]

            rollout = None

            if rollout_cache is not None and traj_id in rollout_cache:
                starts = rollout_cache[traj_id]["starts"]
                rollouts = rollout_cache[traj_id]["rollouts"]

                match = np.where(starts == start0)[0]
                if len(match) > 0:
                    rollout = rollouts[int(match[0])][: h + 1]

            if rollout is None:
                x0 = make_rollout_initial_condition(
                    X_traj=X_traj,
                    t0=start0,
                    model_name=model_name,
                    model=model,
                )

                rollout = predict_rollout_from_x0(
                    x0=x0,
                    steps=h,
                    model_name=model_name,
                    model=model,
                    extras=extras,
                )

            # Skip rollouts that produce NaN/Inf values
            if not np.all(np.isfinite(rollout)):
                warnings.warn(
                    f"Skipping full-rollout for traj_id={traj_id} horizon={h} due to NaN/Inf in predictions",
                    RuntimeWarning,
                )
                continue

            X_true = X_traj[start0 : start0 + h + 1]

            err = rollout[1:] - X_true[1:]
            all_errors.append(err.reshape(-1, err.shape[-1]))
            traj_mse.append(np.mean(err ** 2))

        all_errors = np.vstack(all_errors)
        stats = _mse_rmse_nrmse_from_errors(all_errors, scale_std)

        out["rollout_mse"].append(stats["mse"])
        out["rollout_rmse"].append(stats["rmse"])
        out["rollout_nrmse"].append(stats["nrmse"])
        out["rollout_traj_mse_mean"].append(float(np.mean(traj_mse)))
        out["rollout_traj_mse_std"].append(float(np.std(traj_mse)))

    out["rollout_mse"] = np.asarray(out["rollout_mse"], dtype=float)
    out["rollout_rmse"] = np.asarray(out["rollout_rmse"], dtype=float)
    out["rollout_nrmse"] = np.asarray(out["rollout_nrmse"], dtype=float)
    out["rollout_traj_mse_mean"] = np.asarray(out["rollout_traj_mse_mean"], dtype=float)
    out["rollout_traj_mse_std"] = np.asarray(out["rollout_traj_mse_std"], dtype=float)

    return out


def compute_composite_validation_score(
    *,
    one_step_rmse: float,
    horizon_rmse: np.ndarray,
    rollout_rmse: np.ndarray,
    one_step_weight: float = 0.35,
    horizon_weight: float = 0.40,
    rollout_weight: float = 0.25,
) -> float:
    """
    RMSE-based composite score. Lower is better.
    """
    return float(
        one_step_weight * one_step_rmse
        + horizon_weight * float(np.mean(horizon_rmse))
        + rollout_weight * float(np.mean(rollout_rmse))
    )


def save_summary_npz(path: str, payload: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **payload)