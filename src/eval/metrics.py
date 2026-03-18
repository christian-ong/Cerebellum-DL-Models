import os
from typing import Dict, List, Optional

import numpy as np

from src.eval.model_io import predict_rollout_from_x0


EPS = 1e-12


def get_state_scale_from_train_split(data_path: str) -> Dict[str, np.ndarray]:
    """
    Compute state-wise scale statistics from the TRAIN trajectories only.
    """
    data = np.load(data_path)
    X = data["X"]

    if "train_idx" not in data:
        raise ValueError("Dataset must contain train_idx to compute normalization scales.")

    train_idx = data["train_idx"]
    X_train = X[:, train_idx, :]  # (T, Ntrain, d)

    flat = X_train.reshape(-1, X_train.shape[-1])

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
    mse_per_dim = np.mean(sq, axis=0)
    rmse_per_dim = np.sqrt(mse_per_dim)
    nrmse_per_dim = rmse_per_dim / np.maximum(scale, EPS)

    mse = float(np.mean(mse_per_dim))
    rmse = float(np.sqrt(mse))
    nrmse = float(np.sqrt(np.mean((rmse_per_dim / np.maximum(scale, EPS)) ** 2)))

    return {
        "mse": mse,
        "rmse": rmse,
        "nrmse": nrmse,
        "mse_per_dim": mse_per_dim,
        "rmse_per_dim": rmse_per_dim,
        "nrmse_per_dim": nrmse_per_dim,
    }


def compute_one_step_metrics(
    *,
    X: np.ndarray,
    traj_indices: np.ndarray,
    model_name: str,
    model,
    extras: Dict,
    scale_std: np.ndarray,
    max_pairs_per_traj: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Evaluate direct one-step prediction x_t -> x_{t+1} over selected trajectories.
    """
    T, _, d = X.shape
    err_list = []
    per_traj_mse = []

    for traj_id in traj_indices:
        X_traj = X[:, traj_id, :]
        n_pairs = T - 1
        starts = np.arange(n_pairs)

        if max_pairs_per_traj is not None and len(starts) > max_pairs_per_traj:
            keep = np.linspace(0, len(starts) - 1, max_pairs_per_traj, dtype=int)
            starts = starts[keep]

        traj_sq_err = []

        for t0 in starts:
            x0 = X_traj[t0]
            x1_true = X_traj[t0 + 1]

            rollout = predict_rollout_from_x0(
                x0=x0,
                steps=1,
                model_name=model_name,
                model=model,
                extras=extras,
            )
            x1_hat = rollout[1]

            err = x1_hat - x1_true
            err_list.append(err)
            traj_sq_err.append(np.mean(err ** 2))

        per_traj_mse.append(np.mean(traj_sq_err))

    errors = np.asarray(err_list)  # (N, d)
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
) -> Dict[str, np.ndarray]:
    """
    Terminal h-step metrics over selected trajectories and valid starting points.
    """
    T, _, d = X.shape
    max_h = max(horizons)

    per_h_errors = {h: [] for h in horizons}
    per_h_traj_mse = {h: [] for h in horizons}

    for traj_id in traj_indices:
        X_traj = X[:, traj_id, :]
        n_valid_starts = T - max_h
        if n_valid_starts <= 0:
            raise ValueError(f"Trajectory length {T} is too short for max horizon {max_h}.")

        starts = np.arange(0, n_valid_starts, start_stride)

        if max_starts_per_traj is not None and len(starts) > max_starts_per_traj:
            keep = np.linspace(0, len(starts) - 1, max_starts_per_traj, dtype=int)
            starts = starts[keep]

        traj_h_sq = {h: [] for h in horizons}

        for t0 in starts:
            x0 = X_traj[t0]
            rollout = predict_rollout_from_x0(
                x0=x0,
                steps=max_h,
                model_name=model_name,
                model=model,
                extras=extras,
            )

            for h in horizons:
                err = rollout[h] - X_traj[t0 + h]
                per_h_errors[h].append(err)
                traj_h_sq[h].append(np.mean(err ** 2))

        for h in horizons:
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
) -> Dict[str, np.ndarray]:
    """
    Full-rollout metrics from the first point of each selected trajectory.
    """
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

        for traj_id in traj_indices:
            X_true = X[: h + 1, traj_id, :]
            x0 = X_true[0]

            rollout = predict_rollout_from_x0(
                x0=x0,
                steps=h,
                model_name=model_name,
                model=model,
                extras=extras,
            )

            err = rollout - X_true
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
    one_step_nrmse: float,
    horizon_nrmse: np.ndarray,
    rollout_nrmse: np.ndarray,
    one_step_weight: float = 0.35,
    horizon_weight: float = 0.40,
    rollout_weight: float = 0.25,
) -> float:
    """
    Lower is better.
    """
    return float(
        one_step_weight * one_step_nrmse
        + horizon_weight * float(np.mean(horizon_nrmse))
        + rollout_weight * float(np.mean(rollout_nrmse))
    )


def save_summary_npz(path: str, payload: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **payload)