import numpy as np

from src.eval.model_io import predict_rollout_from_x0
from src.eval.delay_utils import (
    get_model_delay_depth,
    delay_start_index,
    make_rollout_initial_condition,
)


def evaluate_rollouts(
    *,
    X,
    traj_indices,
    model_name,
    model,
    steps,
    extras,
):
    """
    Computes rollout MSE over a set of trajectories.

    For delay models, the rollout starts at t0 = delay_depth - 1,
    because this is the first point where a real delay history exists.
    """
    mse_list = []

    delay_depth = get_model_delay_depth(model_name, model)
    t0 = delay_start_index(delay_depth)

    for traj_id in traj_indices:
        X_traj = X[:, traj_id, :]
        T = X_traj.shape[0]

        if t0 >= T - 1:
            raise ValueError(
                f"Trajectory length T={T} is too short for delay_depth={delay_depth}."
            )

        steps_local = min(int(steps), T - t0 - 1)

        x_true = X_traj[t0 : t0 + steps_local + 1]

        x0 = make_rollout_initial_condition(
            X_traj=X_traj,
            t0=t0,
            model_name=model_name,
            model=model,
        )

        x_hat = predict_rollout_from_x0(
            x0=x0,
            steps=steps_local,
            model_name=model_name,
            model=model,
            extras=extras,
        )

        mse = np.mean((x_hat - x_true) ** 2)
        mse_list.append(mse)

    mse_list = np.asarray(mse_list, dtype=float)
    return float(np.mean(mse_list)), float(np.std(mse_list)), mse_list


def compute_single_rollout(
    *,
    X,
    traj_id,
    steps,
    model_name,
    model,
    extras,
):
    """
    Returns X_true and X_hat for plotting.

    For delay models, plotting starts at t0 = delay_depth - 1,
    not at t0 = 0.
    """
    X_traj = X[:, traj_id, :]
    T = X_traj.shape[0]

    delay_depth = get_model_delay_depth(model_name, model)
    t0 = delay_start_index(delay_depth)

    if t0 >= T - 1:
        raise ValueError(
            f"Trajectory length T={T} is too short for delay_depth={delay_depth}."
        )

    steps_local = min(int(steps), T - t0 - 1)

    x_true = X_traj[t0 : t0 + steps_local + 1]

    x0 = make_rollout_initial_condition(
        X_traj=X_traj,
        t0=t0,
        model_name=model_name,
        model=model,
    )

    x_hat = predict_rollout_from_x0(
        x0=x0,
        steps=steps_local,
        model_name=model_name,
        model=model,
        extras=extras,
    )

    return x_true, x_hat