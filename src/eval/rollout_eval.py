import numpy as np

from src.eval.model_io import predict_rollout_from_x0


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

    Returns
    -------
    mse_mean : float
    mse_std : float
    mse_list : np.ndarray
    """
    mse_list = []

    for traj_id in traj_indices:
        x_true = X[:, traj_id, :]
        steps_local = min(steps, x_true.shape[0] - 1)
        x_true = x_true[: steps_local + 1]

        x0 = x_true[0]
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
    """
    x_true = X[:, traj_id, :]
    steps_local = min(steps, x_true.shape[0] - 1)
    x_true = x_true[: steps_local + 1]

    x0 = x_true[0]
    x_hat = predict_rollout_from_x0(
        x0=x0,
        steps=steps_local,
        model_name=model_name,
        model=model,
        extras=extras,
    )

    return x_true, x_hat

