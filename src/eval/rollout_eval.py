import numpy as np


def evaluate_validation_rollouts(
    X,
    val_idx,
    model_name,
    model,
    steps,
    rollout_linear_map=None,
    rollout_dmd_eig=None,
    M=None,
    Lambda=None,
    Phi=None,
    K=None,
    C=None,
):
    """
    Computes rollout MSE over all validation trajectories.
    Returns:
        mse_mean
        mse_std
    """

    mse_list = []

    for traj_id in val_idx:
        X_true = X[:, traj_id, :]
        steps_local = min(steps, X_true.shape[0] - 1)
        X_true = X_true[: steps_local + 1]

        x0 = X_true[0]

        if model_name == "linear_baseline":
            X_hat = rollout_linear_map(M, x0=x0, steps=steps_local)

        elif model_name == "dmd_baseline":
            X_hat = rollout_dmd_eig(Lambda, Phi, x0=x0, steps=steps_local)

        elif model_name == "manual_expansion_manual_dmd":
            X_hat = model.rollout(K=K, C=C, x0=x0, steps=steps_local).cpu().numpy()
        
        elif model_name == "sindy_baseline":
            X_hat = model.rollout(x0, steps=steps_local)

        else:
            X_hat = model.rollout(x0=x0, steps=steps_local).detach().cpu().numpy()

        mse = np.mean((X_hat - X_true) ** 2)
        mse_list.append(mse)

    return np.mean(mse_list), np.std(mse_list)


def compute_single_rollout(
    X,
    traj_id,
    steps,
    model_name,
    model,
    rollout_linear_map=None,
    rollout_dmd_eig=None,
    M=None,
    Lambda=None,
    Phi=None,
    K=None,
    C=None,
):
    """
    Returns X_true and X_hat for plotting.
    """

    X_true = X[:, traj_id, :]
    steps = min(steps, X_true.shape[0] - 1)
    X_true = X_true[: steps + 1]

    x0 = X_true[0]

    if model_name == "linear_baseline":
        X_hat = rollout_linear_map(M, x0=x0, steps=steps)

    elif model_name == "dmd_baseline":
        X_hat = rollout_dmd_eig(Lambda, Phi, x0=x0, steps=steps)

    elif model_name == "manual_expansion_manual_dmd":
        X_hat = model.rollout(K=K, C=C, x0=x0, steps=steps).cpu().numpy()
            
    elif model_name == "sindy_baseline":
        X_hat = model.rollout(x0, steps=steps)
        
    else:
        X_hat = model.rollout(x0=x0, steps=steps).detach().cpu().numpy()

    return X_true, X_hat