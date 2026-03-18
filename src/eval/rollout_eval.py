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

# import numpy as np


# def evaluate_validation_rollouts(
#     X,
#     test_idx,
#     model_name,
#     model,
#     steps,
#     rollout_linear_map=None,
#     rollout_dmd_eig=None,
#     M=None,
#     Lambda=None,
#     Phi=None,
#     K=None,
#     C=None,
# ):
#     """
#     Computes rollout MSE over all test trajectories.
#     Returns:
#         mse_mean
#         mse_std
#     """

#     mse_list = []

#     for traj_id in test_idx:
#         X_true = X[:, traj_id, :]
#         steps_local = min(steps, X_true.shape[0] - 1)
#         X_true = X_true[: steps_local + 1]

#         x0 = X_true[0]

#         if model_name == "linear_baseline":
#             X_hat = rollout_linear_map(M, x0=x0, steps=steps_local)

#         elif model_name == "dmd_baseline":
#             X_hat = rollout_dmd_eig(Lambda, Phi, x0=x0, steps=steps_local)

#         elif model_name == "manual_expansion_manual_dmd":
#             X_hat = model.rollout(K=K, C=C, x0=x0, steps=steps_local).cpu().numpy()
        
#         elif model_name == "sindy_baseline":
#             X_hat = model.rollout(x0, steps=steps_local)

#         else:
#             X_hat = model.rollout(x0=x0, steps=steps_local).detach().cpu().numpy()

#         mse = np.mean((X_hat - X_true) ** 2)
#         mse_list.append(mse)

#     return np.mean(mse_list), np.std(mse_list)


# def compute_single_rollout(
#     X,
#     traj_id,
#     steps,
#     model_name,
#     model,
#     rollout_linear_map=None,
#     rollout_dmd_eig=None,
#     M=None,
#     Lambda=None,
#     Phi=None,
#     K=None,
#     C=None,
# ):
#     """
#     Returns X_true and X_hat for plotting.
#     """

#     X_true = X[:, traj_id, :]
#     steps = min(steps, X_true.shape[0] - 1)
#     X_true = X_true[: steps + 1]

#     x0 = X_true[0]

#     if model_name == "linear_baseline":
#         X_hat = rollout_linear_map(M, x0=x0, steps=steps)

#     elif model_name == "dmd_baseline":
#         X_hat = rollout_dmd_eig(Lambda, Phi, x0=x0, steps=steps)

#     elif model_name == "manual_expansion_manual_dmd":
#         X_hat = model.rollout(K=K, C=C, x0=x0, steps=steps).cpu().numpy()
            
#     elif model_name == "sindy_baseline":
#         X_hat = model.rollout(x0, steps=steps)
        
#     else:
#         X_hat = model.rollout(x0=x0, steps=steps).detach().cpu().numpy()

#     return X_true, X_hat