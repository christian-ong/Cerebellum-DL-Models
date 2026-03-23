import numpy as np


def fit_linear_map(X, Y):
    """
    Fit a linear map M such that:
        y ≈ x @ M.T

    Args:
        X: np.ndarray of shape (N, d) or (T, n_traj, d)
        Y: np.ndarray of same shape as X

    Returns:
        M: np.ndarray of shape (d, d)
    """
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have same shape, got {X.shape} and {Y.shape}")

    if X.ndim == 3:
        x = X.reshape(-1, X.shape[-1])
        y = Y.reshape(-1, Y.shape[-1])
    elif X.ndim == 2:
        x = X
        y = Y
    else:
        raise ValueError(f"Expected X to have 2 or 3 dims, got shape {X.shape}")

    MT, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    M = MT.T
    return M


def rollout_linear_map(M: np.ndarray, x0: np.ndarray, steps: int):
    """
    Rollout x_{t+1} = M x_t

    Args:
        M: (state_dim, state_dim)
        x0: (state_dim,)
        steps: int

    Returns:
        X_hat: (steps+1, state_dim)
    """
    x = x0.copy()
    X_hat = [x]

    for _ in range(steps):
        x = M @ x
        X_hat.append(x)

    return np.stack(X_hat, axis=0)
