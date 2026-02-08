import numpy as np


def fit_linear_map(X: np.ndarray):
    """
    Fit a linear map M such that:
        y ≈ x @ M.T
    using least squares.

    Args:
        X: np.ndarray with shape:
           - (T, state_dim) OR
           - (T, n_traj, state_dim)

    Returns:
        M: np.ndarray (state_dim, state_dim)
    """
    if X.ndim == 2:
        # (T, state_dim)
        x = X[:-1]
        y = X[1:]
    elif X.ndim == 3:
        # (T, n_traj, state_dim)
        x = X[:-1].reshape(-1, X.shape[-1])
        y = X[1:].reshape(-1, X.shape[-1])
    else:
        raise ValueError(f"Expected X to have 2 or 3 dims, got shape {X.shape}")

    # Solve y = x @ M.T  ->  least squares for M.T
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
