import numpy as np
from src.data_generation.data_simulation import rk4_step

def get_model_delay_depth(model_name, model) -> int:
    """
    Return delay_depth for models that use delay embeddings.
    Defaults to 1 for ordinary non-delay models.
    """
    if model_name == "regression_dmd":
        return int(getattr(model, "delay_depth", 1))

    expander = getattr(model, "expander", None)
    return int(getattr(expander, "delay_depth", 1))


def delay_start_index(delay_depth: int) -> int:
    """
    First valid index where a full delay history exists.

    delay_depth=1 -> first valid index 0
    delay_depth=10 -> first valid index 9
    """
    return max(0, int(delay_depth) - 1)


def make_delay_x0_from_trajectory(
    X_traj: np.ndarray,
    t0: int,
    delay_depth: int,
) -> np.ndarray:
    """
    Build flattened delay initial condition:

        [x(t0), x(t0-1), ..., x(t0-delay_depth+1)]

    X_traj has shape (T, state_dim).

    Example for Lorenz with state_dim=3 and delay_depth=3:
        [x_t, y_t, z_t, x_{t-1}, y_{t-1}, z_{t-1}, x_{t-2}, y_{t-2}, z_{t-2}]
    """
    X_traj = np.asarray(X_traj)

    if X_traj.ndim != 2:
        raise ValueError(f"Expected X_traj with shape (T, state_dim), got {X_traj.shape}.")

    delay_depth = int(delay_depth)

    if delay_depth <= 1:
        return X_traj[t0]

    first_valid = delay_depth - 1
    if t0 < first_valid:
        raise ValueError(
            f"Cannot build delay state at t0={t0}. "
            f"Need t0 >= {first_valid} for delay_depth={delay_depth}."
        )

    idx = t0 - np.arange(delay_depth)
    return X_traj[idx].reshape(-1)


def make_rollout_initial_condition(
    X_traj: np.ndarray,
    t0: int,
    model_name,
    model,
) -> np.ndarray:
    delay_depth = get_model_delay_depth(model_name, model)
    return make_delay_x0_from_trajectory(
        X_traj=X_traj,
        t0=t0,
        delay_depth=delay_depth,
    )


def valid_start_indices(
    T: int,
    horizon: int,
    model_name,
    model,
    *,
    start_stride: int = 1,
    max_starts_per_traj=None,
) -> np.ndarray:
    """
    Valid t0 values satisfying:

        t0 >= delay_depth - 1
        t0 + horizon < T

    This prevents delay models from being evaluated before enough history exists.
    """
    delay_depth = get_model_delay_depth(model_name, model)
    start_min = delay_start_index(delay_depth)
    start_max_exclusive = T - int(horizon)

    if start_max_exclusive <= start_min:
        return np.array([], dtype=int)

    starts = np.arange(start_min, start_max_exclusive, int(start_stride), dtype=int)

    if max_starts_per_traj is not None and len(starts) > max_starts_per_traj:
        idx = np.linspace(0, len(starts) - 1, int(max_starts_per_traj)).astype(int)
        starts = starts[idx]

    return starts

def make_backward_delay_x0_from_current_states(
    *,
    current_states: np.ndarray,
    f_true,
    dt: float,
    delay_depth: int,
) -> np.ndarray:
    """
    Build physically consistent delay histories for dense-grid points.

    Given current states x_0, integrate the true ODE backwards to obtain

        [x_0, x_{-1}, x_{-2}, ..., x_{-q+1}]

    and flatten it into the delay-model input format.

    For state_dim=2 and delay_depth=200, output shape is (N, 400).
    """
    current_states = np.asarray(current_states, dtype=float)
    delay_depth = int(delay_depth)

    if delay_depth <= 1:
        return current_states

    if current_states.ndim != 2:
        raise ValueError(
            f"Expected current_states with shape (N, state_dim), got {current_states.shape}."
        )

    n_points, state_dim = current_states.shape

    history = np.empty((n_points, delay_depth, state_dim), dtype=float)
    history[:, 0, :] = current_states

    x = current_states.copy()

    for lag in range(1, delay_depth):
        # Integrate one step backwards using the same RK4 stepper as the simulations.
        x = rk4_step(f_true, x, -float(dt), t=-(lag - 1) * float(dt))
        history[:, lag, :] = x

    return history.reshape(n_points, delay_depth * state_dim)