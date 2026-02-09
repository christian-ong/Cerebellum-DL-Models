import numpy as np
from typing import List, Optional, Tuple


def snapshot_mats(traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return X, Y with columns as snapshots: Y ~= A X."""
    if traj.ndim == 2:  # (T+1, d)
        X = traj[:-1].T
        Y = traj[1:].T
        return X, Y
    if traj.ndim == 3:  # (T+1, N, d)
        t1, n_traj, d = traj.shape
        X = traj[:-1].reshape((t1 - 1) * n_traj, d).T
        Y = traj[1:].reshape((t1 - 1) * n_traj, d).T
        return X, Y
    raise ValueError(f"traj must be (T+1,d) or (T+1,N,d), got {traj.shape}")


def fit_linear_map(
    X: np.ndarray,
    Y: np.ndarray,
    rank: Optional[int] = None,
    ridge: float = 0.0,
) -> np.ndarray:
    """
    Fit the best linear map M such that Y ~= M X using an SVD pseudoinverse.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    r = len(s) if rank is None else max(1, min(rank, len(s)))
    Ur, sr, Vtr = U[:, :r], s[:r], Vt[:r, :]

    inv = (sr / (sr ** 2 + ridge)) if ridge > 0 else (1.0 / sr)
    M = (Y @ (Vtr.T * inv)) @ Ur.T
    return M


# -------- EDMD dictionary: polynomial features up to given degree --------

def _exponents_upto_degree(d: int, degree: int) -> List[Tuple[int, ...]]:
    """All exponent tuples (e1,...,ed) with sum(ei) <= degree."""
    exps: List[Tuple[int, ...]] = []

    def rec(pos: int, remaining: int, current: List[int]) -> None:
        if pos == d:
            exps.append(tuple(current))
            return
        for k in range(remaining + 1):
            current.append(k)
            rec(pos + 1, remaining - k, current)
            current.pop()

    rec(0, degree, [])
    return exps


def poly_features(X: np.ndarray, degree: int) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """
    Polynomial dictionary psi(x) with all monomials up to 'degree'.
    X: (d, m) columns are samples
    Returns PsiX: (k, m) and exponent list.
    """
    d, m = X.shape
    exps = _exponents_upto_degree(d, degree)  # includes constant term (0,...,0)
    k = len(exps)
    Psi = np.empty((k, m), dtype=float)

    for i, e in enumerate(exps):
        ee = np.array(e, dtype=int)[:, None]
        Psi[i] = np.prod(X ** ee, axis=0)

    return Psi, exps


def fit_dmd(
    traj: np.ndarray,
    rank: Optional[int] = None,
    ridge: float = 0.0,
) -> np.ndarray:
    """Fit DMD map A with Y ~= A X."""
    X, Y = snapshot_mats(traj)
    A = fit_linear_map(X, Y, rank=rank, ridge=ridge)
    return A


def fit_edmd(
    traj: np.ndarray,
    degree: int,
    rank: Optional[int] = None,
    ridge: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit EDMD operator K and linear decoder C."""
    X, Y = snapshot_mats(traj)
    PsiX, _ = poly_features(X, degree=degree)
    PsiY, _ = poly_features(Y, degree=degree)

    K = fit_linear_map(PsiX, PsiY, rank=rank, ridge=ridge)
    C = fit_linear_map(PsiX, X, rank=rank, ridge=ridge)
    return K, C


def rollout_dmd(A: np.ndarray, x0: np.ndarray, steps: int) -> np.ndarray:
    """Roll out x_{t+1} = A x_t. Returns (steps+1, d)."""
    d = x0.shape[0]
    out = np.empty((steps + 1, d), dtype=float)
    out[0] = x0
    x = x0.copy()
    for k in range(steps):
        x = A @ x
        out[k + 1] = x
    return out


def rollout_edmd(
    K: np.ndarray,
    C: np.ndarray,
    degree: int,
    x0: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Roll out in feature space: psi_{t+1} = K psi_t, x_t = C psi_t."""
    d = x0.shape[0]
    out = np.empty((steps + 1, d), dtype=float)

    psi, _ = poly_features(x0.reshape(d, 1), degree=degree)
    psi = psi[:, 0]

    out[0] = C @ psi
    for k in range(steps):
        psi = K @ psi
        out[k + 1] = C @ psi
    return out
