import numpy as np
from typing import List, Optional, Tuple

# ==================================================
# EDMD dictionary: polynomial features
# ==================================================

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


def poly_features(
    X: np.ndarray,
    degree: int,
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """
    Polynomial dictionary ψ(x) with all monomials up to given degree.

    Parameters
    ----------
    X : (N, d)
        Samples
    degree : int

    Returns
    -------
    PsiX : (N, k)
        Lifted features
    exps : list of exponent tuples
    """
    N, d = X.shape
    exps = _exponents_upto_degree(d, degree)
    k = len(exps)

    Psi = np.empty((N, k), dtype=float)

    for i, e in enumerate(exps):
        e = np.array(e, dtype=int)[None, :]
        Psi[:, i] = np.prod(X ** e, axis=1)

    return Psi, exps


#### MAYBE DELETE/ADJUST #########
def fit_linear_map(
    X: np.ndarray,
    Y: np.ndarray,
    rank: Optional[int] = None,
    ridge: float = 0.0,
) -> np.ndarray:
    """
    Fit the best linear map M such that:
        Y ≈ X @ M^T   (or equivalently y = M x)
    
    (Kept for backward compatibility with EDMD)

    Parameters
    ----------
    X : (N, d)
        Input snapshots
    Y : (N, d)
        Output snapshots
    rank : int or None
        Truncation rank (SVD)
    ridge : float
        Ridge regularization parameter

    Returns
    -------
    M : (d, d)
        Linear map
    """
    # Work with column snapshots
    Xc = X.T  # (d, N)
    Yc = Y.T  # (d, N)

    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    r = len(s) if rank is None else max(1, min(rank, len(s)))
    Ur, sr, Vtr = U[:, :r], s[:r], Vt[:r, :]

    if ridge > 0.0:
        inv = sr / (sr**2 + ridge)
    else:
        inv = 1.0 / sr

    M = (Yc @ (Vtr.T * inv)) @ Ur.T
    return M

# ==================================================
# EDMD
# ==================================================

def fit_edmd(
    X: np.ndarray,
    Y: np.ndarray,
    degree: int,
    rank: Optional[int] = None,
    ridge: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit EDMD Koopman operator K and linear decoder C.

    ψ(x_{t+1}) ≈ K ψ(x_t)
    x_t ≈ C ψ(x_t)
    """
    PsiX, _ = poly_features(X, degree)
    PsiY, _ = poly_features(Y, degree)

    K = fit_linear_map(PsiX, PsiY, rank=rank, ridge=ridge)
    C = fit_linear_map(PsiX, X, rank=rank, ridge=ridge)

    return K, C


def rollout_edmd(
    K: np.ndarray,
    C: np.ndarray,
    degree: int,
    x0: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Roll out EDMD dynamics in feature space."""
    d = x0.shape[0]
    out = np.empty((steps + 1, d), dtype=float)

    psi, _ = poly_features(x0[None, :], degree)
    psi = psi[0]

    out[0] = C @ psi
    for k in range(steps):
        psi = K @ psi
        out[k + 1] = C @ psi

    return out
