import numpy as np
from typing import List, Optional, Tuple

# ==================================================
# Core linear regression (used by linear / DMD / EDMD)
# ==================================================

def fit_dmd_eig(
    X: np.ndarray,
    Y: np.ndarray,
    rank: Optional[int] = None,
    ridge: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit DMD using eigenvalue decomposition of the reduced operator.
    
    Based on DMD formulation:
        1. X = U Σ V*          (SVD of input snapshots)
        2. Ā = U* X' V Σ^(-1)  (reduced operator)
        3. Ā W = W Λ           (eigendecomposition)
        4. Φ = X' V Σ^(-1) W   (dynamic modes)

    Parameters
    ----------
    X : (N, d)
        Input snapshots
    Y : (N, d)
        Output snapshots (X shifted by one timestep)
    rank : int or None
        Truncation rank (SVD)
    ridge : float
        Ridge regularization parameter

    Returns
    -------
    eigenvalues : (r,)
        Eigenvalues Λ of the reduced operator
    eigenvectors : (r, r)
        Eigenvectors W of the reduced operator
    U : (d, r)
        Left singular vectors
    sigma_inv : (r,)
        Inverse of singular values
    V_T : (r, N)
        Right singular vectors transposed
    """
    # Work with column snapshots
    Xc = X.T  # (d, N)
    Yc = Y.T  # (d, N)

    # Step 1: SVD of X
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    r = len(s) if rank is None else max(1, min(rank, len(s)))
    U_r = U[:, :r]      # (d, r)
    sigma = s[:r]       # (r,)
    V_T_r = Vt[:r, :]   # (r, N)

    # Handle regularization
    if ridge > 0.0:
        sigma_inv = sigma / (sigma**2 + ridge)
    else:
        sigma_inv = 1.0 / sigma

    # Step 2: Compute reduced operator Ā = U* Y V Σ^(-1)
    A_tilde = (U_r.T @ Yc) @ (V_T_r.T * sigma_inv)  # (r, r)

    # Step 3: Eigendecomposition of Ā
    Lambda, W = np.linalg.eig(A_tilde)

    return Lambda, W, U_r, sigma_inv, V_T_r

# ==================================================
# DMD
# ==================================================

def fit_dmd(
    X: np.ndarray,
    Y: np.ndarray,
    rank: Optional[int] = None,
    ridge: float = 0.0,
) -> np.ndarray:
    """
    Fit DMD operator A such that:
        x_{t+1} ≈ A x_t

    X, Y are one-step snapshot pairs.
    
    Parameters
    ----------
    X : (N, d)
        Input snapshots
    Y : (N, d)
        Output snapshots
    rank : int or None
        Truncation rank
    ridge : float
        Ridge regularization
    return_eig : bool
        If True, return the eigenvalue decomposition components.
        If False, return the reconstructed A matrix.

    Returns
    -------
    A or components : depends on return_eig
        If return_eig=False: (d, d) linear operator A
        If return_eig=True: tuple of (Lambda, W, U, sigma_inv, V_T, X_prime)
    """
    Lambda, W, U, sigma_inv, V_T = fit_dmd_eig(
        X, Y, rank=rank, ridge=ridge
    )
    
    # Reconstruct A matrix from eigendecomposition for backward compatibility
    # Φ = X' V Σ^(-1) W where Φ is (d, r)
    Yc = Y.T  # (d, N)
    Phi = Yc @ (V_T.T * sigma_inv) @ W  # (d, r) @ (r, r) = (d, r)
    
    return Lambda, Phi


# ==================================================
# Rollouts
# ==================================================

def rollout_dmd_eig(
    Lambda: np.ndarray,
    Phi: np.ndarray,
    x0: np.ndarray,
    steps: int,
) -> np.ndarray:
    """
    Roll out DMD dynamics using the efficient eigenvalue formula.
    
    Computes x(k) using:
        x(k) = Φ Λ^k b_0
    where b_0 = Φ^(-1) x_0 are the initial coefficients
    
    This avoids matrix multiplication k times by directly computing Λ^k.
    
    Parameters
    ----------
    Lambda : (r,)
        Eigenvalues from DMD
    Phi : (d, r)
        Dynamic modes (eigenvectors scaled)
    x0 : (d,)
        Initial condition
    steps : int
        Number of rollout steps

    Returns
    -------
    trajectory : (steps+1, d)
        Rollout trajectory
    """
    d = x0.shape[0]
    out = np.empty((steps + 1, d), dtype=float)
    out[0] = x0

    # Compute initial coefficients: b_0 = Φ^(-1) x_0
    Phi_pinv = np.linalg.pinv(Phi)
    b0 = Phi_pinv @ x0  # (r,)
    
    # For each timestep k, compute: x(k) = Φ Λ^k b_0
    for k in range(1, steps + 1):
        Lambda_k = np.diag(Lambda ** k)  # Λ^k as diagonal matrix
        x_k = Phi @ Lambda_k @ b0
        out[k] = x_k

    return out


import numpy as np
import matplotlib.pyplot as plt
import os


# ==================================================
# Utilities
# ==================================================

def discrete_to_continuous_eigs(Lambda, dt):
    """
    Convert discrete-time eigenvalues to continuous-time:
        mu = log(lambda) / dt
    """
    return np.log(Lambda) / dt


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ==================================================
# 1. Eigenvalues in complex plane (STABILITY)
# ==================================================

def plot_dmd_eigenvalues(
    Lambda,
    savepath,
    title="DMD Eigenvalues",
):
    """
    Question answered:
        Is the system stable / oscillatory?
    """
    ensure_dir(savepath)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Unit circle (quiet)
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(
        np.cos(theta),
        np.sin(theta),
        linestyle="--",
        color="gray",
        linewidth=1,
        alpha=0.6,
        label="Unit circle",
    )

    # Eigenvalues
    sc = ax.scatter(
        Lambda.real,
        Lambda.imag,
        c=np.abs(Lambda),
        cmap="viridis",
        s=120,
        edgecolors="black",
        zorder=3,
    )

    for i, lam in enumerate(Lambda):
        ax.text(
            lam.real * 1.05,
            lam.imag * 1.05,
            f"{i}",
            fontsize=10,
            ha="center",
            va="center",
        )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)

    ax.set_xlabel("Re($\\lambda$)")
    ax.set_ylabel("Im($\\lambda$)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(False)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("$|\\lambda|$")

    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.show()
    plt.close()


# ==================================================
# 2. Mode amplitudes (IMPORTANCE)
# ==================================================

def plot_mode_amplitudes(
    Lambda,
    Phi,
    x0,
    savepath,
    title="DMD Mode Amplitudes",
):
    """
    Question answered:
        Which modes actually contribute to the trajectory?
    """
    ensure_dir(savepath)

    Phi_pinv = np.linalg.pinv(Phi)
    b0 = Phi_pinv @ x0

    order = np.argsort(-np.abs(b0))
    b0 = b0[order]
    Lambda = Lambda[order]

    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(
        range(len(b0)),
        np.abs(b0),
        color=plt.cm.viridis(np.abs(Lambda) / np.max(np.abs(Lambda))),
        edgecolor="black",
    )

    ax.set_xlabel("Mode index (sorted)")
    ax.set_ylabel("$|b_0|$")
    ax.set_title(title)

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.show()
    plt.close()


# ==================================================
# 3. DMD modes as directions (GEOMETRY)
# ==================================================

def plot_dmd_modes_2d(
    Phi,
    Lambda,
    savepath,
    title="DMD Modes (Geometry)",
):
    """
    Question answered:
        What directions dominate the dynamics?
    """
    ensure_dir(savepath)
    assert Phi.shape[0] == 2, "Only valid for 2D systems"

    fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(Phi.shape[1]):
        v = Phi[:, i].real
        v /= np.linalg.norm(v)

        ax.arrow(
            0,
            0,
            v[0],
            v[1],
            head_width=0.06,
            length_includes_head=True,
            linewidth=2,
            color=plt.cm.viridis(np.abs(Lambda[i]) / np.max(np.abs(Lambda))),
            alpha=0.9,
        )

        ax.text(
            v[0] * 1.15,
            v[1] * 1.15,
            f"$\\lambda_{i}$",
            fontsize=11,
            ha="center",
            va="center",
        )

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.show()
    plt.close()


# ==================================================
# 4. Continuous-time spectrum (TIME SCALES)
# ==================================================

def plot_continuous_spectrum(
    Lambda,
    dt,
    savepath,
    title="Continuous-Time DMD Spectrum",
):
    """
    Question answered:
        What are the decay rates and frequencies?
    """
    ensure_dir(savepath)

    mu = discrete_to_continuous_eigs(Lambda, dt)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(
        mu.real,
        mu.imag,
        s=120,
        c=np.abs(mu.imag),
        cmap="plasma",
        edgecolors="black",
    )

    ax.axvline(0, color="black", linewidth=0.7)
    ax.axhline(0, color="black", linewidth=0.5)

    ax.set_xlabel("Growth / Decay rate")
    ax.set_ylabel("Frequency")
    ax.set_title(title)

    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.show()
    plt.close()


# ==================================================
# 5. Dominant conjugate-pair reconstruction
# ==================================================

def plot_conjugate_mode_reconstruction(
    Lambda,
    Phi,
    x0,
    steps,
    true_traj,
    savepath,
    title="Dominant DMD Mode Pair Reconstruction",
):
    """
    Question answered:
        Does the dominant oscillatory mode explain the dynamics?
    """
    ensure_dir(savepath)

    # Find dominant conjugate pair
    idx = np.argsort(-np.abs(Lambda))
    i, j = idx[:2]

    Phi_ij = Phi[:, [i, j]]
    Lambda_ij = np.diag(Lambda[[i, j]])

    b0 = np.linalg.pinv(Phi_ij) @ x0

    X_rec = np.zeros_like(true_traj)
    for k in range(steps + 1):
        X_rec[k] = (Phi_ij @ (Lambda_ij ** k) @ b0).real

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(true_traj[:, 0], true_traj[:, 1], label="True", linewidth=2)
    ax.plot(
        X_rec[:, 0],
        X_rec[:, 1],
        "--",
        linewidth=2,
        label="Dominant DMD mode pair",
    )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.show()
    plt.close()
