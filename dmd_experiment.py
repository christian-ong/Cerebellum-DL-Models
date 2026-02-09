import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

# ==================================================
# USER CONFIG
# ==================================================

DT = 1.0        # discrete-time step (purely conceptual here)
T = 80          # number of steps
SEED = 0

rng = np.random.default_rng(SEED)

# ==================================================
# Define discrete-time linear system
# x_{k+1} = A x_k
# ==================================================

A = np.array([
    [0.9, -0.3],
    [0.3,  0.9],
], dtype=float)

x0 = np.array([1.0, 0.2])

print("True system matrix A:\n", A)

# ==================================================
# Eigen-decomposition of A
# ==================================================

eigvals_A, V = np.linalg.eig(A)
V_inv = np.linalg.inv(V)

print("\nEigenvalues of A:", eigvals_A)
print("Eigenvectors V:\n", V)

# ==================================================
# Simulation via eigen-coordinates
# y_{k+1} = Λ y_k
# x_k = V y_k
# ==================================================

def simulate_via_eigendecomp(V, eigvals, x0, T):
    y = V_inv @ x0
    X = np.zeros((T + 1, len(x0)), dtype=complex)
    X[0] = V @ y

    for k in range(T):
        y = eigvals * y
        X[k + 1] = V @ y

    return np.real_if_close(X)

X_modal = simulate_via_eigendecomp(V, eigvals_A, x0, T)

# ==================================================
# Direct simulation with A
# ==================================================

def simulate_direct(A, x0, T):
    X = np.zeros((T + 1, len(x0)))
    X[0] = x0
    for k in range(T):
        X[k + 1] = A @ X[k]
    return X

X_direct = simulate_direct(A, x0, T)

# ==================================================
# Verify equivalence
# ==================================================

rel_err = np.linalg.norm(X_direct - X_modal) / np.linalg.norm(X_direct)
print(f"\nRelative error (direct vs eigen-sim): {rel_err:.2e}")

# ==================================================
# Build DMD snapshot matrices
# ==================================================

X = X_direct[:-1].T   # x_k
Y = X_direct[1:].T    # x_{k+1}

print("Snapshot shapes:", X.shape, Y.shape)

# ==================================================
# DMD operator identification
# ==================================================

def fit_linear_map(X, Y, rank=None, ridge=0.0):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    r = len(s) if rank is None else max(1, min(rank, len(s)))
    Ur, sr, Vtr = U[:, :r], s[:r], Vt[:r, :]

    inv = (sr / (sr**2 + ridge)) if ridge > 0 else (1.0 / sr)
    M = (Y @ (Vtr.T * inv)) @ Ur.T
    return M

M_dmd = fit_linear_map(X, Y)

print("\nRecovered DMD operator M:\n", M_dmd)
print("Operator error ||A - M|| =", np.linalg.norm(A - M_dmd))

# ==================================================
# Eigen-analysis: A vs DMD operator
# ==================================================

eigvals_M, _ = np.linalg.eig(M_dmd)

print("\nEigenvalues of A:", eigvals_A)
print("Eigenvalues of M:", eigvals_M)

idxA = np.argsort(-np.abs(eigvals_A))
idxM = np.argsort(-np.abs(eigvals_M))

print("\nSorted eig(A):", eigvals_A[idxA])
print("Sorted eig(M):", eigvals_M[idxM])
print("Eigenvalue difference:",
      np.abs(eigvals_A[idxA] - eigvals_M[idxM]))

# ==================================================
# Visualization
# ==================================================

plt.figure(figsize=(5,5))
plt.plot(X_direct[:,0], X_direct[:,1], "o-", ms=3, label="Direct A")
plt.plot(X_modal[:,0], X_modal[:,1], "x--", ms=3, label="Eigen-sim")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("Linear system: direct vs eigen-coordinate simulation")
plt.show()
