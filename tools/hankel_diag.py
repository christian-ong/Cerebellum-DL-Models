import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.expander import HankelSVDDelayExpansion, DelayExpansion


def make_sin_trajectory(T=200, state_dim=2, freqs=(0.1, 0.05), noise_std=0.0):
    t = np.arange(T)
    X = np.zeros((T, state_dim), dtype=np.float64)
    for i in range(state_dim):
        X[:, i] = np.sin(2 * np.pi * freqs[i] * t)
    if noise_std > 0:
        X += np.random.randn(*X.shape) * noise_std
    return X


def build_delay_windows(X, delay_depth=25):
    # X: (T, d)
    T, d = X.shape
    windows = []
    for t in range(delay_depth - 1, T):
        window = []
        for k in range(delay_depth):
            window.append(X[t - k])
        windows.append(np.concatenate(window, axis=-1))
    return np.asarray(windows)


def run_diag():
    torch.set_printoptions(precision=6, sci_mode=True)

    # Synthetic data
    state_dim = 2
    T = 200
    delay_depth = 20
    X = make_sin_trajectory(T=T, state_dim=state_dim, freqs=(0.05, 0.08), noise_std=0.01)
    X_delay = build_delay_windows(X, delay_depth=delay_depth)
    print("X_delay shape:", X_delay.shape)

    # Convert to torch
    X_delay_t = torch.as_tensor(X_delay, dtype=torch.float64)

    # Fit HankelSVD
    rank = 6
    hankel = HankelSVDDelayExpansion(state_dim=state_dim, delay_depth=delay_depth, rank=rank, center=True)
    hankel.fit_state_scaler(X_delay_t)
    print("history_scale stats: min,median,max:", float(hankel.history_scale.min()), float(hankel.history_scale.median()), float(hankel.history_scale.max()))

    hankel.fit(X_delay_t)
    print("is_fitted:", hankel.is_fitted)
    print("singular_values:", hankel.singular_values.cpu().numpy())
    print("components shape:", hankel.components.shape)

    # Project and reconstruct
    Z = hankel.expand(X_delay_t).to(dtype=torch.float64)
    if hankel.bias:
        Z_scores = Z[:, 1:]
    else:
        Z_scores = Z
    print("Z shape:", Z.shape)

    # reconstruct full history and compute RMSE on head state
    X_recon_head = hankel.de_expand(Z).cpu().numpy()
    head_true = X_delay[:, :state_dim]
    rmse_head = np.sqrt(np.mean((X_recon_head - head_true) ** 2))
    print(f"Head-state reconstruction RMSE: {rmse_head:.6e}")

    # Compare to raw DelayExpansion unscaled
    delay_exp = DelayExpansion(state_dim=state_dim, delay_depth=delay_depth, bias=True)
    delay_exp.fit_state_scaler(X_delay_t.float())
    H_scaled = (torch.as_tensor(X_delay, dtype=torch.float32) / delay_exp.history_scale).numpy()
    print("Delay history scale stats (min,median,max):", float(delay_exp.history_scale.min()), float(delay_exp.history_scale.median()), float(delay_exp.history_scale.max()))

    # Print a few sample singular energy ratios
    s = hankel.singular_values.cpu().numpy()
    energy = s ** 2
    print("singular energy ratios:", (energy / energy.sum()))
    print("cumulative energy:", np.cumsum(energy) / energy.sum())

    # Inspect conditioning of component matrix
    C = hankel.components.cpu().numpy()  # (rank, history_dim)
    cond = np.linalg.cond(C)
    print("components cond (approx):", cond)


if __name__ == '__main__':
    run_diag()
