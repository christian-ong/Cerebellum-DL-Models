import numpy as np
import torch
from torch.utils.data import Dataset


class OneStepTrajectoryDataset(Dataset):
    """
    Loads X with shape:
      - (T, state_dim)
      - (T, n_traj, state_dim)

    Then selects a subset of trajectories (traj_indices),
    and returns one-step pairs flattened across (time, traj).
    """
    def __init__(self, npz_path: str, traj_indices=None):
        data = np.load(npz_path)
        X = data["X"]

        # Ensure (T, n_traj, state_dim)
        if X.ndim == 2:
            X = X[:, None, :]
        elif X.ndim != 3:
            raise ValueError(f"Expected X to have 2 or 3 dims, got shape {X.shape}")

        # Select trajectories
        if traj_indices is None:
            X_sel = X
        else:
            X_sel = X[:, traj_indices, :]

        # One-step pairs
        x = X_sel[:-1]   # (T-1, n_traj_sel, state_dim)
        y = X_sel[1:]    # (T-1, n_traj_sel, state_dim)

        # Flatten
        Tm1, n_traj_sel, state_dim = x.shape
        x = x.reshape(Tm1 * n_traj_sel, state_dim)
        y = y.reshape(Tm1 * n_traj_sel, state_dim)

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
