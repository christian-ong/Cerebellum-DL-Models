import numpy as np
import torch
from torch.utils.data import Dataset


class OneStepTrajectoryDataset(Dataset):
    """
    One-step prediction dataset from simulated trajectories.

    Supports:
      - split = "train" | "val" | "all"
      - X shape (T, d) or (T, n_traj, d)
    """

    def __init__(self, npz_path: str, split: str = "train"):
        data = np.load(npz_path)

        X = data["X"]

        # Ensure (T, n_traj, d)
        if X.ndim == 2:
            X = X[:, None, :]
        elif X.ndim != 3:
            raise ValueError(f"Expected X to have 2 or 3 dims, got {X.shape}")

        # Select trajectories by split
        if split == "train":
            traj_idx = data["train_idx"]
        elif split == "val":
            traj_idx = data["val_idx"]
        elif split == "all":
            traj_idx = np.arange(X.shape[1])
        else:
            raise ValueError(f"Unknown split: {split}")

        if traj_idx.size == 0:
            self.x = torch.empty(0)
            self.y = torch.empty(0)
            return

        X = X[:, traj_idx, :]

        # One-step pairs
        x = X[:-1]
        y = X[1:]

        Tm1, n_traj, d = x.shape
        x = x.reshape(Tm1 * n_traj, d)
        y = y.reshape(Tm1 * n_traj, d)

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
