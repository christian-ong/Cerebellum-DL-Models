import numpy as np
import torch
import os
from torch.utils.data import Dataset


VALID_SPLITS = ("train", "val", "test")


def resolve_dataset_dir(path: str) -> str:
    """
    Resolve dataset directory that contains train.npz, val.npz, test.npz.

    Accepted inputs:
      - dataset directory path, e.g. data/trajectories/vanderpol
      - split file path, e.g. data/trajectories/vanderpol/train.npz
    """
    p = os.path.normpath(path)

    if os.path.isdir(p):
        return p

    file_name = os.path.basename(p)
    if file_name in {f"{s}.npz" for s in VALID_SPLITS}:
        return os.path.dirname(p)

    raise ValueError(
        "data_path must be a dataset directory (containing train.npz/val.npz/test.npz) "
        "or a split file path like .../train.npz"
    )


def resolve_split_npz_path(path: str, split: str) -> str:
    """
        Resolve path to a split dataset file for the clean directory layout:
            <dataset_dir>/train.npz
            <dataset_dir>/val.npz
            <dataset_dir>/test.npz
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got '{split}'")
    
    dataset_dir = resolve_dataset_dir(path)
    return os.path.join(dataset_dir, f"{split}.npz")


class OneStepTrajectoryDataset(Dataset):
    """
    One-step prediction dataset from simulated trajectories.

        Works with clean split files under a dataset directory:
            - <dataset_dir>/train.npz
            - <dataset_dir>/val.npz
            - <dataset_dir>/test.npz

    Supports:
      - split = "train" | "val" | "test"
      - X shape (T, d) or (T, n_traj, d)
      - optional time-delay embedding when delay_depth > 1
    """

    def __init__(self, 
                 npz_path: str, 
                 split: str = "train",
                 subset: float = 1.0,
                 rollout_horizon: int = 0,
                 delay_depth: int = 1):
        full_path = resolve_split_npz_path(npz_path, split)
        
        data = np.load(full_path)

        X = data["X"]

        # Ensure (T, n_traj, d)
        if X.ndim == 2:
            X = X[:, None, :]
        elif X.ndim != 3:
            raise ValueError(f"Expected X to have 2 or 3 dims, got {X.shape}")

        # Split files contain only trajectories for that split.
        traj_idx = np.arange(X.shape[1])

        if traj_idx.size == 0:
            self.x = torch.empty(0)
            self.y = torch.empty(0)
            return

        # subset trajectories 
        if subset < 1.0:
            print(f"Using subset of data: {subset*100:.1f}% of trajectories")
            print(f"Original shape: {X[:, traj_idx, :].shape}")

            n_traj = len(traj_idx)
            n_subset = int(np.ceil(n_traj * subset))
            traj_idx = np.random.choice(traj_idx, size=n_subset, replace=False)

            print(f"Subset shape: {X[:, traj_idx, :].shape}")

        X = X[:, traj_idx, :]

        if rollout_horizon < 0:
            raise ValueError("rollout_horizon must be non-negative")
        if delay_depth < 1:
            raise ValueError("delay_depth must be positive")

        self.delay_depth = delay_depth

        # Build one-step pairs, optionally accompanied by a short future window.
        # When delay_depth > 1, the input x is a stacked history:
        #   [x(t), x(t-1), ..., x(t-delay_depth+1)].
        x_list = []
        y_list = []
        rollout_list = []

        max_start = X.shape[0] - 1 - rollout_horizon
        for traj in range(X.shape[1]):
            traj_series = X[:, traj, :]

            for t in range(delay_depth - 1, max_start):
                if delay_depth == 1:
                    x_list.append(traj_series[t])
                else:
                    x_list.append(
                        np.concatenate(
                            [traj_series[t - k] for k in range(delay_depth)],
                            axis=-1,
                        )
                    )

                y_list.append(traj_series[t + 1])

                if rollout_horizon > 0:
                    future_window = traj_series[t + 1 : t + 1 + rollout_horizon]
                    rollout_list.append(future_window)

        if len(x_list) == 0:
            self.x = torch.empty(0)
            self.y = torch.empty(0)
            self.rollout_targets = None
            return

        x = np.asarray(x_list)
        y = np.asarray(y_list)

        print(x.shape, y.shape)

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        if rollout_horizon > 0:
            rollout_targets = np.asarray(rollout_list)
            self.rollout_targets = torch.tensor(rollout_targets, dtype=torch.float32)
        else:
            self.rollout_targets = None

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.rollout_targets is None:
            return self.x[idx], self.y[idx]

        return self.x[idx], self.y[idx], self.rollout_targets[idx]
