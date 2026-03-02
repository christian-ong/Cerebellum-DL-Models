import torch
import torch.nn as nn
import numpy as np


class ManualExpand_ManualDMD(nn.Module):
    """
    Simply expands the state
    """

    def __init__(self, state_dim=2, expansion_degree=3, rank=None, ridge=0.0):
        super().__init__()

        # Basis expansion of the state
        self.polynomial_expansions = []
        self.expand_names = []
        for d in range(1, expansion_degree + 1):
            for i in range(d + 1):
                e_x = d-i
                e_y = i
                self.polynomial_expansions.append((e_x, e_y))
                self.expand_names.append(f"x^{e_x} * y^{e_y}")
        self.expanded_dim = len(self.polynomial_expansions)
        self.rank = rank
        self.ridge = ridge


    def expand(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        expanded_features = []

        for i, j in self.polynomial_expansions:
            expanded_features.append((x[:, 0] ** i) * (x[:, 1] ** j))

        x_expanded = torch.stack(expanded_features, dim=1)

        return x_expanded
    

    def regression_K(self, x_expanded, x_next, method="svd", rank=None, ridge=0.0):
        """
        DMD regression to find K such that x_next ≈ x_expanded @ K.T.
        
        K maps from expanded space (k features) to original state space (d dims).
        This learns how to predict the next original state from the expanded current state.

        Solves the least-squares problem:
            min_K ||x_next - x_expanded @ K.T||_F
        
        Uses SVD-based pseudoinverse with optional rank truncation and ridge regularization,
        matching the traditional DMD implementation.

        Parameters
        ----------
        x_expanded : torch.Tensor (N, k)
            Expanded features
        x_next : torch.Tensor (N, d)
            Next state in original space
        rank : int or None
            SVD rank truncation for stability
        ridge : float
            Ridge regularization to prevent overfitting on small singular values
        """
        # Convert to torch if needed
        if isinstance(x_expanded, np.ndarray):
            x_expanded = torch.tensor(x_expanded, dtype=torch.float32)
        if isinstance(x_next, np.ndarray):
            x_next = torch.tensor(x_next, dtype=torch.float32)

        # SVD-based approach matching traditional DMD
        # x_expanded is (N, k), transpose to (k, N) for SVD
        X = x_expanded.T  # (k, N)
        Y = x_next.T      # (d, N)
        
        if method == "svd":

            U, s, Vt = torch.linalg.svd(X, full_matrices=False)
        
            # Rank truncation
            r = len(s) if rank is None else max(1, min(rank, len(s)))
            U_r = U[:, :r]      # (k, r)
            s_r = s[:r]         # (r,)
            Vt_r = Vt[:r, :]    # (r, N)
        
            # Ridge regularization
            if ridge > 0.0:
                s_inv = s_r / (s_r**2 + ridge)
            else:
                s_inv = 1.0 / s_r
            
            # K = Y @ Vt_r.T @ diag(s_inv) @ U_r.T
            # K will be shape (d, k)
            K = (Y @ (Vt_r.T * s_inv)) @ U_r.T
        else:
            # Direct least-squares solution (not SVD-based) 
            W = torch.linalg.lstsq(X.T, Y.T).solution  # (k, d)
            K = W.T  # (d, k)

        return K
    

    def fit(self, x, x_next):
        """
        Fit the DMD model to data (x, x_next) by expanding x and regressing K.
        
        K maps: expanded(x) @ K.T → x_next (in original space)
        """

        # Expand current state only
        x_big = self.expand(x)

        # Regress K such that x_next ≈ x_big @ K.T
        # K will be (state_dim, expanded_dim), e.g., (2, 9) for 2D state with degree 3
        K = self.regression_K(x_big, x_next, rank=self.rank, ridge=self.ridge)

        return K

    def rollout(self, K, x0, steps):
        """
        Roll out the DMD model from initial state x0 for steps using K.
        Returns trajectory in original state space with shape (steps+1, state_dim).
        
        K maps: expanded(x) @ K.T → x_next (shape: state_dim × expanded_dim)
        """
        if isinstance(K, np.ndarray):
            K = torch.tensor(K, dtype=torch.float32)

        if isinstance(x0, np.ndarray):
            x0 = torch.tensor(x0, dtype=torch.float32)

        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)

        K = K.to(dtype=x0.dtype, device=x0.device)

        trajectory = [x0[0]]
        x_current = x0

        for _ in range(steps):
            # Expand current state
            x_big = self.expand(x_current)
            # Predict next state in original space
            x_next = x_big @ K.T
            trajectory.append(x_next[0])
            x_current = x_next

        return torch.stack(trajectory, dim=0)