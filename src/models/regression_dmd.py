import torch
import torch.nn as nn
import numpy as np
from itertools import product
from src.models.expander import ManualExpansion

import torch
import numpy as np

from src.models.expander import ManualExpansion


class Regression_DMD(ManualExpansion):
    """
    EDMD model built on top of the shared ManualExpansion class.
    """

    def __init__(
        self,
        state_dim=2,
        expansion_degree=3,
        rank=None,
        ridge=0.0,
        decoder_mode="fixed",
        bias=True,
        sine_cosine_expansion=False,
        expansion_type="general",
        system=None,
    ):
        super().__init__(
            state_dim=state_dim,
            expansion_degree=expansion_degree,
            bias=bias,
            sine_cosine_expansion=sine_cosine_expansion,
            expansion_type=expansion_type,
            system=system,
        )

        self.state_dim = state_dim
        self.rank = rank
        self.ridge = ridge
        self.decoder_mode = decoder_mode

        if self.decoder_mode not in {"regressed", "fixed"}:
            raise ValueError("decoder_mode must be 'regressed' or 'fixed'")

    def _build_fixed_decoder(self, dtype, device):
        """
        Build hardcoded decoder C (state_dim, expanded_dim):
        each state coordinate picks its corresponding linear monomial.
        """
        C = torch.zeros(
            self.state_dim,
            self.expanded_dim,
            dtype=dtype,
            device=device,
        )

        generic_names = [f"x{i+1}" for i in range(self.state_dim)]
        specific_names = ["x", "y", "z", "w", "v", "u"][:self.state_dim]

        for state_idx in range(self.state_dim):
            candidates = [generic_names[state_idx], specific_names[state_idx]]

            feature_idx = None
            for name in candidates:
                if name in self.expand_names:
                    feature_idx = self.expand_names.index(name)
                    break

            if feature_idx is None:
                raise ValueError(
                    f"Could not find state coordinate for index {state_idx} "
                    f"in expansion names {self.expand_names}"
                )

            C[state_idx, feature_idx] = 1.0

        return C

    def regression_matrix(self, x_in, y_out, method="svd", rank=None, ridge=0.0):
        """
        Regress matrix M such that y_out ≈ x_in @ M.T.
        
        Solves:
            min_M ||y_out - x_in @ M.T||_F
        
        Uses SVD-based pseudoinverse with optional rank truncation and ridge regularization.

        Parameters
        ----------
        x_in : torch.Tensor (N, k)
            Input features
        y_out : torch.Tensor (N, d)
            Output targets
        rank : int or None
            SVD rank truncation for stability
        ridge : float
            Ridge regularization to prevent overfitting on small singular values
        """
        if isinstance(x_in, np.ndarray):
            x_in = torch.tensor(x_in, dtype=torch.float64)
        if isinstance(y_out, np.ndarray):
            y_out = torch.tensor(y_out, dtype=torch.float64)

        X = x_in.T
        Y = y_out.T

        if method == "svd":
            U, s, Vt = torch.linalg.svd(X, full_matrices=False)

            r = len(s) if rank is None else max(1, min(rank, len(s)))
            U_r = U[:, :r]
            s_r = s[:r]
            Vt_r = Vt[:r, :]

            if ridge > 0.0:
                s_inv = s_r / (s_r**2 + ridge)
            else:
                s_inv = 1.0 / s_r

            M = (Y @ (Vt_r.T * s_inv)) @ U_r.T
        else:
            W = torch.linalg.lstsq(X.T, Y.T).solution
            M = W.T

        return M

    def fit(self, x, x_next, method="svd"):
        """
        Fit EDMD model to data (x, x_next).

        Returns
        -------
        K : torch.Tensor (expanded_dim, expanded_dim)
            Koopman operator in lifted space.
        C : torch.Tensor (state_dim, expanded_dim)
            Linear decoder from lifted space to original state space.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float64)
        if isinstance(x_next, np.ndarray):
            x_next = torch.tensor(x_next, dtype=torch.float64)

        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x_next.ndim == 1:
            x_next = x_next.unsqueeze(0)

        psi_x = self.expand(x)
        psi_y = self.expand(x_next)

        K = self.regression_matrix(
            psi_x,
            psi_y,
            method=method,
            rank=self.rank,
            ridge=self.ridge,
        )

        if self.decoder_mode == "fixed":
            C = self._build_fixed_decoder(dtype=psi_x.dtype, device=psi_x.device)
        else:
            C = self.regression_matrix(
                psi_x,
                x,
                method=method,
                rank=self.rank,
                ridge=self.ridge,
            )

        return K, C

    def rollout(self, K, C, x0, steps):
        """
        Roll out the EDMD model from initial state x0 for `steps`.

        Returns trajectory in original state space with shape (steps+1, state_dim).

        K maps lifted -> lifted, C maps lifted -> original.
        """        
        if isinstance(K, np.ndarray):
            K = torch.tensor(K, dtype=torch.float64)
        if isinstance(C, np.ndarray):
            C = torch.tensor(C, dtype=torch.float64)
        if isinstance(x0, np.ndarray):
            x0 = torch.tensor(x0, dtype=torch.float64)

        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)

        K = K.to(dtype=x0.dtype, device=x0.device)
        C = C.to(dtype=x0.dtype, device=x0.device)

        psi_current = self.expand(x0)
        x_current = psi_current @ C.T
        trajectory = [x_current[0]]

        for _ in range(steps):
            psi_next = psi_current @ K.T
            x_next = psi_next @ C.T
            trajectory.append(x_next[0])
            psi_current = psi_next

        return torch.stack(trajectory, dim=0)