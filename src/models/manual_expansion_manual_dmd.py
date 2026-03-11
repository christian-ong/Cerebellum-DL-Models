import torch
import torch.nn as nn
import numpy as np
from itertools import product


class ManualExpansion_ManualDMD(nn.Module):
    """
    Manual polynomial EDMD model.

    Learns:
      - K: Koopman operator in lifted space, psi(x_{t+1}) ~= psi(x_t) @ K.T
      - C: Fixed linear decoder to state space, x ~= psi(x) @ C.T
    """

    def __init__(
        self,
        state_dim=2,
        expansion_degree=3,
        rank=None,
        ridge=0.0,
        include_bias=True,
        decoder_mode="fixed",
    ):
        super().__init__()

        self.state_dim = state_dim
        self.expansion_degree = expansion_degree
        self.include_bias = include_bias
        self.decoder_mode = decoder_mode
        if self.decoder_mode not in {"regressed", "fixed"}:
            raise ValueError("decoder_mode must be 'regressed' or 'fixed'")

        # Basis expansion of the state.
        # all exponent tuples with total degree <= expansion_degree
        self.polynomial_expansions = []
        for exps in product(range(self.expansion_degree + 1), repeat=self.state_dim):
            total_degree = sum(exps)
            if total_degree <= self.expansion_degree and (self.include_bias or total_degree > 0):
                self.polynomial_expansions.append(exps)

        # Sort by total degree, then lexicographically for reproducibility and stability
        self.polynomial_expansions.sort(key=lambda exps: (sum(exps), exps))

        # Lookup dict for fast exponent-to-index mapping
        self._exp_to_idx = {exps: i for i, exps in enumerate(self.polynomial_expansions)}

        self.expand_names = []
        for exps in self.polynomial_expansions:
            terms = [f"x{i+1}^{e}" for i, e in enumerate(exps)]
            self.expand_names.append(" * ".join(terms))

        self.expanded_dim = len(self.polynomial_expansions)
        self.rank = rank
        self.ridge = ridge


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

        for state_idx in range(self.state_dim):
            exps = [0] * self.state_dim
            exps[state_idx] = 1
            exps = tuple(exps)
            if exps not in self._exp_to_idx:
                raise ValueError(
                    "Fixed decoder requires linear terms in expansion. "
                    "Use expansion_degree >= 1."
                )
            feature_idx = self._exp_to_idx[exps]
            C[state_idx, feature_idx] = 1.0

        return C


    def expand(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float64)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        if x.shape[1] != self.state_dim:
            raise ValueError(
                f"Expected input with state_dim={self.state_dim}, got {x.shape[1]}"
            )

        expanded_features = []

        for exponents in self.polynomial_expansions:
            term = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
            for dim_idx, exponent in enumerate(exponents):
                if exponent > 0:
                    term = term * (x[:, dim_idx] ** exponent)
            expanded_features.append(term)

        x_expanded = torch.stack(expanded_features, dim=1)

        return x_expanded
    

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
        # Convert to torch if needed
        if isinstance(x_in, np.ndarray):
            x_in = torch.tensor(x_in, dtype=torch.float64)
        if isinstance(y_out, np.ndarray):
            y_out = torch.tensor(y_out, dtype=torch.float64)

        # SVD-based approach matching traditional DMD
        # x_in is (N, k), transpose to (k, N) for SVD
        X = x_in.T   # (k, N)
        Y = y_out.T  # (d, N)
        
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
            
            # M will be shape (d, k)
            M = (Y @ (Vt_r.T * s_inv)) @ U_r.T
        else:
            # Direct least-squares solution (not SVD-based) 
            W = torch.linalg.lstsq(X.T, Y.T).solution  # (k, d)
            M = W.T  # (d, k)

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

        psi_x = self.expand(x)
        psi_y = self.expand(x_next)

        # Koopman in lifted space: psi_y ~= psi_x @ K.T
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
            # Decoder: x ~= psi_x @ C.T
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