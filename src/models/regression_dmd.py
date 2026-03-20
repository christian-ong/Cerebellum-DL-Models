import torch
import numpy as np
from src.models.expander import ManualExpansion


class Regression_DMD(ManualExpansion):
    """
    EDMD model built on top of the shared ManualExpansion class.

    Improvements over the original:
    - optional input-state normalization before expansion
    - optional lifted-feature normalization before regression
    - separate ridge for Koopman and decoder fits
    - optional spectral-radius clipping for stability
    - optional residual decoding: predict dx and add to x
    """

    def __init__(
        self,
        state_dim=2,
        expansion_degree=3,
        rank=None,
        ridge=0.0,
        decoder_ridge=None,
        decoder_mode="regressed",
        bias=True,
        sine_cosine_expansion=False,
        expansion_type="general",
        system=None,
        normalize_state=True,
        normalize_lifted=True,
        residual_decode=False,
        max_spectral_radius=None,
        eps=1e-10,
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
        self.decoder_ridge = ridge if decoder_ridge is None else decoder_ridge
        self.decoder_mode = decoder_mode
        self.normalize_state = normalize_state
        self.normalize_lifted = normalize_lifted
        self.residual_decode = residual_decode
        self.max_spectral_radius = max_spectral_radius
        self.eps = eps

        if self.decoder_mode not in {"regressed", "fixed"}:
            raise ValueError("decoder_mode must be 'regressed' or 'fixed'")

        # learned / stored at fit time
        self.x_mean = None
        self.x_scale = None
        self.psi_scale = None
        self.K_fitted = None
        self.C_fitted = None

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float64)
        return x.to(dtype=torch.float64)

    def _safe_scale(self, x, dim=0):
        scale = torch.std(x, dim=dim, unbiased=False)
        scale = torch.where(scale > self.eps, scale, torch.ones_like(scale))
        return scale

    def _normalize_x(self, x):
        if not self.normalize_state:
            return x
        return (x - self.x_mean) / self.x_scale

    def _denormalize_x(self, x):
        if not self.normalize_state:
            return x
        return x * self.x_scale + self.x_mean

    def _build_fixed_decoder(self, dtype, device):
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
        Regress matrix M such that y_out ≈ x_in @ M.T
        """
        x_in = self._to_tensor(x_in)
        y_out = self._to_tensor(y_out)

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
                s_inv = 1.0 / torch.clamp(s_r, min=self.eps)

            M = (Y @ (Vt_r.T * s_inv)) @ U_r.T
        else:
            W = torch.linalg.lstsq(X.T, Y.T).solution
            M = W.T

        return M

    def _clip_spectral_radius(self, K):
        if self.max_spectral_radius is None:
            return K

        eigvals = torch.linalg.eigvals(K)
        rho = torch.max(torch.abs(eigvals)).real
        if rho > self.max_spectral_radius:
            K = K * (self.max_spectral_radius / rho)
        return K

    def fit(self, x, x_next, method="svd"):
        x = self._to_tensor(x)
        x_next = self._to_tensor(x_next)

        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x_next.ndim == 1:
            x_next = x_next.unsqueeze(0)

        # state normalization
        self.x_mean = torch.mean(x, dim=0)
        self.x_scale = self._safe_scale(x, dim=0)

        x_n = self._normalize_x(x)
        x_next_n = self._normalize_x(x_next)

        psi_x = self.expand(x_n)
        psi_y = self.expand(x_next_n)

        # lifted normalization
        if self.normalize_lifted:
            self.psi_scale = self._safe_scale(psi_x, dim=0)
        else:
            self.psi_scale = torch.ones(psi_x.shape[1], dtype=psi_x.dtype, device=psi_x.device)

        psi_x_s = psi_x / self.psi_scale
        psi_y_s = psi_y / self.psi_scale

        K = self.regression_matrix(
            psi_x_s,
            psi_y_s,
            method=method,
            rank=self.rank,
            ridge=self.ridge,
        )

        K = self._clip_spectral_radius(K)

        if self.decoder_mode == "fixed":
            # fixed decoder is in normalized lifted coordinates
            C = self._build_fixed_decoder(dtype=psi_x.dtype, device=psi_x.device)
            C = C * self.psi_scale[None, :]
        else:
            target = (x_next_n - x_n) if self.residual_decode else x_next_n
            C = self.regression_matrix(
                psi_y_s,
                target,
                method=method,
                rank=self.rank,
                ridge=self.decoder_ridge,
            )

        self.K_fitted = K
        self.C_fitted = C
        return K, C

    def predict_one_step(self, K, C, x):
        K = self._to_tensor(K)
        C = self._to_tensor(C)
        x = self._to_tensor(x)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        x_n = self._normalize_x(x)
        psi = self.expand(x_n)
        psi_s = psi / self.psi_scale
        psi_next_s = psi_s @ K.T

        if self.decoder_mode == "fixed":
            x_next_n = psi_next_s @ C.T
        else:
            pred = psi_next_s @ C.T
            x_next_n = x_n + pred if self.residual_decode else pred

        x_next = self._denormalize_x(x_next_n)
        return x_next

    def rollout(self, K, C, x0, steps):
        K = self._to_tensor(K)
        C = self._to_tensor(C)
        x0 = self._to_tensor(x0)

        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)

        traj = [x0[0].clone()]
        x = x0.clone()

        for _ in range(steps):
            x = self.predict_one_step(K, C, x)
            traj.append(x[0].clone())

        return torch.stack(traj, dim=0)

# import torch
# import torch.nn as nn
# import numpy as np
# from itertools import product
# from src.models.expander import ManualExpansion

# import torch
# import numpy as np

# from src.models.expander import ManualExpansion


# class ManualExpansion_ManualDMD(ManualExpansion):
#     """
#     EDMD model built on top of the shared ManualExpansion class.
#     """

#     def __init__(
#         self,
#         state_dim=2,
#         expansion_degree=3,
#         rank=None,
#         ridge=0.0,
#         decoder_mode="fixed",
#         bias=True,
#         sine_cosine_expansion=False,
#         expansion_type="general",
#         system=None,
#     ):
#         super().__init__(
#             state_dim=state_dim,
#             expansion_degree=expansion_degree,
#             bias=bias,
#             sine_cosine_expansion=sine_cosine_expansion,
#             expansion_type=expansion_type,
#             system=system,
#         )

#         self.state_dim = state_dim
#         self.rank = rank
#         self.ridge = ridge
#         self.decoder_mode = decoder_mode

#         if self.decoder_mode not in {"regressed", "fixed"}:
#             raise ValueError("decoder_mode must be 'regressed' or 'fixed'")

#     def _build_fixed_decoder(self, dtype, device):
#         """
#         Build hardcoded decoder C (state_dim, expanded_dim):
#         each state coordinate picks its corresponding linear monomial.
#         """
#         C = torch.zeros(
#             self.state_dim,
#             self.expanded_dim,
#             dtype=dtype,
#             device=device,
#         )

#         generic_names = [f"x{i+1}" for i in range(self.state_dim)]
#         specific_names = ["x", "y", "z", "w", "v", "u"][:self.state_dim]

#         for state_idx in range(self.state_dim):
#             candidates = [generic_names[state_idx], specific_names[state_idx]]

#             feature_idx = None
#             for name in candidates:
#                 if name in self.expand_names:
#                     feature_idx = self.expand_names.index(name)
#                     break

#             if feature_idx is None:
#                 raise ValueError(
#                     f"Could not find state coordinate for index {state_idx} "
#                     f"in expansion names {self.expand_names}"
#                 )

#             C[state_idx, feature_idx] = 1.0

#         return C

#     def regression_matrix(self, x_in, y_out, method="svd", rank=None, ridge=0.0):
#         """
#         Regress matrix M such that y_out ≈ x_in @ M.T.
        
#         Solves:
#             min_M ||y_out - x_in @ M.T||_F
        
#         Uses SVD-based pseudoinverse with optional rank truncation and ridge regularization.

#         Parameters
#         ----------
#         x_in : torch.Tensor (N, k)
#             Input features
#         y_out : torch.Tensor (N, d)
#             Output targets
#         rank : int or None
#             SVD rank truncation for stability
#         ridge : float
#             Ridge regularization to prevent overfitting on small singular values
#         """
#         if isinstance(x_in, np.ndarray):
#             x_in = torch.tensor(x_in, dtype=torch.float64)
#         if isinstance(y_out, np.ndarray):
#             y_out = torch.tensor(y_out, dtype=torch.float64)

#         X = x_in.T
#         Y = y_out.T

#         if method == "svd":
#             U, s, Vt = torch.linalg.svd(X, full_matrices=False)

#             r = len(s) if rank is None else max(1, min(rank, len(s)))
#             U_r = U[:, :r]
#             s_r = s[:r]
#             Vt_r = Vt[:r, :]

#             if ridge > 0.0:
#                 s_inv = s_r / (s_r**2 + ridge)
#             else:
#                 s_inv = 1.0 / s_r

#             M = (Y @ (Vt_r.T * s_inv)) @ U_r.T
#         else:
#             W = torch.linalg.lstsq(X.T, Y.T).solution
#             M = W.T

#         return M

#     def fit(self, x, x_next, method="svd"):
#         """
#         Fit EDMD model to data (x, x_next).

#         Returns
#         -------
#         K : torch.Tensor (expanded_dim, expanded_dim)
#             Koopman operator in lifted space.
#         C : torch.Tensor (state_dim, expanded_dim)
#             Linear decoder from lifted space to original state space.
#         """
#         if isinstance(x, np.ndarray):
#             x = torch.tensor(x, dtype=torch.float64)
#         if isinstance(x_next, np.ndarray):
#             x_next = torch.tensor(x_next, dtype=torch.float64)

#         if x.ndim == 1:
#             x = x.unsqueeze(0)
#         if x_next.ndim == 1:
#             x_next = x_next.unsqueeze(0)

#         psi_x = self.expand(x)
#         psi_y = self.expand(x_next)

#         K = self.regression_matrix(
#             psi_x,
#             psi_y,
#             method=method,
#             rank=self.rank,
#             ridge=self.ridge,
#         )

#         if self.decoder_mode == "fixed":
#             C = self._build_fixed_decoder(dtype=psi_x.dtype, device=psi_x.device)
#         else:
#             C = self.regression_matrix(
#                 psi_x,
#                 x,
#                 method=method,
#                 rank=self.rank,
#                 ridge=self.ridge,
#             )

#         return K, C

#     def rollout(self, K, C, x0, steps):
#         """
#         Roll out the EDMD model from initial state x0 for `steps`.

#         Returns trajectory in original state space with shape (steps+1, state_dim).

#         K maps lifted -> lifted, C maps lifted -> original.
#         """        
#         if isinstance(K, np.ndarray):
#             K = torch.tensor(K, dtype=torch.float64)
#         if isinstance(C, np.ndarray):
#             C = torch.tensor(C, dtype=torch.float64)
#         if isinstance(x0, np.ndarray):
#             x0 = torch.tensor(x0, dtype=torch.float64)

#         if x0.ndim == 1:
#             x0 = x0.unsqueeze(0)

#         K = K.to(dtype=x0.dtype, device=x0.device)
#         C = C.to(dtype=x0.dtype, device=x0.device)

#         psi_current = self.expand(x0)
#         x_current = psi_current @ C.T
#         trajectory = [x_current[0]]

#         for _ in range(steps):
#             psi_next = psi_current @ K.T
#             x_next = psi_next @ C.T
#             trajectory.append(x_next[0])
#             psi_current = psi_next

#         return torch.stack(trajectory, dim=0)