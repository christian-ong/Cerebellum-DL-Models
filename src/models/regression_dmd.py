import torch
import numpy as np
from src.models.expander import ManualExpansion


class Regression_DMD(ManualExpansion):
    """
    Clean EDMD model with two rollout modes:

    - linear_dynamics:
        direct iteration in normalized lifted space using K_full
    - DMD:
        reduced exact-DMD / exact-EDMD style rollout using
        K_tilde, U_r, W_reduced, Lambda, Phi_lift, Phi_state

    Important:
    - rollout mode is inference-only
    - fit() always computes and stores BOTH representations
    """

    def __init__(
        self,
        state_dim=2,
        expansion_degree=3,
        bias=True,
        sine_cosine_expansion=False,
        expansion_type="general",
        system=None,
        normalize_state=False,
        normalize_lifted=True,
        rollout_mode="DMD",
        ridge=0.0,
        rank=None,
        eps=1e-12,
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
        self.normalize_state = normalize_state
        self.normalize_lifted = normalize_lifted
        self.rollout_mode = rollout_mode
        self.ridge = ridge
        self.rank = rank
        self.eps = eps

        # scaling
        self.x_mean = None
        self.x_scale = None
        self.psi_scale = None

        # direct lifted dynamics
        self.K_fitted = None
        self.C_fitted = None

        # reduced exact-DMD / EDMD objects
        self.K_tilde_fitted = None
        self.U_r_fitted = None
        self.W_reduced_fitted = None
        self.Lambda_fitted = None
        self.Phi_lift_fitted = None
        self.Phi_state_fitted = None

        # optional alias for convenience / backward compatibility
        self.Phi_fitted = None

        self.Phi_pinv_fitted = None

    def _canonical_mode(self, mode):
        mode = self.rollout_mode if mode is None else mode
        aliases = {
            "linear_dynamics": "linear_dynamics",
            "DMD": "DMD",
            "projected_DMD": "projected_DMD",
            "iterative": "linear_dynamics",
            "modal": "DMD",
        }
        if mode not in aliases:
            raise ValueError(f"Unknown rollout mode: {mode}")
        return aliases[mode]

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float64)
        return x.to(dtype=torch.float64)

    def _feature_is_trig(self, name: str) -> bool:
        return ("sin(" in name) or ("cos(" in name)

    def _safe_rms_scale(self, x, dim=0):
        s = torch.sqrt(torch.mean(x**2, dim=dim))
        return torch.clamp(s, min=self.eps)

    def _build_state_scale(self, x):
        """
        Scale-only normalization.
        Important:
        - no mean-centering
        - if trig features are present, do NOT scale state before expansion
        """
        if not self.normalize_state:
            return torch.ones(x.shape[1], dtype=x.dtype, device=x.device)

        if any(self._feature_is_trig(name) for name in self.expand_names):
            return torch.ones(x.shape[1], dtype=x.dtype, device=x.device)

        return self._safe_rms_scale(x, dim=0)

    def _build_lifted_scale(self, psi):
        """
        Normalize only polynomial/state-like lifted coordinates.
        Keep constants and trig features unscaled.
        """
        if not self.normalize_lifted:
            return torch.ones(psi.shape[1], dtype=psi.dtype, device=psi.device)

        rms = self._safe_rms_scale(psi, dim=0)
        scale = torch.ones(psi.shape[1], dtype=psi.dtype, device=psi.device)

        for j, name in enumerate(self.expand_names):
            if name == "1" or self._feature_is_trig(name):
                scale[j] = 1.0
            else:
                scale[j] = rms[j]

        return scale

    def _normalize_x(self, x):
        return x / self.x_scale

    def _denormalize_x(self, x):
        return x * self.x_scale

    def _build_fixed_decoder(self):
        """
        Decoder from normalized lifted coordinates z_s back to normalized state x_n.

        Since z_s[idx] = x_n[idx] / psi_scale[idx] for the original state features,
        we need C[row, idx] = psi_scale[idx].
        """
        C = torch.zeros((self.state_dim, self.expanded_dim), dtype=torch.float64)
        for i, idx in enumerate(self.state_indices):
            C[i, idx] = self.psi_scale[idx]
        return C

    def _solve_modal_coeffs_exact(self, z0):
        """
        Solve z0 ≈ Phi_lift @ b0 for the exact DMD modes.
        This is the right amplitude solve when using
        Phi_lift = Y B W as the modal basis.
        """
        z0 = z0.to(torch.complex128)
        Phi = self.Phi_lift_fitted.to(torch.complex128)

        # Square full-rank case
        if Phi.shape[0] == Phi.shape[1]:
            try:
                return torch.linalg.solve(Phi, z0)
            except RuntimeError:
                pass

        # General fallback
        return torch.linalg.lstsq(Phi, z0.unsqueeze(1)).solution.squeeze(1)
    
    def _prepare_mode_subset(self, mode_indices):
        if mode_indices is None:
            return None

        idx = torch.as_tensor(np.array(mode_indices, copy=True), dtype=torch.long)

        if idx.ndim != 1:
            raise ValueError("mode_indices must be a 1D list/array of indices.")
        if torch.any(idx < 0):
            raise ValueError("mode_indices must be nonnegative.")

        return idx
    
    def fit(self, x, x_next):
        x = self._to_tensor(x)
        x_next = self._to_tensor(x_next)

        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x_next.ndim == 1:
            x_next = x_next.unsqueeze(0)

        # -------------------------------------------------
        # 1) state normalization
        # -------------------------------------------------
        self.x_mean = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
        self.x_scale = self._build_state_scale(x)

        x_n = self._normalize_x(x)
        x_next_n = self._normalize_x(x_next)

        # -------------------------------------------------
        # 2) lifted snapshots
        # -------------------------------------------------
        psi_x = self.expand(x_n)       # (N, p)
        psi_y = self.expand(x_next_n)  # (N, p)

        # -------------------------------------------------
        # 3) lifted normalization
        # -------------------------------------------------
        self.psi_scale = self._build_lifted_scale(psi_x)

        Z_x = psi_x / self.psi_scale   # (N, p)
        Z_y = psi_y / self.psi_scale   # (N, p)

        # -------------------------------------------------
        # 4) reduced exact-DMD / EDMD fit
        # -------------------------------------------------
        Xc = Z_x.T   # (p, N)
        Yc = Z_y.T   # (p, N)

        U, s, Vh = torch.linalg.svd(Xc, full_matrices=False)

        r = len(s) if self.rank is None else max(1, min(int(self.rank), len(s)))
        U_r = U[:, :r]
        s_r = s[:r]
        Vh_r = Vh[:r, :]

        if self.ridge > 0.0:
            s_inv = s_r / (s_r**2 + self.ridge)
        else:
            s_inv = 1.0 / torch.clamp(s_r, min=self.eps)

        B = Vh_r.T * s_inv                        # (N, r)
        K_tilde = (U_r.T @ Yc) @ B               # (r, r)
        K_full = (Yc @ B) @ U_r.T                # (p, p)

        # -------------------------------------------------
        # 5) fixed decoder
        # -------------------------------------------------
        self.C_fitted = self._build_fixed_decoder()

        # -------------------------------------------------
        # 6) spectral objects for reduced exact-DMD rollout
        # -------------------------------------------------
        Lambda, W_reduced = torch.linalg.eig(K_tilde.to(torch.complex128))
        Phi_lift = (
            Yc.to(torch.complex128)
            @ B.to(torch.complex128)
            @ W_reduced
        )
        Phi_state = self.C_fitted.to(torch.complex128) @ Phi_lift

        # -------------------------------------------------
        # 7) store everything
        # -------------------------------------------------
        self.K_fitted = K_full
        self.K_tilde_fitted = K_tilde
        self.U_r_fitted = U_r
        self.W_reduced_fitted = W_reduced
        self.Lambda_fitted = Lambda
        self.Phi_lift_fitted = Phi_lift
        self.Phi_state_fitted = Phi_state
        self.Phi_pinv_fitted = torch.linalg.pinv(Phi_lift)

        # alias
        self.Phi_fitted = Phi_lift

        return self.K_fitted, self.C_fitted

    def _predict_one_step(self, x):
        x = self._to_tensor(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)

        x_n = self._normalize_x(x)
        z = self.expand(x_n) / self.psi_scale
        z_next = z @ self.K_fitted.T
        x_next_n = z_next @ self.C_fitted.T
        return self._denormalize_x(x_next_n)

    def _rollout_linear_dynamics(self, x0, steps):
        x0 = self._to_tensor(x0)
        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)

        traj = [x0[0].clone()]
        x = x0.clone()

        for _ in range(steps):
            x = self._predict_one_step(x)
            traj.append(x[0].clone())

        return torch.stack(traj, dim=0)

    def _rollout_DMD(self, x0, steps, mode_indices=None):  # lambda^k step
        if (
            self.Lambda_fitted is None
            or self.Phi_lift_fitted is None
            or self.C_fitted is None
        ):
            raise ValueError("Missing DMD spectral objects. Call fit() first.")

        x0 = self._to_tensor(x0)
        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)

        x0_n = self._normalize_x(x0)
        z0 = (self.expand(x0_n) / self.psi_scale)[0].to(torch.complex128)

        traj = [x0[0].clone()]

        Phi_lift = self.Phi_lift_fitted.to(torch.complex128)
        Lambda = self.Lambda_fitted.to(torch.complex128)
        C = self.C_fitted.to(torch.complex128)

        idx = self._prepare_mode_subset(mode_indices)
        if idx is not None:
            Phi_lift = Phi_lift[:, idx]
            Lambda = Lambda[idx]
            b0 = torch.linalg.pinv(Phi_lift) @ z0
        else:
            b0 = self._solve_modal_coeffs_exact(z0)

        for k in range(1, steps + 1):
            z_k = Phi_lift @ ((Lambda ** k) * b0)
            x_k_n = C @ z_k
            x_k = self._denormalize_x(x_k_n.real.to(torch.float64))
            traj.append(x_k)

        return torch.stack(traj, dim=0)

    def _rollout_projected_DMD(self, x0, steps, mode_indices=None):  # phi lambda phi^-1 step
        if (
            self.Phi_lift_fitted is None
            or self.Phi_pinv_fitted is None
            or self.Lambda_fitted is None
            or self.C_fitted is None
        ):
            raise ValueError("Missing DMD spectral objects. Call fit() first.")

        x0 = self._to_tensor(x0)
        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)

        traj = [x0[0].clone()]

        Phi = self.Phi_lift_fitted.to(torch.complex128)
        Lambda = self.Lambda_fitted.to(torch.complex128)
        C = self.C_fitted.to(torch.complex128)

        idx = self._prepare_mode_subset(mode_indices)
        if idx is not None:
            Phi = Phi[:, idx]
            Lambda = Lambda[idx]
            Phi_pinv = torch.linalg.pinv(Phi)
        else:
            Phi_pinv = self.Phi_pinv_fitted.to(torch.complex128)

        x = x0.clone()

        for _ in range(steps):
            x_n = self._normalize_x(x)
            z = (self.expand(x_n) / self.psi_scale)[0].to(torch.complex128)

            b = Phi_pinv @ z
            z_next = Phi @ (Lambda * b)

            x_next_n = C @ z_next
            x_next = self._denormalize_x(x_next_n.real.to(torch.float64)).unsqueeze(0)

            traj.append(x_next[0].clone())
            x = x_next

        return torch.stack(traj, dim=0)

    def rollout(self, x0, steps, mode=None, mode_indices=None):
        mode = self._canonical_mode(mode)

        if mode_indices is not None and mode not in {"DMD", "projected_DMD"}:
            raise ValueError("mode_indices are only supported for DMD and projected_DMD rollouts.")

        if mode == "linear_dynamics":
            return self._rollout_linear_dynamics(x0, steps)
        if mode == "DMD":
            return self._rollout_DMD(x0, steps, mode_indices=mode_indices)
        if mode == "projected_DMD":
            return self._rollout_projected_DMD(x0, steps, mode_indices=mode_indices)

        raise ValueError(f"Unknown rollout mode: {mode}")