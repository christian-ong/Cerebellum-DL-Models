import torch
import torch.nn as nn
import numpy as np
from src.models.expander import build_expander

class Regression_DMD(nn.Module):
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
        delay_depth=1,
        normalize_state=False,
        normalize_lifted=True,
        rollout_mode="DMD",
        ridge=0.0,
        rank=None,
        eps=1e-12,
        rbf_n_centers=50,
        rbf_center_selection="farthest",
        rbf_bandwidth_mode="knn",
        rbf_knn_k=5,
        hankel_rank=None,
    ):
        super().__init__()

        self.expansion_type = expansion_type
        self.expansion_degree = expansion_degree
        self.state_dim = state_dim
        self.delay_depth = delay_depth
        self.normalize_state = normalize_state
        self.normalize_lifted = normalize_lifted
        self.rollout_mode = rollout_mode
        self.ridge = ridge
        self.rank = rank
        self.eps = eps
        self.hankel_rank = hankel_rank

        self.singular_values_fitted = None
        self.svd_energy_fitted = None
        # -------------------------------------------------
        # Expansion module
        # -------------------------------------------------
        self.expander = build_expander(
            state_dim=state_dim,
            expansion_type=expansion_type,
            expansion_degree=expansion_degree,
            bias=bias,
            sine_cosine_expansion=sine_cosine_expansion,
            system=system,
            rbf_n_centers=rbf_n_centers,
            rbf_center_selection=rbf_center_selection,
            rbf_bandwidth_mode=rbf_bandwidth_mode,
            rbf_knn_k=rbf_knn_k,
            delay_depth=delay_depth,
            hankel_rank=hankel_rank,
        )

        # Convenience aliases so the rest of the code stays readable
        self.expand_names = self.expander.expand_names
        self.state_indices = self.expander.state_indices
        self.expanded_dim = self.expander.expanded_dim

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
    
    def expand(self, x):
        return self.expander.expand(x)

    def de_expand(self, x_expanded):
        return self.expander.de_expand(x_expanded)

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

    def _infer_device(self):
        for tensor in self.parameters(recurse=True):
            return tensor.device

        for tensor in self.buffers(recurse=True):
            return tensor.device

        for attr_name in (
            "x_mean",
            "x_scale",
            "psi_scale",
            "K_fitted",
            "C_fitted",
            "K_tilde_fitted",
            "U_r_fitted",
            "W_reduced_fitted",
            "Lambda_fitted",
            "Phi_lift_fitted",
            "Phi_state_fitted",
            "Phi_pinv_fitted",
        ):
            value = getattr(self, attr_name, None)
            if torch.is_tensor(value):
                return value.device

        expander = getattr(self, "expander", None)
        if expander is not None:
            for tensor in expander.buffers(recurse=True):
                return tensor.device
            for attr_name in ("centers", "sigmas", "state_scale", "mean", "components", "singular_values"):
                value = getattr(expander, attr_name, None)
                if torch.is_tensor(value):
                    return value.device

        return torch.device("cpu")

    def _to_tensor(self, x):
        device = self._infer_device()

        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float64, device=device)
        return x.to(device=device, dtype=torch.float64)
    
    def _validate_delay_rollout_input(self, x, *, caller: str):
        """
        For delay models, rollout must start from a real delay history.
        We deliberately do not allow x(t) to be repeated across the delay window.
        """
        if self.delay_depth <= 1:
            return

        expected = self.state_dim * self.delay_depth

        if x.shape[1] == self.state_dim:
            raise ValueError(
                f"{caller} received only the current state x(t), but this model has "
                f"delay_depth={self.delay_depth}. Pass a full delay history with width "
                f"{expected}: [x(t), x(t-1), ..., x(t-delay_depth+1)]."
            )

        if x.shape[1] != expected:
            raise ValueError(
                f"{caller} expected delay-state width {expected}, got {x.shape[1]}."
            )
    
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
        - When using delay embedding, scale is built from current state, then repeated across delays
        """
        # When using delay embedding, x is stacked [x(t), x(t-1), ...].
        # Build scale from the current state portion x(t), then repeat for all delays.
        if self.delay_depth > 1:
            x_current = x[:, :self.state_dim]
            target_size = x.shape[1]  # Final scale size must match x's full width
        else:
            x_current = x
            target_size = x_current.shape[1]

        if not self.normalize_state:
            return torch.ones(target_size, dtype=x.dtype, device=x.device)

        if any(self._feature_is_trig(name) for name in self.expand_names):
            return torch.ones(target_size, dtype=x.dtype, device=x.device)

        scale_state = self._safe_rms_scale(x_current, dim=0)
        
        # If delay_depth > 1, repeat scale across all delay blocks
        if self.delay_depth > 1:
            scale = scale_state.repeat(self.delay_depth)
        else:
            scale = scale_state
        
        return scale

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
        if self.x_scale is None:
            return x

        n_features = x.shape[-1]
        scale = self.x_scale[:n_features] if self.x_scale.shape[0] > n_features else self.x_scale
        return x / scale

    def _denormalize_x(self, x):
        if self.x_scale is None:
            return x

        n_features = x.shape[-1]
        scale = self.x_scale[:n_features] if self.x_scale.shape[0] > n_features else self.x_scale
        return x * scale

    def _build_fixed_decoder(self):
        """
        Decoder from normalized lifted coordinates z_s back to normalized state x_n.

        Since z_s[idx] = x_n[idx] / psi_scale[idx] for the original state features,
        we need C[row, idx] = psi_scale[idx].
        """
        device = self.psi_scale.device
        C = torch.zeros((self.state_dim, self.expanded_dim), dtype=torch.float64, device=device)
        for i, idx in enumerate(self.state_indices):
            C[i, idx] = self.psi_scale[idx]
        return C
    
    def _build_learned_decoder(self, Z_x, x_n):
        """
        Least-squares decoder from normalized lifted coordinates to normalized current state.

        Needed for expansions such as HankelSVDDelayExpansion where the original state
        is not directly one of the lifted coordinates.
        """
        target = x_n[:, : self.state_dim]              # (N, state_dim)
        sol = torch.linalg.lstsq(Z_x, target).solution # (p, state_dim)
        return sol.T.contiguous()                      # (state_dim, p)
    
    def _solve_modal_coeffs_exact(self, z0):
        """
        Solve z0 ≈ Phi_lift @ b0 for the exact DMD modes.
        Vectorized to support z0 of shape (N, p) -> returns (N, r).
        """
        z0_T = z0.T.to(torch.complex128) # (p, N)
        Phi = self.Phi_lift_fitted.to(torch.complex128) # (p, r)

        # Square full-rank case
        if Phi.shape[0] == Phi.shape[1]:
            try:
                return torch.linalg.solve(Phi, z0_T).T # (N, r)
            except RuntimeError:
                pass

        # General fallback
        return torch.linalg.lstsq(Phi, z0_T).solution.T # (N, r)
    
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

        # 1) state normalization
        self.x_mean = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
        self.x_scale = self._build_state_scale(x)

        x_n = self._normalize_x(x)
        
        # Build symmetric next state for delay embeddings
        if self.delay_depth > 1:
            x_next_n_head = self._normalize_x(x_next)
            # Shift history: [new_step, x_t, x_{t-1}, ...]
            x_next_n = torch.cat([x_next_n_head, x_n[:, :-self.state_dim]], dim=1)
        else:
            x_next_n = self._normalize_x(x_next)

        with torch.no_grad():
            if hasattr(self.expander, "state_scale"):
                self.expander.state_scale.fill_(1.0)
            if hasattr(self.expander, "history_scale"):
                self.expander.history_scale.fill_(1.0)

        if self.expansion_type in {"rbf", "hankel_svd"}:
            fit_device = x_n.device
            self.expander = self.expander.to(fit_device)
            x_n = x_n.to(fit_device)
            x_next_n = x_next_n.to(fit_device)
            self.expander.fit(x_n)
            self.expand_names = self.expander.expand_names
            self.state_indices = self.expander.state_indices
            self.expanded_dim = self.expander.expanded_dim

        # 2) lifted snapshots (Generic for ALL expansion types now!)
        psi_x = self.expand(x_n)       # (N, p)
        psi_y = self.expand(x_next_n)  # (N, p)
        
        if not torch.isfinite(psi_x).all():
            raise ValueError("psi_x contains non-finite values after expansion.")
        if not torch.isfinite(psi_y).all():
            raise ValueError("psi_y contains non-finite values after expansion.")
            
        # 3) lifted normalization
        self.psi_scale = self._build_lifted_scale(psi_x)

        Z_x = psi_x / self.psi_scale   # (N, p)
        Z_y = psi_y / self.psi_scale   # (N, p)

        # 4) reduced exact-DMD / EDMD fit (always square now)
        Xc = Z_x.T   # (p, N)
        Yc = Z_y.T   # (p, N)

        U, s, Vh = torch.linalg.svd(Xc, full_matrices=False)

        # for noise experiments:
        self.singular_values_fitted = s.detach().clone()
        energy = s**2
        self.svd_energy_fitted = torch.cumsum(energy, dim=0) / torch.sum(energy)

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

        # Spectral objects
        Lambda, W_reduced = torch.linalg.eig(K_tilde.to(torch.complex128))
        Phi_lift = (
            Yc.to(torch.complex128)
            @ B.to(torch.complex128)
            @ W_reduced
        )

        # 5) decoder
        if getattr(self.expander, "has_exact_state", True):
            self.C_fitted = self._build_fixed_decoder()
        else:
            self.C_fitted = self._build_learned_decoder(Z_x, x_n)

        # 6) store everything
        self.K_fitted = K_full
        self.K_tilde_fitted = K_tilde
        self.U_r_fitted = U_r
        self.Lambda_fitted = Lambda
        self.W_reduced_fitted = W_reduced
        self.Phi_lift_fitted = Phi_lift
        
        if Phi_lift.ndim > 1:
            C_state = self.C_fitted.to(device=Phi_lift.device, dtype=torch.complex128)
            Phi_state = C_state @ Phi_lift
        else:
            Phi_state = self.C_fitted.to(device=Phi_lift.device, dtype=torch.complex128)
        
        self.Phi_state_fitted = Phi_state
        self.Phi_pinv_fitted = torch.linalg.pinv(Phi_lift) if Phi_lift.ndim > 1 else torch.linalg.pinv(Phi_lift.unsqueeze(1))
        self.Phi_fitted = Phi_lift

        return self.K_fitted, self.C_fitted

    def forward(self, x):
        """Return the next-state prediction using the configured rollout mode."""
        return self.rollout(x, 1)[1]

    def _predict_one_step(self, x):
        x = self._to_tensor(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)

        x_n = self._normalize_x(x)
        z = self.expand(x_n) / self.psi_scale
        lifted_dtype = torch.promote_types(z.dtype, self.K_fitted.dtype)
        z = z.to(dtype=lifted_dtype)
        K = self.K_fitted.to(device=z.device, dtype=lifted_dtype)
        z_next = z @ K.T
        C = self.C_fitted.to(device=z_next.device, dtype=lifted_dtype)
        x_next_n = z_next @ C.T
        
        # C_fitted outputs ONLY the head (state_dim). We must shift it for delay!
        x_next_head = self._denormalize_x(x_next_n)
        if self.delay_depth > 1:
            return torch.cat([x_next_head, x[:, :-self.state_dim]], dim=1)
        return x_next_head

    def _rollout_linear_dynamics(self, x0, steps):
        x0 = self._to_tensor(x0)
        is_1d = x0.ndim == 1
        if is_1d:
            x0 = x0.unsqueeze(0)

        self._validate_delay_rollout_input(x0, caller="_rollout_linear_dynamics")
        x = x0.clone()

        traj = [x[:, :self.state_dim].clone()] # ONLY STORE THE HEAD

        for _ in range(steps):
            x = self._predict_one_step(x)
            traj.append(x[:, :self.state_dim].clone())

        out = torch.stack(traj, dim=0)
        if is_1d:
            out = out.squeeze(1)
        return out
    
    def _rollout_hankel_svd_linear_dynamics(self, x0, steps):
        """
        Rollout for SVD-compressed delay coordinates.

        Important:
        We do NOT decode back to raw delay history and re-encode each step.
        We evolve directly in the fitted Hankel-SVD coordinate space:
            z_{k+1} = K z_k
        and decode only the current physical state for output.
        """
        x0 = self._to_tensor(x0)
        is_1d = x0.ndim == 1

        if is_1d:
            x0 = x0.unsqueeze(0)

        self._validate_delay_rollout_input(x0, caller="_rollout_hankel_svd_linear_dynamics")

        x0_n = self._normalize_x(x0)
        z = self.expand(x0_n) / self.psi_scale
        lifted_dtype = torch.promote_types(z.dtype, self.K_fitted.dtype)
        z = z.to(dtype=lifted_dtype)
        K = self.K_fitted.to(device=z.device, dtype=lifted_dtype)

        traj = [x0[:, :self.state_dim].clone()]

        has_bias_coord = len(self.expand_names) > 0 and self.expand_names[0] == "1"
        C = self.C_fitted.to(device=z.device, dtype=lifted_dtype)

        for _ in range(steps):
            z = z @ K.T

            # If the Hankel expansion includes a constant coordinate, keep it exactly constant.
            if has_bias_coord:
                z[:, 0] = 1.0

            x_next_n = z @ C.T
            x_next = self._denormalize_x(x_next_n)

            traj.append(x_next[:, :self.state_dim].clone())

        out = torch.stack(traj, dim=0)

        if is_1d:
            out = out.squeeze(1)

        return out
    
    def _rollout_DMD(self, x0, steps, mode_indices=None):
        if (self.Lambda_fitted is None or self.Phi_lift_fitted is None or self.C_fitted is None):
            raise ValueError("Missing DMD spectral objects. Call fit() first.")

        x0 = self._to_tensor(x0)
        is_1d = x0.ndim == 1
        if is_1d:
            x0 = x0.unsqueeze(0)

        self._validate_delay_rollout_input(x0, caller="_rollout_DMD")

        x0_n = self._normalize_x(x0)
        z0 = (self.expand(x0_n) / self.psi_scale).to(torch.complex128) # (N, p)

        traj = [x0[:, :self.state_dim].clone()]

        Phi_lift = self.Phi_lift_fitted.to(torch.complex128)
        Lambda = self.Lambda_fitted.to(torch.complex128)
        C = self.C_fitted.to(device=z0.device, dtype=torch.complex128)

        # 1. Get exact modal coordinates using the FULL basis
        b0_full = self._solve_modal_coeffs_exact(z0) # (N, r)
        
        # 2. Apply truncation mask by zeroing out dropped modes
        idx = self._prepare_mode_subset(mode_indices)
        if idx is not None:
            mask = torch.zeros_like(b0_full)
            mask[:, idx] = 1.0
            b0 = b0_full * mask
        else:
            b0 = b0_full

        for k in range(1, steps + 1):
            # Dropped modes stay exactly zero!
            b_k = (Lambda ** k).unsqueeze(0) * b0 # (N, r)
            z_k = (Phi_lift @ b_k.T).T # (N, p)
            
            x_k_n = (C @ z_k.T).T.real.to(torch.float64) # (N, state_dim)
            x_k = self._denormalize_x(x_k_n)
            
            traj.append(x_k.clone())

        out = torch.stack(traj, dim=0)
        if is_1d:
            out = out.squeeze(1)
        return out

    def _rollout_projected_DMD(self, x0, steps, mode_indices=None):
        if (self.Phi_lift_fitted is None or self.Phi_pinv_fitted is None or self.Lambda_fitted is None or self.C_fitted is None):
            raise ValueError("Missing DMD spectral objects. Call fit() first.")

        x0 = self._to_tensor(x0)
        is_1d = x0.ndim == 1
        if is_1d:
            x0 = x0.unsqueeze(0)

        self._validate_delay_rollout_input(x0, caller="_rollout_projected_DMD")
        x = x0.clone()
        traj = [x[:, :self.state_dim].clone()]

        Phi = self.Phi_lift_fitted.to(torch.complex128)
        Lambda = self.Lambda_fitted.to(torch.complex128)
        C = self.C_fitted.to(device=x.device, dtype=torch.complex128)
        Phi_pinv = self.Phi_pinv_fitted.to(torch.complex128)

        # Create mask
        idx = self._prepare_mode_subset(mode_indices)
        if idx is not None:
            mask = torch.zeros(Lambda.shape[0], dtype=Lambda.dtype, device=Lambda.device)
            mask[idx] = 1.0
        else:
            mask = torch.ones(Lambda.shape[0], dtype=Lambda.dtype, device=Lambda.device)

        for _ in range(steps):
            x_n = self._normalize_x(x)
            z = (self.expand(x_n) / self.psi_scale).to(torch.complex128) # (N, p)

            # Extract full coefficients, mask dropped modes
            b = (Phi_pinv @ z.T).T # (N, r)
            b = b * mask 
            
            # Evolve and reconstruct
            z_next = (Phi @ (Lambda.unsqueeze(1) * b.T)).T # (N, p)

            x_next_n = (C @ z_next.T).T.real.to(torch.float64) # (N, state_dim)
            x_next_head = self._denormalize_x(x_next_n)

            traj.append(x_next_head.clone())
            
            # Shift history
            if self.delay_depth > 1:
                x = torch.cat([x_next_head, x[:, :-self.state_dim]], dim=1)
            else:
                x = x_next_head

        out = torch.stack(traj, dim=0)
        if is_1d:
            out = out.squeeze(1)
        return out

    def rollout(self, x0, steps, mode=None, mode_indices=None):
        mode = self._canonical_mode(mode)

        if mode_indices is not None and mode not in {"DMD", "projected_DMD"}:
            raise ValueError("mode_indices are only supported for DMD and projected_DMD rollouts.")

        if mode == "linear_dynamics":
            if self.expansion_type == "hankel_svd":
                return self._rollout_hankel_svd_linear_dynamics(x0, steps)
            return self._rollout_linear_dynamics(x0, steps)
        if mode == "DMD":
            return self._rollout_DMD(x0, steps, mode_indices=mode_indices)
        if mode == "projected_DMD":
            return self._rollout_projected_DMD(x0, steps, mode_indices=mode_indices)

        raise ValueError(f"Unknown rollout mode: {mode}")