import torch
import torch.nn as nn

from src.models.expander import build_expander


class ML_DMD_BAND(nn.Module):
    def __init__(
        self,
        state_dim=2,
        expansion_degree=2,
        bias=True,
        sine_cosine_expansion=False,
        expansion_type="general",
        system=None,
        delay_depth=1,
        hankel_rank=None,
        rbf_n_centers=50,
        rbf_center_selection="farthest",
        rbf_bandwidth_mode="knn",
        rbf_knn_k=5,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.expansion_type = expansion_type
        self.delay_depth = int(delay_depth)
        self.hankel_rank = hankel_rank
        # ------------------------------------------------
        # Initialize basis expansion
        # ------------------------------------------------
        # This creates the lifted state representation z.
        # The expansion can now be either:
        #   - ManualExpansion  (general / specific)
        #   - RBFExpansion     (rbf)
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

        # Public aliases used elsewhere in the model / training code
        self.expand_names = self.expander.expand_names
        self.state_indices = self.expander.state_indices
        self.expanded_dim = self.expander.expanded_dim

        self.latent_dim = self.expanded_dim

        # ------------------------------------------------
        # Structural penalty hyperparameters
        # ------------------------------------------------
        # Smooth activation for rotation-block penalties (act in [0,1])
        self.rotation_act_tol = 1e-3
        self.rotation_act_scale = 1e-3

        # ------------------------------------------------
        # Fixed lifted-feature scaling (dataset-level stats, not batch stats)
        # ------------------------------------------------
        self.register_buffer("lift_mean", torch.zeros(self.latent_dim))
        self.register_buffer("lift_scale", torch.ones(self.latent_dim))
        self.lift_norm_eps = 1e-6

        # ------------------------------------------------
        # Eigenvector matrix Φ
        # ------------------------------------------------
        self.Phi = nn.Parameter(
            torch.eye(self.latent_dim)
            + 0.001 * torch.randn(self.latent_dim, self.latent_dim)
        )

        # ------------------------------------------------
        # Tridiagonal Eigenvalue Matrix Λ
        # ------------------------------------------------
        self.eig_diag = nn.Parameter(torch.ones(self.latent_dim) + torch.randn(self.latent_dim) * 0.01)
        
        # Initialize cleanly into the rotation basin (perfectly anti-symmetric)
        super_init = torch.randn(self.latent_dim - 1) * 0.01
        self.eig_super = nn.Parameter(super_init)
        self.eig_sub = nn.Parameter(-super_init.clone().detach())

        # ------------------------------------------------
        # Buffers and Constraints
        # ------------------------------------------------
        self.max_abs_z_norm = 1e6
        self.rollout_horizon = 20
        self.current_epoch = 0

    def set_lifted_normalization_stats(self, mean, scale):
        mean = torch.as_tensor(mean, dtype=self.lift_mean.dtype, device=self.lift_mean.device)
        scale = torch.as_tensor(scale, dtype=self.lift_scale.dtype, device=self.lift_scale.device)
        self.lift_mean.copy_(mean)
        self.lift_scale.copy_(torch.clamp(scale, min=self.lift_norm_eps))

    def _normalize(self, z, update_stats=True):
        del update_stats
        return z / self.lift_scale

    def _unnormalize(self, z_norm):
        return z_norm * self.lift_scale

    # ------------------------------------------------
    # Matrix Accessors
    # ------------------------------------------------

    def get_Phi(self):
        """Returns Physical Phi for visualization and analysis."""
        return self.Phi

    def get_Phi_inv(self):
        """Returns Physical Phi_inv."""
        return torch.linalg.pinv(self.Phi, rcond=1e-6)

    def get_Lambda(self):
        Lambda = (
            torch.diag(self.eig_diag) + 
            torch.diag(self.eig_super, diagonal=1) + 
            torch.diag(self.eig_sub, diagonal=-1)
        )
        return Lambda

    def get_K(self):
        """Returns Physical K."""
        return self.Phi @ self.get_Lambda() @ self.get_Phi_inv()

    def get_eigenvalues(self):
        return torch.linalg.eigvals(self.get_Lambda())

    def _get_modal_coords(self, z):
        I_eps = 1e-6 * torch.eye(self.latent_dim, device=self.Phi.device)
        return torch.linalg.solve(self.Phi + I_eps, z.mT).mT

    def _step_modal(self, b):
        # FIX: Call the method get_Lambda()
        return b @ self.get_Lambda().mT

    def _modal_to_latent(self, b):
        z = b @ self.Phi.mT
        return torch.clamp(z, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)

    # ------------------------------------------------
    # Forward pass
    # ------------------------------------------------
    def forward(self, x):
        z = self.expander.expand(x)
        z_norm = self._normalize(z)
        b = self._get_modal_coords(z_norm)
        b_next = self._step_modal(b)
        # Use your helper to get clamping for free!
        z_next = self._modal_to_latent(b_next) 
        z_next_physical = self._unnormalize(z_next)
        return self.expander.de_expand(z_next_physical)

    # ------------------------------------------------
    # Training loss
    # ------------------------------------------------
    def compute_loss(self, x, x_next_true, future_x=None):
        """
        Refactored loss function for ML_DMD_BAND.
        Prioritizes state accuracy while using soft, continuous regularization 
        to encourage clean Koopman modes without shattering gradients.
        """
        # ------------------------------------------------------------------
        # 1. Data Preparation and Lifting
        # ------------------------------------------------------------------
        z_raw = self.expander.expand(x)
        z_next_true_raw = self.expander.expand(x_next_true)

        # Use fixed scaling (calibrated from dataset stats)
        z_norm = self._normalize(z_raw)
        z_next_true_norm = self._normalize(z_next_true_raw, update_stats=False)

        # ------------------------------------------------------------------
        # 2. Prediction Step
        # ------------------------------------------------------------------
        b_curr = self._get_modal_coords(z_norm)
        b_next = self._step_modal(b_curr)
        z_next_pred_norm = self._modal_to_latent(b_next)

        # ------------------------------------------------------------------
        # 3. Primary Accuracy Losses (The anchor of the model)
        # ------------------------------------------------------------------
        loss_lift = torch.mean((z_next_pred_norm - z_next_true_norm)**2)
        z_next_phys = self._unnormalize(z_next_pred_norm)
        
        # Physical state loss remains our absolute highest priority
        loss_state = torch.mean((self.expander.de_expand(z_next_phys) - x_next_true)**2)

        # ------------------------------------------------------------------
        # 4. Multi-step Rollout Loss
        # ------------------------------------------------------------------
        loss_rollout = torch.tensor(0.0, device=x.device)
        if future_x is not None and future_x.ndim == 3:
            horizon = min(self.rollout_horizon, future_x.shape[1])
            if horizon >= 2:
                z_targets = self.expander.expand(future_x.reshape(-1, self.state_dim))
                z_targets = z_targets.reshape(x.shape[0], future_x.shape[1], -1)
                z_targets_norm = self._normalize(z_targets.reshape(-1, self.latent_dim), update_stats=False)
                z_targets_norm = z_targets_norm.reshape_as(z_targets)

                b_rollout = b_next
                for k in range(1, horizon):
                    b_rollout = self._step_modal(b_rollout)
                    z_pred_k = self._modal_to_latent(b_rollout)
                    z_pred_k_phys = self._unnormalize(z_pred_k)
                    
                    # Compute errors with a slight decay to prioritize near-term physics
                    discount = 0.9 ** k
                    loss_rollout += discount * torch.mean((z_pred_k - z_targets_norm[:, k, :])**2)
                    loss_rollout += discount * torch.mean((self.expander.de_expand(z_pred_k_phys) - future_x[:, k, :])**2)
                
                loss_rollout = loss_rollout / float(horizon - 1)

        # ------------------------------------------------------------------
        # 5. Phi Constraints (Eigenvectors)
        # ------------------------------------------------------------------
        phi_phys = self.get_Phi()
        col_norms = torch.linalg.norm(phi_phys, dim=0)
        
        # Keeping Phi columns near unit length prevents scaling issues with Phi_inv
        loss_unit_length = torch.mean((col_norms - 1.0) ** 2)

        # ------------------------------------------------------------------
        # 6. Lambda Constraints (Dynamics)
        # ------------------------------------------------------------------
        eigs = self.get_eigenvalues()
        
        # Soft stability: Only penalize eigenvalues that go beyond 1.0
        # (Allows strict rotation / conservation of energy)
        loss_stability = torch.mean(torch.nn.functional.relu(torch.abs(eigs) - 1.0))

        b = self.eig_super  # Upper diagonal
        c = self.eig_sub    # Lower diagonal
        d = self.eig_diag   # Main diagonal

        # REWRITTEN STRUCTURAL LOSS:
        # Instead of a harsh min() manifold, we apply a gentle L1 pressure to the off-diagonals.
        # We also add a penalty pushing (b + c) towards 0, which smoothly encourages 
        # skew-symmetry (rotations) or zero (independent modes) without hard constraints.
        loss_lam_sparse = torch.mean(torch.abs(b)) + torch.mean(torch.abs(c))
        loss_skew_sym = torch.mean(torch.abs(b + c)) 

        # ------------------------------------------------------------------
        # 8. Total Loss Assembly
        # ------------------------------------------------------------------
        loss_total = (
            2.0 * loss_state                   # Anchor: Must predict the physical state
            + 2.0 * loss_rollout               # Anchor: Must be stable over time
            + 0.1 * loss_lift                  # Latent guidance
            + 1e-3 * loss_unit_length          # Keep matrices numerically stable
            + 1e-2 * loss_stability            # Prevent explosive dynamics
            + 1e-3 * loss_lam_sparse          # Encourage simpler dynamics
            + 1e-3 * loss_skew_sym             # Encourage clean rotations / independent modes
            
        )
        
        loss_dict = {
            "state": loss_state.item(),
            "lift": loss_lift.item(),
            "rollout": loss_rollout.item(),
            "unit": loss_unit_length.item(),
            "stability": loss_stability.item(),
            "lam_sparse": loss_lam_sparse.item(),
            "skew_sym": loss_skew_sym.item(),
        }
        return (loss_total, loss_dict)

    def rollout(self, x0, steps):
        if not torch.is_tensor(x0):
            x0 = torch.tensor(
                x0,
                dtype=next(self.parameters()).dtype,
                device=next(self.parameters()).device,
            )

        is_1d = x0.ndim == 1
        if is_1d:
            x0 = x0.unsqueeze(0)

        delay_depth = int(getattr(self.expander, "delay_depth", 1))
        expected_width = self.state_dim * delay_depth

        if delay_depth > 1:
            if x0.shape[1] == self.state_dim:
                raise ValueError(
                    f"{self.__class__.__name__}.rollout received only the current state, "
                    f"but delay_depth={delay_depth}. Pass a full delay history with width "
                    f"{expected_width}: [x(t), x(t-1), ..., x(t-q+1)]."
                )

            if x0.shape[1] != expected_width:
                raise ValueError(
                    f"{self.__class__.__name__}.rollout expected delay-state width "
                    f"{expected_width}, got {x0.shape[1]}."
                )

        x = x0

        traj = [x.squeeze(0)]
        
        # 1. Expand the state to latent space exactly ONCE
        z = self.expander.expand(x)
        z_norm = self._normalize(z, update_stats=False)
        b = self._get_modal_coords(z_norm)
        for _ in range(steps):
            b = self._step_modal(b)   # MATMUL LOOP
            z = self._modal_to_latent(b)
            z_phys = self._unnormalize(z) # MUST unnormalize before de-expanding
            traj.append(self.expander.de_expand(z_phys).squeeze(0))
        return torch.stack(traj)