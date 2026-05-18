import torch
import torch.nn as nn

from src.models.expander import build_expander

class ML_DMD_L1(nn.Module):
    def __init__(
        self,
        state_dim=2,
        expansion_degree=2,
        bias=True,
        sine_cosine_expansion=False,
        expansion_type="general",
        system=None,
        rbf_n_centers=50,
        rbf_center_selection="farthest",
        rbf_bandwidth_mode="knn",
        rbf_knn_k=5,
        l1_weight=1e-6,
    ):

        super().__init__()

        self.state_dim = state_dim
        self.expansion_type = expansion_type

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
        )

        # Public aliases used elsewhere in the model / training code
        self.expand_names = self.expander.expand_names
        self.state_indices = self.expander.state_indices
        self.expanded_dim = self.expander.expanded_dim

        self.latent_dim = self.expanded_dim

        # ------------------------------------------------
        # Fixed lifted-feature scaling (dataset-level stats, not batch stats)
        # ------------------------------------------------
        self.register_buffer("lift_scale", torch.ones(self.latent_dim))
        self.lift_norm_eps = 1e-6

        # ------------------------------------------------
        # Eigenvector matrix Φ
        # ------------------------------------------------
        # Columns correspond to Koopman modes.
        # Initialized close to identity for stability.
        self.Phi = nn.Parameter(
            torch.eye(self.latent_dim)
            + 0.001 * torch.randn(self.latent_dim, self.latent_dim)
        )

        # ------------------------------------------------
        # Eigenvalue matrix Λ
        # ------------------------------------------------
        # We allow Λ to be a full matrix instead of diagonal.
        # This allows the model to represent complex eigenvalue
        # pairs using real-valued 2×2 blocks.
        self.Lambda = nn.Parameter(
            torch.eye(self.latent_dim)
            + 0.001 * torch.randn(self.latent_dim, self.latent_dim)
        )

        # ------------------------------------------------
        # Feature scaling buffer
        # ------------------------------------------------
        self.max_abs_z_norm = 1e6
        self.rollout_horizon = 20
        self.l1_weight = l1_weight  # Weight for the L1 sparsity penalty (tune as needed)

    def _normalize(self, z):
        return z / self.lift_scale

    def _unnormalize(self, z_norm):
        return z_norm * self.lift_scale

    # ------------------------------------------------
    # Set lifted scaling
    # ------------------------------------------------

    def get_Phi(self):
        """Return Phi in physical coordinates."""
        return self.Phi

    def get_Phi_inv(self):
        """Return pseudo-inverse of Phi."""
        return torch.linalg.pinv(self.Phi, rcond=1e-6) 

    def get_Lambda(self):
        """Return the learned Lambda matrix."""
        return self.Lambda

    def get_K(self):
        """Return the lifted Koopman operator: K = Phi Lambda Phi^{-1}"""
        Phi_inv = self.get_Phi_inv()
        return self.Phi @ self.Lambda @ Phi_inv

    def get_eigenvalues(self):
        """Eigenvalues of the lifted Koopman operator."""
        K = self.get_K()
        return torch.linalg.eigvals(K)

    # Standardize these three helpers in ML_DMD_FREE
    def _get_modal_coords(self, z):
        I_eps = 1e-6 * torch.eye(self.latent_dim, device=self.Phi.device)
        return torch.linalg.solve(self.Phi + I_eps, z.mT).mT

    def _step_modal(self, b):
        return b @ self.Lambda.mT

    def _modal_to_latent(self, b):
        z = b @ self.Phi.mT
        return torch.clamp(z, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)

    # ------------------------------------------------
    # Forward pass
    # ------------------------------------------------

    def forward(self, x):
        """
        One-step prediction

            x_t → x_{t+1}

        using the Koopman eigendecomposition.
        """
        z = self.expander.expand(x)
        z_norm = self._normalize(z)
        b = self._get_modal_coords(z_norm)
        b_next = self._step_modal(b)
        z_next = self._modal_to_latent(b_next)
        z_next_physical = self._unnormalize(z_next)
        return self.expander.de_expand(z_next_physical)

    # ------------------------------------------------
    # Training loss
    # ------------------------------------------------

    def compute_loss(self, x, x_next_true, future_x=None):
        z_raw = self.expander.expand(x)
        z_next_true_raw = self.expander.expand(x_next_true)

        z_norm = self._normalize(z_raw)
        z_next_true_norm = self._normalize(z_next_true_raw)

        # Use helper for consistency
        b_curr = self._get_modal_coords(z_norm)
        b_next = self._step_modal(b_curr)
        z_next_pred = self._modal_to_latent(b_next)

        loss_lift = torch.mean((z_next_pred - z_next_true_norm)**2)
        loss_state = nn.MSELoss()(self.expander.de_expand(self._unnormalize(z_next_pred)), x_next_true)

        loss_rollout = torch.tensor(0.0, device=x.device)
        if future_x is not None and future_x.ndim == 3:
            horizon = min(self.rollout_horizon, future_x.shape[1])
            if horizon >= 2:
                # PRE-EXPAND (Saves massive time!)
                z_targets = self.expander.expand(future_x.reshape(-1, self.state_dim))
                z_targets = z_targets.reshape(x.shape[0], future_x.shape[1], -1)
                z_targets_norm = self._normalize(z_targets.reshape(-1, self.latent_dim))
                z_targets_norm = z_targets_norm.reshape_as(z_targets)

                b_rollout = b_next 
                for k in range(1, horizon):
                    b_rollout = self._step_modal(b_rollout)
                    z_pred_k = self._modal_to_latent(b_rollout)
                    z_pred_k_phys = self._unnormalize(z_pred_k)
                    loss_rollout += torch.mean((z_pred_k - z_targets_norm[:, k, :])**2)
                    loss_rollout += torch.mean((self.expander.de_expand(z_pred_k_phys) - future_x[:, k, :])**2)

                loss_rollout /= (horizon - 1)

        # Structural constraints (keep existing code)
        phi_phys = self.get_Phi()
        col_norms = torch.linalg.norm(phi_phys, dim=0)
        loss_unit_length = torch.mean((col_norms - 1.0) ** 2)

        # --- NEW: Off-Diagonal L1 Sparsity Penalty ---
        # Get the full Lambda matrix
        lam = self.get_Lambda() if hasattr(self, "get_Lambda") else self.Lambda
        
        # Create a boolean mask for everything EXCEPT the main diagonal
        off_diag_mask = ~torch.eye(self.latent_dim, dtype=torch.bool, device=lam.device)
        
        # Sum the absolute values of the off-diagonal elements
        loss_sparsity = torch.sum(torch.abs(lam[off_diag_mask]))
        # ---------------------------------------------

        # Total loss
        # Note: You can tune the 1e-3 weight on loss_sparsity. 
        # If the matrix stays too dense, increase it. If the physics break, decrease it.
        loss = (
              1.0 * loss_state 
            + 1.0 * loss_rollout
            + 0.1 * loss_lift 
            + 1e-3 * loss_unit_length
            + self.l1_weight * loss_sparsity  # Add the sparsity penalty
        )

        loss_dict = {
            "lift": loss_lift.item(),
            "state": loss_state.item(),
            "rollout": loss_rollout.item(),
            "unit": loss_unit_length.item(),
            "sparsity": loss_sparsity.item(), # Track it!
        }

        return (loss, loss_dict)
    
    # ------------------------------------------------
    # Rollout simulation
    # ------------------------------------------------

    def rollout(self, x0, steps):
        if not torch.is_tensor(x0):
            x0 = torch.tensor(
                x0,
                dtype=next(self.parameters()).dtype,
                device=next(self.parameters()).device,
            )

        if x0.ndim == 1:
            x = x0.unsqueeze(0)
        else:
            x = x0

        traj = [x.squeeze(0)]
        
        # 1. Expand the state to latent space exactly ONCE
        z = self.expander.expand(x)
        z_norm = self._normalize(z)
        b = self._get_modal_coords(z_norm) # SOLVE ONCE
        for _ in range(steps):
            b = self._step_modal(b)   # MATMUL LOOP
            z = self._modal_to_latent(b)
            z_phys = self._unnormalize(z) # MUST unnormalize before de-expanding
            traj.append(self.expander.de_expand(z_phys).squeeze(0))
        return torch.stack(traj)