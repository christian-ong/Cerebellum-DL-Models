import torch
import torch.nn as nn

from src.models.expander import build_expander

class ML_DMD(nn.Module):
    """
    Manual expansion + learned Koopman eigendecomposition.

    Instead of learning the lifted linear operator K directly, we learn

        K = Phi Λ Phi^{-1}

    but we NEVER explicitly construct K.

    The lifted dynamics are applied as

        z_{t+1} = Phi Λ Phi^{-1} z_t

    which corresponds to:

        z_norm -> modal coordinates -> modal evolution -> lifted coordinates

    Pipeline:

        x  --standardize-->  x_scaled
        x_scaled --expand-->  z
        z  --scale-->   z_norm
        z_norm --Phi^{-1}--> modal coordinates b
        b --Lambda--> modal_next
        modal_next --Phi--> z_norm_next
        z_norm_next --descale--> z_next
        z_next --de_expand--> x_next_scaled
        x_next_scaled --unscale--> x_next

    Important design choices
    ------------------------
    • Polynomial / manual basis expansion
    • Fixed feature normalization (z_scale)
    • Degree-weighted lifted loss
    • Regularization for Phi conditioning
    • Eigenvalue stability regularization
    """

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

    def _advance_z(self, z):
        I_eps = 1e-6 * torch.eye(self.latent_dim, device=self.Phi.device, dtype=self.Phi.dtype)
        Phi_reg = self.Phi + I_eps
        b = torch.linalg.solve(Phi_reg, z.mT).mT
        b_next = b @ self.Lambda.mT
        z_next = b_next @ self.Phi.mT
        return torch.clamp(z_next, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)

    # ------------------------------------------------
    # Forward pass
    # ------------------------------------------------

    def forward(self, x):
        """
        One-step prediction

            x_t → x_{t+1}

        using the Koopman eigendecomposition.
        """

        # Lift state into observable space
        z_raw = self.expander.expand(x)

        # Normalize lifted coordinates
        z = torch.clamp(z_raw, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)

        I_eps = 1e-6 * torch.eye(self.latent_dim, device=self.Phi.device, dtype=self.Phi.dtype)
        Phi_reg = self.Phi + I_eps
        cond_number = torch.linalg.cond(self.Phi)
        if cond_number > 1e6:
            print(f"Warning: High condition number for Phi: {cond_number:.2e}")

        # Convert to modal coordinates
        b = torch.linalg.solve(Phi_reg, z.mT).mT

        # Modal evolution
        b_next = b @ self.Lambda.mT

        # Convert back to lifted coordinates
        z_next = b_next @ self.Phi.mT

        # Recover original state
        x_next = self.expander.de_expand(z_next)
        return x_next

    # ------------------------------------------------
    # Training loss
    # ------------------------------------------------

    def compute_loss(self, x, x_next_true, future_x=None):
        z = self.expander.expand(x)
        z_next_true = self.expander.expand(x_next_true)

        z_next_pred = self._advance_z(z)

        # 1) Lifted loss
        diff = z_next_pred - z_next_true
        loss_lift = torch.mean(diff**2)

        # 2) State prediction loss
        x_next_pred = self.expander.de_expand(z_next_pred)
        loss_state = nn.MSELoss()(x_next_pred, x_next_true)

        # 3) Multi-step Rollout Loss
        loss_rollout = torch.tensor(0.0, device=x.device)
        if future_x is not None and future_x.ndim == 3:
            horizon = min(self.rollout_horizon, future_x.shape[1])
            if horizon >= 2:
                z_curr_pred = z_next_pred 
                for k in range(1, horizon):
                    z_curr_pred = self._advance_z(z_curr_pred)
                    z_true_k = self.expander.expand(future_x[:, k, :])
                    z_true_k = torch.clamp(z_true_k, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)
                    loss_rollout += torch.mean((z_curr_pred - z_true_k)**2)
                loss_rollout = loss_rollout / float(horizon - 1)

        # 4) Φ conditioning regularization
        col_norms = torch.linalg.norm(self.Phi, dim=0)
        loss_unit_length = torch.mean((col_norms - 1.0) ** 2)

        # Total loss
        loss = (
            loss_lift 
            + 0.5 * loss_state 
            + 0.1 * loss_rollout 
            + 1e-3 * loss_unit_length
        )

        loss_dict = {
            "lift": loss_lift.item(),
            "state": loss_state.item(),
            "rollout": loss_rollout.item(),
            "unit": loss_unit_length.item(),
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

        for _ in range(steps):
            x = self.forward(x)
            traj.append(x.squeeze(0))

        return torch.stack(traj)