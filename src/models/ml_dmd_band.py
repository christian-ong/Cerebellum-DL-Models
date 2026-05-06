import torch
import torch.nn as nn

from src.models.expander import ManualExpansion


class ML_DMD_BAND(ManualExpansion):
    """
    Manual expansion + learned Koopman eigendecomposition.
    Analytical Inverse Version: Computes Phi_inv mathematically during training.
    
    Removed all z_scale logic to focus on raw lifted observables.
    """

    def __init__(
        self,
        state_dim=2,
        expansion_degree=2,
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

        self.latent_dim = self.expanded_dim

        # ------------------------------------------------
        # Structural penalty hyperparameters
        # ------------------------------------------------
        # Smooth activation for rotation-block penalties (act in [0,1])
        self.rotation_act_tol = 1e-3
        self.rotation_act_scale = 1e-3

        # ------------------------------------------------
        # Eigenvector matrix Φ
        # ------------------------------------------------
        self.Phi = nn.Parameter(
            torch.eye(self.latent_dim)
            + 0.01 * torch.randn(self.latent_dim, self.latent_dim)
        )

        # ------------------------------------------------
        # Tridiagonal Eigenvalue Matrix Λ
        # ------------------------------------------------
        self.eig_diag = nn.Parameter(torch.ones(self.latent_dim) + torch.randn(self.latent_dim) * 0.01)
        self.eig_super = nn.Parameter(torch.randn(self.latent_dim - 1) * 0.01)
        self.eig_sub = nn.Parameter(torch.randn(self.latent_dim - 1) * 0.01)

        # ------------------------------------------------
        # Buffers and Constraints
        # ------------------------------------------------
        self.max_abs_z_norm = 1e6
        self.rollout_horizon = 20

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

    def _advance_z(self, z):
        """Advances the latent state z by one step using the current Phi and Lambda."""
        I_eps = 1e-6 * torch.eye(self.latent_dim, device=self.Phi.device, dtype=self.Phi.dtype)
        Phi_reg = self.Phi + I_eps
        
        b = torch.linalg.solve(Phi_reg, z.T).T
        b_next = b @ self.get_Lambda().T
        z_next = b_next @ self.Phi.T
        
        return torch.clamp(z_next, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)
    
    # ------------------------------------------------
    # Forward pass
    # ------------------------------------------------

    def forward(self, x):
        # Directly expand raw state
        z = self.expand(x)
        z = torch.clamp(z, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)

        # Reg for solve
        I_eps = 1e-6 * torch.eye(self.latent_dim, device=self.Phi.device, dtype=self.Phi.dtype)
        Phi_reg = self.Phi + I_eps

        # Project to modal coordinates
        b = torch.linalg.solve(Phi_reg, z.T).T

        # Evolve
        Lambda = self.get_Lambda()
        b_next = b @ Lambda.T

        # Reconstruct
        z_next = b_next @ self.Phi.T

        # Back to state dims (no unscaling needed)
        x_next = self.de_expand(z_next)
        return x_next

    # ------------------------------------------------
    # Training loss
    # ------------------------------------------------

    def compute_loss(self, x, x_next_true, future_x=None):
        # Expand raw inputs directly
        z = self.expand(x)
        z_next_true = self.expand(x_next_true)
        
        z = torch.clamp(z, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)
        z_next_true = torch.clamp(z_next_true, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)

        # 1. One-Step Dynamics
        z_next_pred = self._advance_z(z)
        
        loss_lift = torch.mean((z_next_pred - z_next_true)**2)
        x_next_pred = self.de_expand(z_next_pred)
        loss_state = nn.MSELoss()(x_next_pred, x_next_true)

        # 2. Multi-step Rollout Loss
        loss_rollout = torch.tensor(0.0, device=x.device)
        
        if future_x is not None and future_x.ndim == 3:
            horizon = min(self.rollout_horizon, future_x.shape[1])
            
            if horizon >= 2:
                z_curr_pred = z_next_pred 
                
                for k in range(1, horizon):
                    z_curr_pred = self._advance_z(z_curr_pred)
                    # Remove self.scale_state() here
                    z_true_k = self.expand(future_x[:, k, :])
                    z_true_k = torch.clamp(z_true_k, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)
                    
                    loss_rollout += torch.mean((z_curr_pred - z_true_k)**2)
                
                loss_rollout = loss_rollout / float(horizon - 1)

        # --------------------------------------------------
        # 3. Structural Constraints
        # --------------------------------------------------
        # Apply normalization directly to the PHYSICAL Phi.
        # This keeps the physical matrix from generating explosive values.
        phi_phys = self.get_Phi()
        col_norms = torch.linalg.norm(phi_phys, dim=0)
        loss_unit_length = torch.mean((col_norms - 1.0) ** 2)

        # --------------------------------------------------
        # 4. Lambda Regularization (Physics-Routed Manifold)
        # --------------------------------------------------
        b = self.eig_super
        c = self.eig_sub
        d1 = self.eig_diag[:-1]
        d2 = self.eig_diag[1:]

        # Calculate raw mathematical distances to valid states
        dist_uncoupled = b**2 + c**2
        dist_rot = (b + c)**2 + (d1 - d2)**2
        dist_jordan_upper = c**2 + (d1 - d2)**2
        dist_jordan_lower = b**2 + (d1 - d2)**2

        # FIX: Let the network naturally fall into the closest structural basin
        dist_all = torch.min(
            torch.stack([dist_uncoupled, dist_rot, dist_jordan_upper, dist_jordan_lower], dim=0), 
            dim=0
        ).values
        loss_manifold = torch.mean(dist_all)

        loss_same_sign = torch.mean(torch.relu(b * c))
        loss_sparsity = torch.mean(torch.abs(b)) + torch.mean(torch.abs(c))

        G = phi_phys.T @ phi_phys
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        loss_phi_orth = torch.norm(G - I, p='fro') ** 2 / float(G.shape[0] ** 2)

        # --------------------------------------------------
        # Total Loss Compilation (Using your exact requested weights)
        # --------------------------------------------------
        loss_total = (
            loss_lift
            + 5.0 * loss_state          
            + 1.0 * loss_rollout 
            + 0.01 * loss_unit_length   
            + 5.0 * loss_manifold      
            + 5.0 * loss_same_sign
            + 1e-5 * loss_sparsity
            + 1e-5 * loss_phi_orth
        )
        
        loss_dict = {
            "state": loss_state.item(),
            "lift": loss_lift.item(),
            "rollout": loss_rollout.item(),
            "unit": loss_unit_length.item(),
            "manifold": loss_manifold.item(),
            "same_sign": loss_same_sign.item(),
            "lam_sp": loss_sparsity.item(),
            "phi_orth": loss_phi_orth.item(),
        }
        
        return (loss_total, loss_dict)

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