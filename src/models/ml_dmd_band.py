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

        # Orthogonality penalty for Phi (keeps Phi reasonably conditioned)
        self.phi_orth_weight = 1e-5

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

        # ------------------------------------------------
        # Rollout (Multi-step) parameters
        # ------------------------------------------------
        self.rollout_horizon = 20
        self.rollout_weight = 0.1

        # ------------------------------------------------
        # Degree-based lifted loss weighting
        # ------------------------------------------------
        degrees = torch.tensor(
            [self._feature_degree(name) for name in self.expand_names],
            dtype=torch.float32,
        )
        weights = 1.0 / (degrees + 1.0)
        self.register_buffer("lift_weights", weights)

    def _advance_z(self, z):
        """Advances the latent state z by one step using the current Phi and Lambda."""
        I_eps = 1e-6 * torch.eye(self.latent_dim, device=self.Phi.device, dtype=self.Phi.dtype)
        Phi_reg = self.Phi + I_eps
        
        b = torch.linalg.solve(Phi_reg, z.T).T
        b_next = b @ self.get_Lambda().T
        z_next = b_next @ self.Phi.T
        
        return torch.clamp(z_next, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)
    
    def _feature_degree(self, name: str) -> int:
        name = name.strip()
        if name == "1": return 0
        if name.startswith("sin(") or name.startswith("cos("): return 1
        parts = name.split("*")
        deg = 0
        for part in parts:
            part = part.strip()
            if "^" in part:
                _, power = part.split("^")
                deg += int(power)
            else:
                deg += 1
        return deg

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
        return self.Phi @ self.get_Lambda() @ torch.linalg.pinv(self.Phi, rcond=1e-6)

    def get_eigenvalues(self):
        return torch.linalg.eigvals(self.get_Lambda())

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
        
        loss_lift = torch.mean(self.lift_weights * (z_next_pred - z_next_true)**2)
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
                    
                    loss_rollout += torch.mean(self.lift_weights * (z_curr_pred - z_true_k)**2)
                
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

        # Group the non-oscillatory (real/nilpotent) manifolds
        dist_real = torch.min(
            torch.stack([dist_uncoupled, dist_jordan_upper, dist_jordan_lower], dim=0), 
            dim=0
        ).values

        # PHYSICS ROUTING: 
        # If b * c < 0, the system learned an oscillation during warm-up. Force a Rotation.
        # If b * c >= 0, the system learned a real dynamic. Force a Jordan or Diagonal block.
        is_oscillatory = (b * c < 0).float()

        # Dynamically apply the correct structural constraint
        loss_manifold = torch.mean(
            is_oscillatory * dist_rot + (1.0 - is_oscillatory) * dist_real
        )

        # Strictly penalize b and c having the same sign (forces them toward 0 to become Jordan)
        loss_same_sign = torch.mean(torch.relu(b * c))

        # Sparsity continuously pushes unused off-diagonals to 0
        loss_sparsity = torch.mean(torch.abs(b)) + torch.mean(torch.abs(c))

        # Phi orthogonality (off-diagonal of Phi^T Phi)
        phi_phys = self.get_Phi()
        G = phi_phys.T @ phi_phys
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        loss_phi_orth = torch.norm(G - I, p='fro') ** 2 / float(G.shape[0] ** 2)

        # --------------------------------------------------
        # Structural Warm-up Schedule
        # --------------------------------------------------
        # Safely get the epoch (defaults to a high number during evaluation)
        current_epoch = getattr(self, "current_epoch", 100)
        
        ramp_start = 20
        ramp_end = 40
        
        if current_epoch < ramp_start:
            struct_weight = 0.0
        elif current_epoch >= ramp_end:
            struct_weight = 1.0
        else:
            struct_weight = (current_epoch - ramp_start) / float(ramp_end - ramp_start)

        # --------------------------------------------------
        # 5. Total Loss Compilation
        # --------------------------------------------------
        loss_total = (
            loss_lift
            + 5.0 * loss_state          
            + self.rollout_weight * loss_rollout 
            + 0.01 * loss_unit_length   
            + (10.0 * struct_weight) * loss_manifold      
            + (10.0 * struct_weight) * loss_same_sign     # <-- Bump this to 10.0 to strictly forbid same-sign off-diagonals 
            + 1e-3 * loss_sparsity                        # <-- Bump this from 1e-5 to 1e-3 to aggressively wipe out the 0.001 noise
            + float(self.phi_orth_weight) * loss_phi_orth
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