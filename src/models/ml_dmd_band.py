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

        # Orthogonality penalty weight for Phi (keeps Phi well-conditioned)
        self.phi_orth_weight = 1e-2

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
        self.register_buffer("x_mean", torch.zeros(state_dim))
        self.register_buffer("x_scale", torch.ones(state_dim))
        self.register_buffer("z_scale", torch.ones(self.latent_dim)) # <--- Add this back
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
        
    def set_state_scale(self, x_mean, x_scale):
        if not torch.is_tensor(x_mean):
            x_mean = torch.tensor(x_mean, dtype=torch.float32)
        if not torch.is_tensor(x_scale):
            x_scale = torch.tensor(x_scale, dtype=torch.float32)

        self.x_mean.copy_(x_mean.to(self.x_mean.device))
        self.x_scale.copy_(torch.clamp(x_scale.to(self.x_scale.device), min=1e-6))
    
    def set_z_scale(self, z_scale):
        if not torch.is_tensor(z_scale):
            z_scale = torch.tensor(z_scale, dtype=torch.float32)
        self.z_scale.copy_(torch.clamp(z_scale.to(self.z_scale.device), min=1e-6))

    def _advance_z(self, z):
        """Advances the latent state z by one step using the current Phi and Lambda."""
        I_eps = 1e-6 * torch.eye(self.latent_dim, device=self.Phi.device, dtype=self.Phi.dtype)
        Phi_reg = self.Phi + I_eps
        
        b = torch.linalg.solve(Phi_reg, z.T).T
        b_next = b @ self.get_Lambda().T
        z_next = b_next @ self.Phi.T
        
        return torch.clamp(z_next, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)
    
    def scale_state(self, x):
        return (x - self.x_mean) / self.x_scale    
    
    def unscale_state(self, x):
        return x * self.x_scale + self.x_mean
    
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
        """Returns descaled Physical Phi for visualization and analysis."""
        return torch.diag(self.z_scale) @ self.Phi

    def get_Phi_inv(self):
        """Returns descaled Physical Phi_inv."""
        # Phi_inv_phys = Phi_inv_scaled @ S_inv
        return torch.linalg.pinv(self.Phi, rcond=1e-6) @ torch.diag(1.0 / (self.z_scale + 1e-6))

    def get_Lambda(self):
        Lambda = (
            torch.diag(self.eig_diag) + 
            torch.diag(self.eig_super, diagonal=1) + 
            torch.diag(self.eig_sub, diagonal=-1)
        )
        return Lambda

    def get_K(self):
        """Returns descaled Physical K."""
        # K_phys = S @ K_scaled @ S_inv
        K_scaled = self.Phi @ self.get_Lambda() @ torch.linalg.pinv(self.Phi, rcond=1e-6)
        S = torch.diag(self.z_scale)
        S_inv = torch.diag(1.0 / (self.z_scale + 1e-6))
        return S @ K_scaled @ S_inv

    def get_eigenvalues(self):
        return torch.linalg.eigvals(self.get_Lambda())

    # ------------------------------------------------
    # Forward pass
    # ------------------------------------------------

    def forward(self, x):
        x_scaled = self.scale_state(x)
        z = self.expand(x_scaled)
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

        # Back to state dims
        x_next_scaled = self.de_expand(z_next)
        x_next = self.unscale_state(x_next_scaled)
        return x_next

    # ------------------------------------------------
    # Training loss
    # ------------------------------------------------

    def compute_loss(self, x, x_next_true, future_x=None):
        # Scale inputs
        x_scaled = self.scale_state(x)
        x_next_true_scaled = self.scale_state(x_next_true)

        z = self.expand(x_scaled)
        z_next_true = self.expand(x_next_true_scaled)
        
        z = torch.clamp(z, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)
        z_next_true = torch.clamp(z_next_true, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)

        # --------------------------------------------------
        # 1. One-Step Dynamics
        # --------------------------------------------------
        z_next_pred = self._advance_z(z)
        
        loss_lift = torch.mean((z_next_pred - z_next_true)**2)
        x_next_pred_scaled = self.de_expand(z_next_pred)
        loss_state = nn.MSELoss()(x_next_pred_scaled, x_next_true_scaled)

        # --------------------------------------------------
        # 2. Multi-step Rollout Loss
        # --------------------------------------------------
        loss_rollout = torch.tensor(0.0, device=x.device)
        
        if future_x is not None and future_x.ndim == 3:
            horizon = min(self.rollout_horizon, future_x.shape[1])
            
            if horizon >= 2:
                z_curr_pred = z_next_pred 
                
                for k in range(1, horizon):
                    z_curr_pred = self._advance_z(z_curr_pred)
                    z_true_k = self.expand(self.scale_state(future_x[:, k, :]))
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
        # 4. Lambda Regularization (Universal Form)
        # --------------------------------------------------
        # Smooth activation for when rotation-block penalties should apply.
        # This avoids hard gating when one off-diagonal is tiny.
        tol = float(self.rotation_act_tol)
        scale = float(self.rotation_act_scale)
        act = torch.sigmoid((torch.abs(self.eig_super) + torch.abs(self.eig_sub) - tol) / (scale + 1e-12))

        loss_antisym = torch.mean(act * (self.eig_super + self.eig_sub) ** 2)

        diag_diff = self.eig_diag[:-1] - self.eig_diag[1:]
        loss_diag_match = torch.mean(act * diag_diff ** 2)

        # Sparsity continuously pushes unused off-diagonals to 0
        loss_sparsity = torch.mean(torch.abs(self.eig_sub)) + torch.mean(torch.abs(self.eig_super))

        # Phi orthogonality (off-diagonal of Phi^T Phi)
        phi_phys = self.get_Phi()
        G = phi_phys.T @ phi_phys
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        loss_phi_orth = torch.norm(G - I, p='fro') ** 2 / float(G.shape[0] ** 2)

        # --------------------------------------------------
        # 5. Total Loss Compilation
        # --------------------------------------------------
        loss_total = (
            loss_lift
            + 5.0 * loss_state          
            + self.rollout_weight * loss_rollout 
            + 0.01 * loss_unit_length   # Bounding the physical Phi
            + 5.0 * loss_antisym        # Smoothly gated rotation symmetry penalty
            + 5.0 * loss_diag_match     # Smoothly gated diagonal-matching penalty
            + 1e-6 * loss_sparsity
            + float(self.phi_orth_weight) * loss_phi_orth
        )
        
        loss_dict = {
            "state": loss_state.item(),
            "lift": loss_lift.item(),
            "rollout": loss_rollout.item(),
            "unit": loss_unit_length.item(),
            "lam_asy": loss_antisym.item(),
            "lam_bal": loss_diag_match.item(),
            "lam_sp": loss_sparsity.item(),
            "phi_orth": loss_phi_orth.item(),
            "act_mean": act.mean().item(),
        }
        
        return loss_total, loss_dict

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