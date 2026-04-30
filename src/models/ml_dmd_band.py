import torch
import torch.nn as nn

from src.models.expander import ManualExpansion


class ML_DMD_BAND(ManualExpansion):
    """
    Manual expansion + learned Koopman eigendecomposition.

    Instead of learning the lifted linear operator K directly, we learn

        K = Phi Λ Phi^{-1}

    but we NEVER explicitly construct K.

    The lifted dynamics are applied as

        z_{t+1} = Phi Λ Phi^{-1} z_t

    which corresponds to:

        z_norm -> modal coordinates -> modal evolution -> lifted coordinates
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

        # ------------------------------------------------
        # Initialize manual basis expansion
        # ------------------------------------------------
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
        # Eigenvector matrix Φ
        # ------------------------------------------------
        # Columns correspond to Koopman modes.
        # Initialized close to identity for stability.
        self.Phi_scaled = nn.Parameter(
            torch.eye(self.latent_dim)
            + 0.001 * torch.randn(self.latent_dim, self.latent_dim)
        )

        # ------------------------------------------------
        # Tridiagonal Eigenvalue Matrix Λ
        # ------------------------------------------------
        # We replace the full dense matrix with a tridiagonal structure.
        # This forces diagonalization but allows:
        # 1. 2x2 blocks for complex rotation (using +1 and -1 diagonals)
        # 2. Jordan cascades for polynomial growth (using only +1 diagonal)
        
        # Main diagonal (pure real eigenvalues / stability)
        self.eig_diag = nn.Parameter(torch.ones(self.latent_dim) + torch.randn(self.latent_dim) * 0.01)
        
        # Superdiagonal (+1 offset) handles Jordan couplings and rotation
        self.eig_super = nn.Parameter(torch.randn(self.latent_dim - 1) * 0.01)
        
        # Subdiagonal (-1 offset) handles the feedback needed for complex rotation
        self.eig_sub = nn.Parameter(torch.randn(self.latent_dim - 1) * 0.01)

        # ------------------------------------------------
        # Feature scaling buffer
        # ------------------------------------------------
        self.register_buffer("z_scale", torch.ones(self.latent_dim))
        self.register_buffer("x_mean", torch.zeros(state_dim))
        self.register_buffer("x_scale", torch.ones(state_dim))
        self.max_abs_z_norm = 1e6

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

    def scale_state(self, x):
        return (x - self.x_mean) / self.x_scale    
    
    def unscale_state(self, x):
        return x * self.x_scale + self.x_mean
    
    def _feature_degree(self, name: str) -> int:
        name = name.strip()

        if name == "1":
            return 0

        if name.startswith("sin(") or name.startswith("cos("):
            return 1

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
    # Set lifted scaling
    # ------------------------------------------------

    def set_z_scale(self, z_scale):
        if not torch.is_tensor(z_scale):
            z_scale = torch.tensor(z_scale, dtype=self.Phi_scaled.dtype)
        self.z_scale.copy_(z_scale.to(self.z_scale.device))

    def get_scaling_matrix(self):
        return torch.diag(self.z_scale)

    def get_Phi_scaled(self):
        return self.Phi_scaled

    def get_Phi_scaled_inv(self):
        return torch.linalg.pinv(self.Phi_scaled, rcond=1e-6)

    def get_Lambda(self):
        """
        Dynamically construct the Tridiagonal Lambda matrix.
        """
        Lambda = (
            torch.diag(self.eig_diag) + 
            torch.diag(self.eig_super, diagonal=1) + 
            torch.diag(self.eig_sub, diagonal=-1)
        )
        return Lambda

    def get_K_scaled(self):
        """
        Return the lifted operator in scaled lifted coordinates.
        """
        Phi_inv_scaled = self.get_Phi_scaled_inv()
        Lambda = self.get_Lambda()
        return self.Phi_scaled @ Lambda @ Phi_inv_scaled

    def get_Phi_true(self):
        S = self.get_scaling_matrix()
        return S @ self.Phi_scaled

    def get_K_true(self):
        S = self.get_scaling_matrix()
        S_inv = torch.diag(1.0 / (self.z_scale + 1e-12))
        K_scaled = self.get_K_scaled()
        return S @ K_scaled @ S_inv

    def get_eigenvalues(self):
        K_scaled = self.get_K_scaled()
        return torch.linalg.eigvals(K_scaled)

    # ------------------------------------------------
    # Forward pass
    # ------------------------------------------------

    def forward(self, x):
        # Lift state into observable space
        x_scaled = self.scale_state(x)
        z_raw = self.expand(x_scaled)

        # Normalize lifted coordinates
        z = torch.clamp(z_raw / self.z_scale, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)

        I_eps = 1e-6 * torch.eye(self.latent_dim, device=self.Phi_scaled.device, dtype=self.Phi_scaled.dtype)
        Phi_reg = self.Phi_scaled + I_eps
        cond_number = torch.linalg.cond(self.Phi_scaled)
        if cond_number > 1e6:
            print(f"Warning: High condition number for Phi_scaled: {cond_number:.2e}")

        # Convert to modal coordinates
        b = torch.linalg.solve(Phi_reg, z.mT).mT

        # Modal evolution
        Lambda = self.get_Lambda()
        b_next = b @ Lambda.mT

        # Convert back to lifted coordinates
        z_next = b_next @ self.Phi_scaled.mT

        # De-normalize lifted observables
        z_next_raw = z_next * self.z_scale

        # Recover original state
        x_next_scaled = self.de_expand(z_next_raw)
        x_next = self.unscale_state(x_next_scaled)
        return x_next

    # ------------------------------------------------
    # Training loss
    # ------------------------------------------------

    def compute_loss(self, x, x_next_true):
        # Expand states
        x_scaled = self.scale_state(x)
        x_next_true_scaled = self.scale_state(x_next_true)

        z_raw = self.expand(x_scaled)
        z_next_true_raw = self.expand(x_next_true_scaled)

        # Normalize lifted coordinates
        z = torch.clamp(z_raw / self.z_scale, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)
        z_next_true = torch.clamp(
            z_next_true_raw / self.z_scale,
            min=-self.max_abs_z_norm,
            max=self.max_abs_z_norm,
        )

        I_eps = 1e-6 * torch.eye(self.latent_dim, device=self.Phi_scaled.device, dtype=self.Phi_scaled.dtype)
        Phi_reg = self.Phi_scaled + I_eps

        # Convert to modal coordinates
        b = torch.linalg.solve(Phi_reg, z.mT).mT

        # Modal evolution
        Lambda = self.get_Lambda()
        b_next = b @ Lambda.mT

        # Convert back to lifted coordinates
        z_next_pred = b_next @ self.Phi_scaled.mT

        # --------------------------------------------------
        # 1) Weighted lifted loss
        # --------------------------------------------------
        diff = z_next_pred - z_next_true
        loss_lift = torch.mean(self.lift_weights * diff**2)

        # --------------------------------------------------
        # 2) State prediction loss
        # --------------------------------------------------
        z_next_pred_raw = z_next_pred * self.z_scale
        x_next_pred_scaled = self.de_expand(z_next_pred_raw)
        
        x_next_true_scaled = self.scale_state(x_next_true)
        loss_state = nn.MSELoss()(x_next_pred_scaled, x_next_true_scaled)

        # --------------------------------------------------
        # 3) Φ Gauge Fixing (Mathematical Scale Uniqueness)
        # --------------------------------------------------
        # Eigenvectors are scale-invariant. We must pin their norms to 1.0
        # so the network doesn't arbitrarily scale Phi up and b down.
        col_norms = torch.linalg.norm(self.get_Phi_true(), dim=0)
        loss_unit_length = torch.mean((col_norms - 1.0) ** 2)

        # --------------------------------------------------
        # 4) Canonical Real Schur Form (Strict Diagonalization)
        # --------------------------------------------------
        # If eig_sub is non-zero, strictly force the 2x2 block to be a 
        # mathematically pure complex rotation: [a, b; -b, a]
        
        # Force off-diagonals to be exact opposites: s = -u
        loss_antisym = torch.mean(torch.abs(self.eig_sub) * torch.abs(self.eig_super + self.eig_sub))
        
        # Force neighboring diagonals to be equal: d_i = d_{i+1}
        diag_diff = self.eig_diag[:-1] - self.eig_diag[1:]
        loss_diag_match = torch.mean(torch.abs(self.eig_sub) * torch.abs(diag_diff))

        # Force Single Canonical Orientation
        loss_sign_sub = torch.mean(torch.relu(self.eig_sub))
        loss_sign_super = torch.mean(torch.relu(-self.eig_super))

        # --------------------------------------------------
        # Total loss
        # --------------------------------------------------
        loss = (
            loss_lift
            + 0.1 * loss_state
            + 1e-3 * loss_unit_length
            + 1.0 * loss_antisym          # Bumped to 1.0 (Safe because it zeroes out for Jordan blocks)
            + 1.0 * loss_diag_match       # Bumped to 1.0 
            + 0.1 * loss_sign_sub         # Locks orientation: eig_sub <= 0
            + 0.1 * loss_sign_super       # Locks orientation: eig_super >= 0
        )
        return (loss,)

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