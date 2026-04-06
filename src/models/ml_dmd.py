import torch
import torch.nn as nn

from src.models.expander import ManualExpansion


class ML_DMD(ManualExpansion):
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

        x  --expand-->  z
        z  --scale-->   z_norm
        z_norm --Phi^{-1}--> modal coordinates b
        b --Lambda--> modal_next
        modal_next --Phi--> z_norm_next
        z_norm_next --descale--> z_next
        z_next --de_expand--> x_next

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
        # z_scale normalizes lifted observables
        # (computed from training data).
        self.register_buffer("z_scale", torch.ones(self.latent_dim))

        # ------------------------------------------------
        # Degree-based lifted loss weighting
        # ------------------------------------------------
        # High-degree monomials can dominate the loss.
        # We therefore downweight them.
        degrees = torch.tensor(
            [self._feature_degree(name) for name in self.expand_names],
            dtype=torch.float32,
        )
        weights = 1.0 / (degrees + 1.0)
        self.register_buffer("lift_weights", weights)
        
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
        """
        Store lifted feature scaling computed from training data.

        z_scale_i ≈ mean(|z_i|)

        This ensures the lifted coordinate system stays fixed
        across training, validation, and rollout.
        """

        if not torch.is_tensor(z_scale):
            z_scale = torch.tensor(z_scale, dtype=self.Phi.dtype)

        self.z_scale.copy_(z_scale.to(self.z_scale.device))

    def get_scaling_matrix(self):
        """
        Return diagonal scaling matrix S such that

            z_raw = S z_scaled

        where z_scaled = z_raw / z_scale.
        """
        return torch.diag(self.z_scale)


    def get_Phi_scaled(self):
        """
        Return Phi in scaled lifted coordinates.
        """
        return self.Phi


    def get_Phi_scaled_inv(self):
        """
        Return pseudo-inverse of Phi in scaled lifted coordinates.
        """
        return torch.linalg.pinv(self.Phi, rcond=1e-6)


    def get_Lambda(self):
        """
        Return the learned Lambda matrix in scaled modal coordinates.
        """
        return self.Lambda


    def get_K_scaled(self):
        """
        Return the lifted operator in scaled lifted coordinates:

            K_scaled = Phi Lambda Phi^{-1}
        """
        Phi_inv = self.get_Phi_scaled_inv()
        return self.Phi @ self.Lambda @ Phi_inv


    def get_Phi_true(self):
        """
        Return Phi expressed in the original lifted coordinates.

        Since z_raw = S z_scaled, the eigenvector matrix transforms as
            Phi_true = S Phi_scaled
        """
        S = self.get_scaling_matrix()
        return S @ self.Phi


    def get_K_true(self):
        """
        Return the equivalent operator in the original lifted coordinates.

        If z_scaled = S^{-1} z_raw, then
            K_true = S K_scaled S^{-1}
        """
        S = self.get_scaling_matrix()
        S_inv = torch.diag(1.0 / (self.z_scale + 1e-12))
        K_scaled = self.get_K_scaled()
        return S @ K_scaled @ S_inv


    def get_eigenvalues(self):
        """
        Eigenvalues of the lifted Koopman operator.
        These are identical for K_scaled and K_true.
        """
        K_scaled = self.get_K_scaled()
        return torch.linalg.eigvals(K_scaled)

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
        z_raw = self.expand(x)

        # Normalize lifted coordinates
        z = z_raw / self.z_scale

        # Compute pseudo-inverse of Phi
        Phi_inv = torch.linalg.pinv(self.Phi, rcond=1e-6)
        cond_number = torch.linalg.cond(self.Phi)
        if cond_number > 1e6:
            print(f"Warning: High condition number for Phi: {cond_number:.2e}")

        # Convert to modal coordinates
        b = z @ Phi_inv.mT

        # Modal evolution
        b_next = b @ self.Lambda.mT

        # Convert back to lifted coordinates
        z_next = b_next @ self.Phi.mT

        # De-normalize lifted observables
        z_next_raw = z_next * self.z_scale

        # Recover original state
        x_next = self.de_expand(z_next_raw)

        return x_next

    # ------------------------------------------------
    # Training loss
    # ------------------------------------------------

    def compute_loss(self, x, x_next_true):

        # Expand states
        z_raw = self.expand(x)
        z_next_true_raw = self.expand(x_next_true)

        # Normalize lifted coordinates
        z = z_raw / self.z_scale
        z_next_true = z_next_true_raw / self.z_scale

        Phi_inv = torch.linalg.pinv(self.Phi, rcond=1e-6)

        # Convert to modal coordinates
        b = z @ Phi_inv.mT

        # Modal evolution
        b_next = b @ self.Lambda.mT

        # Convert back to lifted coordinates
        z_next_pred = b_next @ self.Phi.mT

        # --------------------------------------------------
        # 1) Weighted lifted loss
        # --------------------------------------------------

        diff = z_next_pred - z_next_true
        loss_lift = torch.mean(self.lift_weights * diff**2)

        # --------------------------------------------------
        # 2) State prediction loss
        # --------------------------------------------------

        z_next_pred_raw = z_next_pred * self.z_scale
        x_next_pred = self.de_expand(z_next_pred_raw)
        loss_state = nn.MSELoss()(x_next_pred, x_next_true)

        # --------------------------------------------------
        # 3) Φ conditioning regularization
        # --------------------------------------------------

        col_norms = torch.linalg.norm(self.Phi, dim=0)
        loss_unit_length = torch.mean((col_norms - 1.0) ** 2)

        # --------------------------------------------------
        # 4) Stability regularization on effective operator
        # --------------------------------------------------

        # K_eff = self.Phi @ self.Lambda @ Phi_inv
        # eigvals = torch.linalg.eigvals(K_eff)

        # loss_stability = torch.mean(
        #     torch.relu(torch.abs(eigvals) - 1.0) ** 2
        # )

        # --------------------------------------------------
        # Total loss
        # --------------------------------------------------

        loss = (
            loss_lift
            + 0.1 * loss_state
            + 1e-3 * loss_unit_length
            # + 1e-3 * loss_stability
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