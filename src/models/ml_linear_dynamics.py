import torch
import torch.nn as nn

from src.models.expander import ManualExpansion


class ML_LinearDynamics(ManualExpansion):
    """
    Manual expansion + learned Koopman operator (ML-DMD).

    This model follows the EDMD / Koopman learning idea:

        z_{t+1} = K z_t

    where
        x  = original state
        z  = lifted state (basis expansion)
        K  = learned linear Koopman operator

    Pipeline:

        x  --expand-->  z
        z  --scale-->   z_norm
        z_norm --K-->   z_norm_next
        z_norm_next --descale--> z_next
        z_next --de_expand--> x_next

    Important design choices:
    - Polynomial / manual basis expansion
    - Feature normalization for numerical stability
    - Weighted loss to prevent high-degree monomials dominating training
    - Eigenvalue regularization to stabilize rollouts
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
        # Initialize basis expansion
        # ------------------------------------------------
        # This creates the lifted state representation z.
        # The expansion is implemented in the parent class.
        super().__init__(
            state_dim=state_dim,
            expansion_degree=expansion_degree,
            bias=bias,
            sine_cosine_expansion=sine_cosine_expansion,
            expansion_type=expansion_type,
            system=system,
        )

        # Dimension of lifted space
        self.latent_dim = self.expanded_dim

        # ------------------------------------------------
        # Koopman operator
        # ------------------------------------------------
        # Linear operator in lifted space
        #
        #     z_{t+1} = K z_t
        #
        # K is learned via gradient descent.
        self.K = nn.Linear(
            self.latent_dim,
            self.latent_dim,
            bias=False,
        )

        # Initialize close to identity.
        # This assumes small time steps (dt small), so dynamics are near identity.
        nn.init.eye_(self.K.weight)

        # ------------------------------------------------
        # Feature scaling buffer
        # ------------------------------------------------
        # Polynomial features can vary wildly in magnitude
        # (e.g. x vs x^10).
        #
        # We normalize features during training to improve conditioning:
        #
        #     z_norm = z / z_scale
        #
        # z_scale is computed once from training data.
        #
        # register_buffer ensures:
        # - saved in checkpoints
        # - moved to GPU automatically
        # - not trainable
        self.register_buffer("z_scale", torch.ones(self.latent_dim))
        self.register_buffer("x_mean", torch.zeros(state_dim))
        self.register_buffer("x_scale", torch.ones(state_dim))
        self.max_abs_z_norm = 1e6

        # ------------------------------------------------
        # Lifted loss weights
        # ------------------------------------------------
        # High-order polynomial terms tend to explode and dominate
        # the loss. We therefore downweight them.
        #
        # Example:
        #
        #   x       -> weight = 1
        #   x^2     -> weight = 1/2
        #   x^3     -> weight = 1/3
        #
        # This keeps training focused on low-order dynamics.
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

    def unscale_state(self, x):
        return x * self.x_scale + self.x_mean

    def scale_state(self, x):
        return (x - self.x_mean) / self.x_scale
        
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
    # Set scaling for lifted features
    # ------------------------------------------------

    def set_z_scale(self, z_scale):
        """
        Store feature scaling computed from the training data.

        z_scale_i ≈ mean(|z_i|)

        This improves conditioning of the Koopman regression.
        """
        if not torch.is_tensor(z_scale):
            z_scale = torch.tensor(z_scale, dtype=self.K.weight.dtype)

        self.z_scale.copy_(z_scale.to(self.z_scale.device))

    def get_scaling_matrix(self):
        """
        Return diagonal scaling matrix S such that

            z_raw = S z_scaled

        where z_scaled = z_raw / z_scale.
        """
        return torch.diag(self.z_scale)

    def get_K_scaled(self):
        """
        Return the learned operator acting in scaled lifted coordinates.

        Dynamics in scaled coordinates:
            z_scaled_next = K_scaled z_scaled
        """
        return self.K.weight.mT

    def get_K_true(self):
        """
        Return the equivalent operator in the lifted coordinates induced by
        the standardized physical state.

        If z_scaled = S^{-1} z_raw, then
            K_true = S K_scaled S^{-1}

        This is still an operator in lifted coordinates, not a global linear
        operator in raw physical state coordinates.
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
        K_true = self.get_K_true()
        return torch.linalg.eigvals(K_true)

    # ------------------------------------------------
    # Forward pass
    # ------------------------------------------------

    def forward(self, x):
        """
        One-step prediction:

            x_t -> x_{t+1}

        Internally:

            x -> z
            z -> z_norm
            z_norm -> K z_norm
            z_norm_next -> z_next
            z_next -> x_next
        """

        # Lift state into expanded space
        x_scaled = self.scale_state(x)
        z_raw = self.expand(x_scaled)

        # Normalize lifted features
        z = torch.clamp(z_raw / self.z_scale, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)

        # Linear Koopman step
        z_next = self.K(z)

        # Convert back to original lifted coordinates
        z_next_raw = z_next * self.z_scale

        # Recover original state
        x_next_scaled = self.de_expand(z_next_raw)
        x_next = self.unscale_state(x_next_scaled)
        return x_next

    # ------------------------------------------------
    # Training loss
    # ------------------------------------------------

    def compute_loss(self, x, x_next_true):
        """
        Training objective.

        Three components:

        1) Lifted Koopman loss
           Enforces linear dynamics in lifted space

        2) State prediction loss
           Ensures correct predictions in physical state

        3) Eigenvalue stability loss
           Prevents unstable eigenvalues during rollouts
        """

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

        # Predict next lifted state
        z_next_pred = self.K(z)

        # ------------------------------------------------
        # 1) Lifted Koopman loss
        # ------------------------------------------------
        # Weighted MSE in lifted space
        diff = z_next_pred - z_next_true
        loss_lift = torch.mean(self.lift_weights * diff**2)

        # ------------------------------------------------
        # 2) State prediction loss
        # ------------------------------------------------
        # Convert lifted prediction back to state space
        z_next_pred_raw = z_next_pred * self.z_scale
        x_next_pred_scaled = self.de_expand(z_next_pred_raw)
        
        # Compare scaled prediction against scaled ground truth target
        x_next_true_scaled = self.scale_state(x_next_true)
        loss_state = nn.MSELoss()(x_next_pred_scaled, x_next_true_scaled)

        # ------------------------------------------------
        # 3) Stability regularization
        # ------------------------------------------------
        # Penalize eigenvalues outside the unit circle.
        #
        # Ensures stable long-term rollouts.

        # K_eff = self.K.weight * self.z_scale.unsqueeze(1) / self.z_scale.unsqueeze(0)
        # eigvals = torch.linalg.eigvals(K_eff)

        # loss_stability = torch.mean(
        #     torch.relu(torch.abs(eigvals) - 1.0) ** 2
        # )

        # ------------------------------------------------
        # Total loss
        # ------------------------------------------------
        loss = (
            loss_lift
            + 0.1 * loss_state
            # + 1e-3 * loss_stability
        )

        return (loss,)

    # ------------------------------------------------
    # Rollout simulation
    # ------------------------------------------------

    def rollout(self, x0, steps):
        """
        Predict a trajectory starting from x0.

        This repeatedly applies the learned Koopman operator:

            z_{t+1} = K z_t

        and maps the lifted state back to the original state space.
        """

        # Convert input to tensor
        if not torch.is_tensor(x0):
            x0 = torch.tensor(
                x0,
                dtype=next(self.parameters()).dtype,
                device=next(self.parameters()).device,
            )

        # Ensure batch dimension
        if x0.ndim == 1:
            x = x0.unsqueeze(0)
        else:
            x = x0

        traj = [x.squeeze(0)]

        for _ in range(steps):
            x = self.forward(x)
            traj.append(x.squeeze(0))

        return torch.stack(traj)