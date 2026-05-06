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
        self.max_abs_z_norm = 1e6
        self.rollout_horizon = 20
        self.rollout_weight = 0.1

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

    def get_K(self):
        """
        Return the learned operator acting in scaled lifted coordinates.

        Dynamics in scaled coordinates:
            z_scaled_next = K_scaled z_scaled
        """
        return self.K.weight.mT

    def get_eigenvalues(self):
        """
        Eigenvalues of the lifted Koopman operator.
        These are identical for K_scaled and K_true.
        """
        K_true = self.get_K()
        return torch.linalg.eigvals(K_true)
    
    def _advance_z(self, z):
        z_next = self.K(z)
        return torch.clamp(z_next, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)
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
        z_raw = self.expand(x)

        # Normalize lifted features
        z = torch.clamp(z_raw, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)

        # Linear Koopman step
        z_next = self.K(z)

        # Recover original state
        x_next = self.de_expand(z_next)
        return x_next

    # ------------------------------------------------
    # Training loss
    # ------------------------------------------------

    def compute_loss(self, x, x_next_true, future_x=None):
        z = self.expand(x)
        z_next_true = self.expand(x_next_true)

        z_next_pred = self._advance_z(z)

        # 1) Lifted Koopman loss
        diff = z_next_pred - z_next_true
        loss_lift = torch.mean(self.lift_weights * diff**2)

        # 2) State prediction loss
        x_next_pred = self.de_expand(z_next_pred)
        loss_state = nn.MSELoss()(x_next_pred, x_next_true)

        # 3) Multi-step Rollout Loss
        loss_rollout = torch.tensor(0.0, device=x.device)
        if future_x is not None and future_x.ndim == 3:
            horizon = min(self.rollout_horizon, future_x.shape[1])
            if horizon >= 2:
                z_curr_pred = z_next_pred 
                for k in range(1, horizon):
                    z_curr_pred = self._advance_z(z_curr_pred)
                    z_true_k = self.expand(future_x[:, k, :])
                    z_true_k = torch.clamp(z_true_k, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)
                    loss_rollout += torch.mean(self.lift_weights * (z_curr_pred - z_true_k)**2)
                loss_rollout = loss_rollout / float(horizon - 1)

        # Total loss
        loss = (
            loss_lift 
            + 0.1 * loss_state 
            + self.rollout_weight * loss_rollout
        )

        loss_dict = {
            "loss": loss.item(),
            "loss_lift": loss_lift.item(),
            "loss_state": loss_state.item(),
            "rollout": loss_rollout.item(),
        }

        return (loss, loss_dict)
    
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