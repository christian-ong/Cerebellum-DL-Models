import torch
import torch.nn as nn

from src.models.expander import build_expander


class ML_LinearDynamics(nn.Module):
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

    def get_K(self):
        return self.K.weight.mT

    def get_eigenvalues(self):
        K_true = self.get_K()
        return torch.linalg.eigvals(K_true)
    
    def _advance_z(self, z):
        z_next = self.K(z)
        return torch.clamp(z_next, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)
    # ------------------------------------------------
    # Forward pass
    # ------------------------------------------------

    def forward(self, x):
        # Lift state into expanded space
        z_raw = self.expander.expand(x)
        z = torch.clamp(z_raw, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)

        # Linear Koopman step
        z_next = self.K(z)

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

        loss_lift = torch.mean((z_next_pred - z_next_true)**2)
        loss_state = nn.MSELoss()(self.expander.de_expand(z_next_pred), x_next_true)

        loss_rollout = torch.tensor(0.0, device=x.device)
        if future_x is not None and future_x.ndim == 3:
            horizon = min(self.rollout_horizon, future_x.shape[1])
            if horizon >= 2:
                # PRE-EXPAND FUTURE TARGETS (Saves massive time)
                # This moves from (N, T, d) -> (T, N, dz)
                z_targets = self.expander.expand(future_x.reshape(-1, self.state_dim))
                z_targets = z_targets.reshape(x.shape[0], future_x.shape[1], -1)

                z_curr = z_next_pred
                for k in range(1, horizon):
                    z_curr = self._advance_z(z_curr)
                    loss_rollout += torch.mean((z_curr - z_targets[:, k, :])**2)
                    loss_rollout += torch.mean((self.expander.de_expand(z_curr) - future_x[:, k, :])**2)
                loss_rollout /= (horizon - 1)

        # Total loss
        loss = (
            loss_lift 
            + 0.5 * loss_state 
            + 1.0 * loss_rollout    # INCREASED: 0.1 -> 1.0
        )

        loss_dict = {
            "lift": loss_lift.item(),
            "state": loss_state.item(),
            "rollout": loss_rollout.item(),
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
        z = torch.clamp(z, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)

        # 2. Rollout completely in the latent space
        for _ in range(steps):
            z = self._advance_z(z)               # Step linearly forward!
            x_next = self.expander.de_expand(z)           # Peek down to grab the physical state
            traj.append(x_next.squeeze(0))

        return torch.stack(traj)