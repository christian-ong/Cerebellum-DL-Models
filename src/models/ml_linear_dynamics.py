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

        # Fixed lifted-feature scaling (dataset-level stats, not batch stats)
        self.register_buffer("lift_mean", torch.zeros(self.latent_dim))
        self.register_buffer("lift_scale", torch.ones(self.latent_dim))
        self.lift_norm_eps = 1e-6

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

    def set_lifted_normalization_stats(self, mean, scale):
        mean = torch.as_tensor(mean, dtype=self.lift_mean.dtype, device=self.lift_mean.device)
        scale = torch.as_tensor(scale, dtype=self.lift_scale.dtype, device=self.lift_scale.device)
        self.lift_mean.copy_(mean)
        self.lift_scale.copy_(torch.clamp(scale, min=self.lift_norm_eps))

    def _normalize(self, z, update_stats=True):
        del update_stats
        return (z - self.lift_mean) / self.lift_scale

    def _unnormalize(self, z_norm):
        return z_norm * self.lift_scale + self.lift_mean

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
        z = self._normalize(z_raw) # Normalize BEFORE K
        z_next = self.K(z)
        z_next_physical = self._unnormalize(z_next)
        x_next = self.expander.de_expand(z_next_physical)
        return x_next

    # ------------------------------------------------
    # Training loss
    # ------------------------------------------------

    def compute_loss(self, x, x_next_true, future_x=None):
        # compute_loss fix
        z_raw = self.expander.expand(x)
        z_next_true_raw = self.expander.expand(x_next_true)

        z_norm = self._normalize(z_raw)
        z_next_true_norm = self._normalize(z_next_true_raw, update_stats=False)

        z_next_pred = self._advance_z(z_norm)
        z_next_physical = self._unnormalize(z_next_pred)

        loss_lift = torch.mean((z_next_pred - z_next_true_norm)**2)
        loss_state = nn.MSELoss()(self.expander.de_expand(z_next_physical), x_next_true)

        loss_rollout = torch.tensor(0.0, device=x.device)
        if future_x is not None and future_x.ndim == 3:
            horizon = min(self.rollout_horizon, future_x.shape[1])
            if horizon >= 2:
                # PRE-EXPAND FUTURE TARGETS (Saves massive time)
                # This moves from (N, T, d) -> (T, N, dz)
                z_targets = self.expander.expand(future_x.reshape(-1, self.state_dim))
                z_targets = z_targets.reshape(x.shape[0], future_x.shape[1], -1)
                z_targets_norm = self._normalize(z_targets.reshape(-1, self.latent_dim), update_stats=False)
                z_targets_norm = z_targets_norm.reshape_as(z_targets)

                z_curr = z_next_pred
                for k in range(1, horizon):
                    z_curr = self._advance_z(z_curr)
                    z_curr_phys = self._unnormalize(z_curr)
                    loss_rollout += torch.mean((z_curr - z_targets_norm[:, k, :])**2)
                    loss_rollout += torch.mean((self.expander.de_expand(z_curr_phys) - future_x[:, k, :])**2)
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
        z = self._normalize(z, update_stats=False)

        # 2. Rollout completely in the latent space
        for _ in range(steps):
            z = self._advance_z(z)
            z_phys = self._unnormalize(z) 
            traj.append(self.expander.de_expand(z_phys).squeeze(0))
        return torch.stack(traj)