import torch
import torch.nn as nn

from src.models.expander import ManualExpansion


class ManualExpansion_MLDMD(ManualExpansion):

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

        self.K = nn.Linear(
            self.latent_dim,
            self.latent_dim,
            bias=False,
        )

        # important: good initialization
        nn.init.eye_(self.K.weight)

    # ------------------------------------------------
    # Forward
    # ------------------------------------------------

    def forward(self, x):

        z = self.expand(x)
        z_next = self.K(z)
        x_next = self.de_expand(z_next)

        return x_next

    # ------------------------------------------------
    # Loss
    # ------------------------------------------------

    def compute_loss(self, x, x_next_true):

        z = self.expand(x)
        z_next_true = self.expand(x_next_true)

        z_next_pred = self.K(z)

        # main EDMD loss
        loss_lift = nn.MSELoss()(z_next_pred, z_next_true)

        # optional state consistency loss
        x_next_pred = self.de_expand(z_next_pred)
        loss_state = nn.MSELoss()(x_next_pred, x_next_true)

        loss = loss_lift + 0.1 * loss_state

        return (loss,)

    # ------------------------------------------------
    # Rollout
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

            z = self.expand(x)
            z = self.K(z)
            x = self.de_expand(z)

            traj.append(x.squeeze(0))

        return torch.stack(traj)