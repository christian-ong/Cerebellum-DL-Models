import torch
import torch.nn as nn

from src.models.expander import ManualExpansion


class ManualExpansion_MLDMD(ManualExpansion):

    def __init__(
        self,
        state_dim=2,
        expansion_degree=2,
        constant_expansion=True,
        sine_cosine_expansion=False,
        expansion_type="general",
        system=None,
    ):

        super().__init__(
            state_dim=state_dim,
            expansion_degree=expansion_degree,
            constant_expansion=constant_expansion,
            sine_cosine_expansion=sine_cosine_expansion,
            expansion_type=expansion_type,
            system=system,
        )

        self.latent_dim = self.expanded_dim

        # Randomly initialize K matrix
        K_init = torch.randn(self.latent_dim, self.latent_dim) * 0.01
        self.K = nn.Parameter(K_init)


    def forward(self, x):

        # Expand basis
        x_expanded = self.expand(x)

        # Step with linear dynamics in expanded space
        x_expanded_next = x_expanded @ self.K.mT

        # De-expand to original space
        x_next = self.de_expand(x_expanded_next)

        return x_next


    def compute_loss(self, x, x_next_true):

        # Compute the forward pass
        x_next_hat = self.forward(x)

        # Prediction loss
        loss_predict = nn.MSELoss()(x_next_hat, x_next_true)
        step_length = torch.norm(x_next_true - x)
        loss_predict_normalized = loss_predict / (step_length + 1e-6)

        return [loss_predict_normalized]


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