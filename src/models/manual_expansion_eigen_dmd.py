import torch
import torch.nn as nn

from src.models.expander import ManualExpansion


class ManualExpansion_EigenDMD(ManualExpansion):

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

        self.Phi = nn.Parameter(torch.eye(self.latent_dim))
        self.Phi_inv = nn.Parameter(torch.eye(self.latent_dim))
        self.Lambda = nn.Parameter(torch.eye(self.latent_dim))

    def forward(self, x):

        # Expand basis
        x_expanded = self.expand(x)

        # Step in transformed coordinates
        b_t = x_expanded @ self.Phi_inv.mT
        b_next = b_t @ self.Lambda.mT
        x_expanded_next = b_next @ self.Phi.mT

        # Recover original state
        x_next = self.de_expand(x_expanded_next)

        return x_next

    def compute_loss(self, x, x_next_true):

        # Forward in expanded space
        x_expanded = self.expand(x)
        b_t = x_expanded @ self.Phi_inv.mT
        b_next = b_t @ self.Lambda.mT
        x_expanded_next_hat = b_next @ self.Phi.mT

        # True next state in expanded space
        x_expanded_next_true = self.expand(x_next_true)

        # --------------------------------------------------
        # 1) Prediction loss in expanded space
        # --------------------------------------------------
        expanded_loss = nn.MSELoss()(x_expanded_next_hat, x_expanded_next_true)

        step_length = torch.norm(x_next_true - x)
        loss_predict = expanded_loss / (step_length + 1e-6)

        # --------------------------------------------------
        # 2) Phi and Phi_inv should be inverses
        # --------------------------------------------------
        identity = torch.eye(self.latent_dim, device=x.device, dtype=x.dtype)
        loss_phi_inv = torch.norm(self.Phi @ self.Phi_inv - identity)

        # --------------------------------------------------
        # 3) Mild column normalization on Phi
        # --------------------------------------------------
        col_norms = torch.linalg.norm(self.Phi, dim=0)
        loss_unit_length = torch.mean((col_norms - 1.0) ** 2)

        return (loss_predict, loss_phi_inv, loss_unit_length)

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