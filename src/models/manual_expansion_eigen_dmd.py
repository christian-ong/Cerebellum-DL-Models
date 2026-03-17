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

        # eigenvectors
        self.Phi = nn.Parameter(
            torch.eye(self.latent_dim) + 0.01 * torch.randn(self.latent_dim, self.latent_dim)
        )
        # eigenvalue matrix
        self.Lambda = nn.Parameter(
            torch.eye(self.latent_dim) + 0.01 * torch.randn(self.latent_dim, self.latent_dim)
        )

    # ------------------------------------------------
    # Forward
    # ------------------------------------------------

    def forward(self, x):

        z = self.expand(x)

        Phi_inv = torch.linalg.inv(self.Phi)

        b = z @ Phi_inv.mT
        b_next = b @ self.Lambda.mT
        z_next = b_next @ self.Phi.mT

        x_next = self.de_expand(z_next)

        return x_next

    # ------------------------------------------------
    # Loss
    # ------------------------------------------------

    def compute_loss(self, x, x_next_true):

        z = self.expand(x)
        z_next_true = self.expand(x_next_true)

        Phi_inv = torch.linalg.inv(self.Phi)

        b = z @ Phi_inv.mT
        b_next = b @ self.Lambda.mT
        z_next_pred = b_next @ self.Phi.mT

        # --------------------------------------------------
        # 1) Prediction loss in lifted space
        # --------------------------------------------------

        loss_lift = nn.MSELoss()(z_next_pred, z_next_true)

        # --------------------------------------------------
        # 2) Φ conditioning regularization
        # --------------------------------------------------

        Phi_inv_norm = torch.norm(Phi_inv)

        # --------------------------------------------------
        # 3) column normalization for Φ
        # --------------------------------------------------

        col_norms = torch.linalg.norm(self.Phi, dim=0)
        loss_unit_length = torch.mean((col_norms - 1.0) ** 2)

        loss = (
            loss_lift
            + 1e-4 * Phi_inv_norm
            + 1e-3 * loss_unit_length
        )

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

            x = self.forward(x)

            traj.append(x.squeeze(0))

        return torch.stack(traj)