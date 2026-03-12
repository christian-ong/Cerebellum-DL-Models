import torch
import torch.nn as nn

from src.models.expander import ManualExpansion


class ManualExpansion_EigenDMD(ManualExpansion):

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

        self.Phi = nn.Parameter(torch.eye(self.latent_dim))
        self.Phi_inv = nn.Parameter(torch.eye(self.latent_dim))
        self.Lambda = nn.Parameter(torch.eye(self.latent_dim))

    # ------------------------------------------------
    # Forward
    # ------------------------------------------------

    def forward(self, x):

        x_expanded = self.expand(x)

        b_t = x_expanded @ self.Phi_inv.mT
        b_next = b_t @ self.Lambda.mT
        x_expanded_next = b_next @ self.Phi.mT

        x_next = self.de_expand(x_expanded_next)

        return x_next

    # ------------------------------------------------
    # Loss
    # ------------------------------------------------

    def compute_loss(self, x, x_next_true):

        x_next = self.forward(x)

        actual_loss = nn.MSELoss()(x_next, x_next_true)
        step_length = torch.norm(x_next_true - x)
        loss_predict = actual_loss / (step_length + 1e-6)

        A = self.Phi @ self.Lambda @ self.Phi_inv
        loss_eigvec = torch.norm(A @ self.Phi - self.Phi @ self.Lambda)

        identity = torch.eye(self.latent_dim, device=x.device, dtype=x.dtype)
        loss_phi_inv = torch.norm(self.Phi @ self.Phi_inv - identity)

        col_norms = torch.linalg.norm(self.Phi, dim=0)
        loss_unit_length = torch.mean((col_norms - 1.0) ** 2)

        return (loss_predict, loss_eigvec, loss_phi_inv, loss_unit_length)

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