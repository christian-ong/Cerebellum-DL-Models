import torch
import torch.nn as nn


class MLEigenDMD(nn.Module):
    """
    Linear dynamics model.

    This model is EXACTLY linear end-to-end.
    """

    def __init__(self, state_dim=2):
        super().__init__()

        self.state_dim = state_dim

        # 2x2 full, initialized to identity
        self.Phi = nn.Parameter(
            torch.eye(state_dim),
        )

        self.Lambda = nn.Parameter(
            torch.eye(state_dim),
        )

        self.Phi_inv = nn.Parameter(
            torch.eye(state_dim),
        )

    def forward(self, x):
        """
        Apply one linear step using batched row-vectors.
        """

        # x is represented as row-vectors, so parameters are applied on the right.
        b_t = x @ self.Phi_inv.T
        b_next = b_t @ self.Lambda.T
        x_next = b_next @ self.Phi.T
        return x_next
    

    def compute_loss(self, x, x_next_true):
        """
        Compute loss components

            Prediction loss: MSE
            Eigenvector orthorgonal: dot product = zero
            Phi and Phi_inv are inverses: dot product = identity
        """

        x_next = self.forward(x)

        # Prediction loss
        loss_predict = nn.MSELoss()(x_next, x_next_true)

        # Eigenvectors orthogonal
        v1 = self.Phi[:, 0]
        v2 = self.Phi[:, 1]
        loss_orthogonal = torch.abs(torch.dot(v1, v2))

        # Phi and Phi_inv are inverses
        identity = torch.eye(self.state_dim, device=x.device, dtype=x.dtype)
        loss_phi_inv = torch.norm(self.Phi @ self.Phi_inv - identity)

        # Return a scalar training loss

        return (loss_predict, loss_orthogonal, loss_phi_inv)