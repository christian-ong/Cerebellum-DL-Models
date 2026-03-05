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
        One-step prediction.

        Args:
            x: tensor of shape (batch_size, state_dim)

        Returns:
            x_next: predicted next state, shape (batch_size, state_dim)
        """

        # Apply linear latent dynamics
        b_t = self.Phi_inv @ x
        b_next = self.Lambda @ b_t
        x_next = self.Phi @ b_next

        return x_next, None, None
    

    def compute_loss(self, x, x_next_true):
        """
        Compute loss components

            Prediction loss: MSE
            Eigenvector orthorgonal: ?
            Phi and Phi_inv are inverses: check values
        """

        # Apply linear latent dynamics
        b_t = self.Phi_inv @ x
        b_next = self.Lambda @ b_t
        x_next = self.Phi @ b_next

        # Compute prediction loss (MSE)
        prediction_loss = nn.MSELoss()(x_next, x_next_true)

        # Check if eigenvectors are orthogonal
        v1 = self.Phi[:, 0]
        v2 = self.Phi[:, 1]
        orthogonality_loss = torch.abs(torch.dot(v1, v2)).item()

        # Check if Phi and Phi_inv are inverses (Phi @ Phi_inv = I)
        phi_inverses_loss = torch.norm(self.Phi @ self.Phi_inv - torch.eye(self.state_dim)).item()

        # Return loss components
        loss = prediction_loss + orthogonality_loss + phi_inverses_loss
        
        return prediction_loss, orthogonality_loss, phi_inverses_loss