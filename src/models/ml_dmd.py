import torch
import torch.nn as nn


class ML_DMD(nn.Module):
    """
    Linear dynamics model.

    This model is EXACTLY linear end-to-end.

    Mathematical form:
        x_{t+1} = K x_t

    where:
        K ∈ R^{state_dim x state_dim}
    
    """

    def __init__(self, state_dim=2):
        super().__init__()

        # --------------------------------------------------
        # Latent dynamics: x_t -> x_{t+1}
        # --------------------------------------------------
        # Linear Koopman operator in latent space
        #
        # x_{t+1} = K x_t
        #
        # No bias:
        # - represents a pure linear operator
        # - matches the true linear system structure
        # --------------------------------------------------

        self.K = nn.Linear(
            in_features=state_dim,
            out_features=state_dim,
            bias=False,
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
        x_next = self.K(x)

        return x_next, None, None
