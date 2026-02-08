import torch
import torch.nn as nn


class AELinearDynamics(nn.Module):
    """
    Linear autoencoder with linear latent dynamics.

    This model is EXACTLY linear end-to-end.

    Mathematical form:
        z_t     = E x_t
        z_{t+1} = K z_t
        x_{t+1} = D z_{t+1}

    where:
        E ∈ R^{latent_dim x state_dim}
        K ∈ R^{latent_dim x latent_dim}
        D ∈ R^{state_dim x latent_dim}

    The full dynamics in state space are:
        x_{t+1} = (D K E) x_t

    This is equivalent to learning a single linear operator
    M = D K E, but with a structured factorization.
    """

    def __init__(self, state_dim=2, latent_dim=2):
        super().__init__()

        # --------------------------------------------------
        # Encoder: x -> z
        # --------------------------------------------------
        # Linear map from state space to latent space
        #
        # x ∈ R^{state_dim}
        # z ∈ R^{latent_dim}
        #
        # No bias:
        # - keeps the mapping strictly linear
        # - avoids introducing affine offsets
        # --------------------------------------------------

        self.encoder = nn.Linear(
            in_features=state_dim,
            out_features=latent_dim,
            bias=False,
        )

        # --------------------------------------------------
        # Latent dynamics: z_t -> z_{t+1}
        # --------------------------------------------------
        # Linear Koopman operator in latent space
        #
        # z_{t+1} = K z_t
        #
        # No bias:
        # - represents a pure linear operator
        # - matches the true linear system structure
        # --------------------------------------------------

        self.K = nn.Linear(
            in_features=latent_dim,
            out_features=latent_dim,
            bias=False,
        )

        # --------------------------------------------------
        # Decoder: z -> x
        # --------------------------------------------------
        # Linear map from latent space back to state space
        #
        # z ∈ R^{latent_dim}
        # x ∈ R^{state_dim}
        #
        # No bias for strict linearity
        # --------------------------------------------------

        self.decoder = nn.Linear(
            in_features=latent_dim,
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
            z: latent representation of x_t, shape (batch_size, latent_dim)
            z_next: latent representation of x_{t+1}, shape (batch_size, latent_dim)
        """

        # Encode current state into latent coordinates
        z = self.encoder(x)

        # Apply linear latent dynamics
        z_next = self.K(z)

        # Decode latent state back to state space
        x_next = self.decoder(z_next)

        return x_next, z, z_next
