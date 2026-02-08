import torch.nn as nn

class AEKoopmanDynamics(nn.Module):
    """
    Nonlinear autoencoder with linear latent dynamics.

    Structure:
        x_t  -> encoder (nonlinear) -> z_t
        z_t  -> K (linear)          -> z_{t+1}
        z_{t+1} -> decoder (nonlinear) -> x_{t+1}

    This is the simplest nonlinear extension of the linear AE model.
    """

    def __init__(self, state_dim, latent_dim, hidden_dim=64):
        super().__init__()

        # --------------------------------------------------
        # Encoder: x -> z
        # --------------------------------------------------
        # Maps state space to latent space
        #
        # x ∈ R^{state_dim}
        # z ∈ R^{latent_dim}
        #
        # We use a small MLP with tanh nonlinearity.
        # Tanh is often better-behaved for dynamical systems.
        # --------------------------------------------------

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # --------------------------------------------------
        # Latent dynamics: z_t -> z_{t+1}
        # --------------------------------------------------
        # This is the Koopman-style linear operator.
        #
        # z_{t+1} = K z_t
        #
        # No bias: we want a pure linear map.
        # --------------------------------------------------

        self.K = nn.Linear(latent_dim, latent_dim, bias=False)

        # --------------------------------------------------
        # Decoder: z -> x
        # --------------------------------------------------
        # Maps latent space back to state space
        #
        # z ∈ R^{latent_dim}
        # x ∈ R^{state_dim}
        # --------------------------------------------------

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x):
        """
        One-step prediction.

        Args:
            x: tensor of shape (batch_size, state_dim)

        Returns:
            x_next: predicted x_{t+1}
            z: latent state z_t
            z_next: latent state z_{t+1}
        """

        # Encode current state
        z = self.encoder(x)

        # Linear latent dynamics
        z_next = self.K(z)

        # Decode next latent state
        x_next = self.decoder(z_next)

        return x_next, z, z_next
