import torch
import torch.nn as nn

from src.models.expander import ManualExpansion



class ExpandedEigenDMD(ManualExpansion):


    def __init__(self, state_dim=2, expansion_degree=3, **kwargs):
        """

        """
        super().__init__(state_dim, **kwargs)
        self.state_dim = state_dim

        self.latent_dim = self.expanded_dim
        self.expand_names = self.expand_names

        self.Phi = nn.Parameter(torch.eye(self.latent_dim))
        self.Phi_inv = nn.Parameter(torch.eye(self.latent_dim))
        self.Lambda = nn.Parameter(torch.eye(self.latent_dim))


    def forward(self, x):
        """
        Apply one linear step using batched row-vectors.
        """
        x_expanded = self.expand(x)
        b_t = x_expanded @ self.Phi_inv.mT # to latent space
        b_next = b_t @ self.Lambda.mT # step in latent space
        x_expanded_next = b_next @ self.Phi.mT # back to expanded space
        x_next = self.de_expand(x_expanded_next) # de-expand to original space
        return x_next
    

    def compute_loss(self, x, x_next_true):
        """
        Compute loss components

            Prediction loss: MSE
            Phi are eigenvectors: A Phi = Lambda Phi
            Phi and Phi_inv are inverses: dot product = identity
            Unit eigenvectors
        """
        x_next = self.forward(x)

        # Prediction loss, normalize to make loss scale-invariant
        actual_loss = nn.MSELoss()(x_next, x_next_true)
        step_length = torch.norm(x_next_true - x)
        loss_predict = actual_loss / (step_length + 1e-6) # avoid division by zero

        # Phi are eigenvectors: A Phi = Lambda Phi
        A = self.Phi @ self.Lambda @ self.Phi_inv
        loss_eigvec = torch.norm(A @ self.Phi - self.Phi @ self.Lambda)

        # Phi and Phi_inv are inverses
        identity = torch.eye(self.latent_dim, device=x.device, dtype=x.dtype)
        loss_phi_inv = torch.norm(self.Phi @ self.Phi_inv - identity)

        # Unit eigenvectors
        col_norms = torch.linalg.norm(self.Phi, dim=0) 
        loss_unit_length = torch.mean((col_norms - 1.0)**2)

        return (loss_predict, loss_eigvec, loss_phi_inv, loss_unit_length)
    

    def rollout(self, x0, n_steps):
        """
        Rollout trajectory from initial state x0 for n_steps.
        """
        traj = [x0]
        x = x0
        for _ in range(n_steps):
            x = self.forward(x)
            traj.append(x)
        return torch.stack(traj, dim=0)