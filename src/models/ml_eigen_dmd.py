import torch
import torch.nn as nn


class MLEigenDMD(nn.Module):
    """
    Linear dynamics model.

    This model is EXACTLY linear end-to-end.
    """

    def __init__(self, state_dim=2):
        """

        Parameters:

            state_dim: 

            block_lambda: If True, Lambda is rotation matrix [[a, -b], [b, a]] representing conjugate eigenvalues a +/- i b.

        """
        super().__init__()

        self.state_dim = state_dim

        self.Phi = nn.Parameter(torch.eye(state_dim))
        self.Phi_inv = nn.Parameter(torch.eye(state_dim))
        self.Lambda = nn.Parameter(
            torch.tensor([[1.0, 0.0], 
                          [0.0, 1.0]]))


    def forward(self, x):
        """
        Apply one linear step using batched row-vectors.
        """
        b_t = x @ self.Phi_inv.mT # to latent space
        b_next = b_t @ self.Lambda.mT # step in latent space
        x_next = b_next @ self.Phi.mT # back to original space
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
        identity = torch.eye(self.state_dim, device=x.device, dtype=x.dtype)
        loss_phi_inv = torch.norm(self.Phi @ self.Phi_inv - identity)

        # Unit eigenvectors
        col_norms = torch.linalg.norm(self.Phi, dim=0) 
        loss_unit_length = torch.mean((col_norms - 1.0)**2)

        return (loss_predict, loss_eigvec, loss_phi_inv, loss_unit_length)
    

    def rollout(self, x0, steps):

        if not torch.is_tensor(x0):
            x0 = torch.tensor(
                x0,
                dtype=next(self.parameters()).dtype,
                device=next(self.parameters()).device
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