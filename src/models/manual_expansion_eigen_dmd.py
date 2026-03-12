import torch
import torch.nn as nn
from itertools import product


class ManualExpansion_EigenDMD(nn.Module):
    """
    Eigen-DMD model with manual polynomial expansion.

    Works for arbitrary state dimension.
    """

    def __init__(
        self,
        state_dim=2,
        expansion_degree=2,
        constant_expansion=True
    ):

        super().__init__()

        self.state_dim = state_dim

        self.expanded_basis = []
        self.expand_names = []

        # ------------------------------------------------
        # Build polynomial basis
        # ------------------------------------------------

        for exps in product(range(expansion_degree + 1), repeat=state_dim):

            total_degree = sum(exps)

            if total_degree == 0 and not constant_expansion:
                continue

            if total_degree <= expansion_degree:

                self.expanded_basis.append(exps)

                # build readable name
                name_parts = []

                for i, e in enumerate(exps):

                    if e == 0:
                        continue

                    var = f"x{i+1}"

                    if e == 1:
                        name_parts.append(var)
                    else:
                        name_parts.append(f"{var}^{e}")

                name = " ".join(name_parts) if name_parts else "1"

                self.expand_names.append(name)

        self.expanded_dim = len(self.expanded_basis)

        # ------------------------------------------------
        # Eigen-parameterized Koopman operator
        # ------------------------------------------------

        self.Phi = nn.Parameter(torch.eye(self.expanded_dim))
        self.Phi_inv = nn.Parameter(torch.eye(self.expanded_dim))
        self.Lambda = nn.Parameter(torch.eye(self.expanded_dim))

    # ------------------------------------------------
    # Expansion
    # ------------------------------------------------

    def expand(self, x):

        expanded_features = []

        for exps in self.expanded_basis:

            term = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

            for dim, power in enumerate(exps):

                if power > 0:
                    term = term * (x[:, dim] ** power)

            expanded_features.append(term)

        return torch.stack(expanded_features, dim=1)

    # ------------------------------------------------
    # De-expand
    # ------------------------------------------------

    def de_expand(self, x_expanded):
        """
        Recover original state variables.

        Assumes linear terms appear directly after constant.
        """

        start = 1 if self.expanded_basis[0] == tuple([0]*self.state_dim) else 0
        end = start + self.state_dim

        return x_expanded[:, start:end]

    # ------------------------------------------------
    # Forward
    # ------------------------------------------------

    def forward(self, x):

        x_big = self.expand(x)

        b_t = x_big @ self.Phi_inv.mT
        b_next = b_t @ self.Lambda.mT
        x_big_next = b_next @ self.Phi.mT

        x_next = self.de_expand(x_big_next)

        return x_next

    # ------------------------------------------------
    # Loss
    # ------------------------------------------------

    def compute_loss(self, x, x_next_true):

        x_next = self.forward(x)

        # prediction loss
        actual_loss = nn.MSELoss()(x_next, x_next_true)
        step_length = torch.norm(x_next_true - x)
        loss_predict = actual_loss / (step_length + 1e-6)

        # eigenvector constraint
        A = self.Phi @ self.Lambda @ self.Phi_inv
        loss_eigvec = torch.norm(A @ self.Phi - self.Phi @ self.Lambda)

        # inverse constraint
        identity = torch.eye(self.expanded_dim, device=x.device)
        loss_phi_inv = torch.norm(self.Phi @ self.Phi_inv - identity)

        # unit eigenvectors
        col_norms = torch.linalg.norm(self.Phi, dim=0)
        loss_unit_length = torch.mean((col_norms - 1.0)**2)

        return (loss_predict, loss_eigvec, loss_phi_inv, loss_unit_length)
    
    # ------------------------------------------------
    # Rollout
    # ------------------------------------------------
    
    def rollout(self, x0, steps):
        """
        Rollout trajectory from initial state x0.
        """

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