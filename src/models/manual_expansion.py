import torch
import torch.nn as nn
import numpy as np


class ExpandBasis(nn.Module):
    """
    Simply expands the state
    """

    def __init__(self, state_dim=2, expansion_degree=3):
        super().__init__()

        # Basis expansion of the state
        self.expand_combinations = []
        expand_names = []
        for d in range(1, expansion_degree + 1):
            for i in range(d + 1):
                self.expand_combinations.append((i, d - i))
                expand_names.append(f"x1^{i} * x2^{d - i}")
        self.expanded_dim = len(self.expand_combinations)
        self.expand_names = expand_names


    def expand(self, x):
        expanded_features = []

        for i, j in self.expand_combinations:
            expanded_features.append((x[:, 0] ** i) * (x[:, 1] ** j))

        x_expanded = torch.stack(expanded_features, dim=1)

        return x_expanded
    

    def regression_K(self, x_expanded, x_expanded_next):
        """
        DMD regression to find K such that x_expanded_next ≈ x_expanded @ K.T
        """

        K, _ = ... # TODO
        return K
    

    def forward(self, x):
        """
        Simply expands the state
        """

        # Expand state
        x_big = self.expand(x)

        K = self.regression_K(x_big, x_big)
        x_big_next = x_big @ K.T

        return x_big_next, None, None