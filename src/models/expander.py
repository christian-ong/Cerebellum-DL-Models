import torch
import torch.nn as nn
import numpy as np


class ManualExpansion(nn.Module):
    """
    Class to handle manual basis expansions, general and specific
    """

    def __init__(
            self, 
            state_dim=2, 
            expansion_degree=3, 
            constant_expansion=True, 
            sine_cosine_expansion=True
        ):

        super().__init__()

        self.expanded_basis = []
        self.expand_names = []

        # Constant term
        if constant_expansion:
            self.expanded_basis = [(0, 0)]
            self.expand_names = ["1"]

        # Polynomial expansion of the state 
        for d in range(1, expansion_degree + 1):
            for i in range(d + 1):
                e_x = d-i
                e_y = i
                self.expanded_basis.append((e_x, e_y))
                expansion_string = f"x^{e_x} y^{e_y}".replace("x^0", "").replace("y^0", "").replace("^1", "").strip()
                self.expand_names.append(expansion_string)
        self.expanded_dim = len(self.expanded_basis)

        # Sine and cosine expansion of the state
        if sine_cosine_expansion:
            for var in ["x", "y"]:
                for func in ["sin", "cos"]:
                    self.expanded_basis.append((func, var))
                    self.expand_names.append(f"{func}({var})")
            self.expanded_dim += 4


    def expand(self, x):
        expanded_features = []

        for i, j in self.expanded_basis:
            if isinstance(i, int) and isinstance(j, int):
                expanded_features.append((x[:, 0] ** i) * (x[:, 1] ** j))
            elif isinstance(i, str) and isinstance(j, str):
                if i == "sin":
                    if j == "x":
                        expanded_features.append(torch.sin(x[:, 0]))
                    elif j == "y":
                        expanded_features.append(torch.sin(x[:, 1]))
                elif i == "cos":
                    if j == "x":
                        expanded_features.append(torch.cos(x[:, 0]))
                    elif j == "y":
                        expanded_features.append(torch.cos(x[:, 1]))

        x_expanded = torch.stack(expanded_features, dim=1)

        return x_expanded
    

    def de_expand(self, x_expanded):
        # Extract original state (x,y) from expanded features - take features 2 and 3 (1st term is a constant)
        x_reconstructed = x_expanded[:, 1:3]
        return x_reconstructed