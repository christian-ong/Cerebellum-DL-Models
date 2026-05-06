import re
import torch
import torch.nn as nn
from itertools import product


VANDERPOL_BASIS = [
    "x",
    "y",
    "x^2*y",
    "x*y^2",
    "x^3",
    "x^4*y",
    "y^3",
    "x^3*y^2",
    "x^5",
    "x^6*y",
]

LOTKA_BASIS = [
    "x",
    "y",
    "x*y",
    "x*y^2",
    "x^2*y",
    "x*y^3",
    "x^2*y^2",
    "x^3*y",
    "x*y^4",
    "x^2*y^3",
    "x^3*y^2",
    "x^4*y",
]

PENDULUM_BASIS = [
    "x",
    "y",
    "sin(x)",
    "y*cos(x)",
    "sin(2*x)",
    "y^2*sin(x)",
    "y*cos(2*x)",
    "y^3*cos(x)",
    "sin(3*x)",
    "y^2*sin(2*x)",
    "y^4*sin(x)",
]

DUFFING_BASIS = [
    "x",
    "y",
    "x^3",
    "x^2*y",
    "x*y^2",
    "x^5",
    "y^3",
    "x^4*y",
    "x^3*y^2",
    "x^7",
    "x^2*y^3",
    "x^6*y",
]

LORENZ_BASIS = [
    "x",
    "y",
    "z",
    "x*z",
    "x*y",
    "y*z",
    "x^2*y",
    "x^2",
    "y^2",
    "x^2*z",
]

CLOSED_SMALL_BASIS = [
    "x",
    "y",
    "x^2"
]

CLOSED_LARGE_BASIS = [
    "x",
    "y",
    "x^2",
    "x^3",
    "x^4"
]

CLOSED_TRIG_SMALL_BASIS = [
    "1",
    "x",
    "y",
    "x^2",    
    "sin(x)",
    "cos(x)"
]

CLOSED_TRIG_MEDIUM_BASIS = [
    "1",
    "x",
    "y",
    "x^2",
    "sin(x)",
    "cos(x)",
    "sin(2*x)",
    "cos(2*x)"
]

CLOSED_TRIG_LARGE_BASIS = [
    "1",
    "x",
    "y",
    "x^2",    
    "sin(x)",
    "cos(x)",
    "sin(2*x)",
    "cos(2*x)",
    "sin(3*x)",
    "cos(3*x)",
]

SPECIFIC_BASES = {
    "vanderpol": VANDERPOL_BASIS,
    "lotka_volterra": LOTKA_BASIS,
    "pendulum": PENDULUM_BASIS,
    "duffing": DUFFING_BASIS,
    "lorenz": LORENZ_BASIS,
    "closed_small": CLOSED_SMALL_BASIS,
    "closed_large": CLOSED_LARGE_BASIS,
    "closed_trig_small": CLOSED_TRIG_SMALL_BASIS,
    "closed_trig_medium": CLOSED_TRIG_MEDIUM_BASIS,
    "closed_trig_large": CLOSED_TRIG_LARGE_BASIS
}


class ManualExpansion(nn.Module):
    """
    General polynomial or system-specific basis expansion.
    """

    def __init__(
        self,
        state_dim=2,
        expansion_degree=3,
        bias=True,
        sine_cosine_expansion=True,
        expansion_type="general",
        system=None,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.expansion_type = expansion_type
        self.expand_names = []
        self.expanded_basis = []
        self.max_poly_base_abs = 100.0

        if expansion_type == "specific":

            if system is None:
                raise ValueError("system must be provided when expansion_type='specific'")

            if system not in SPECIFIC_BASES:
                raise ValueError(f"No specific basis defined for system '{system}'")

            basis_list = SPECIFIC_BASES[system]

            if expansion_degree > len(basis_list):
                raise ValueError(
                    f"expansion_degree={expansion_degree} exceeds available basis size "
                    f"({len(basis_list)}) for system '{system}'"
                )

            # Strictly use the specific basis. Ignore the `bias` argument entirely.
            selected_basis = basis_list[:expansion_degree]

            self.expand_names = selected_basis
            self.expanded_basis = selected_basis

        elif expansion_type == "general":
            for exps in product(range(expansion_degree + 1), repeat=state_dim):
                total_degree = sum(exps)

                if total_degree == 0 and not bias:
                    continue

                if total_degree <= expansion_degree:
                    self.expanded_basis.append(exps)

                    name_parts = []
                    for i, e in enumerate(exps):
                        if e == 0:
                            continue

                        var = f"x{i+1}"
                        name_parts.append(var if e == 1 else f"{var}^{e}")

                    name = "*".join(name_parts) if name_parts else "1"
                    self.expand_names.append(name)

            # --------------------------------------------------
            # Sort polynomial terms:
            # first by total degree, then by exponent tuple
            # so degree-1 becomes [x1, x2] instead of [x2, x1]
            # --------------------------------------------------
            poly = list(zip(self.expanded_basis, self.expand_names))
            poly.sort(key=lambda item: (sum(item[0]),) + tuple(reversed(item[0])))

            self.expanded_basis = [b for b, _ in poly]
            self.expand_names = [n for _, n in poly]

            if sine_cosine_expansion:
                for i in range(state_dim):
                    for k in range(1, expansion_degree + 1):

                        self.expanded_basis.append(("sin", i, k))
                        if k == 1:
                            self.expand_names.append(f"sin(x{i+1})")
                        else:
                            self.expand_names.append(f"sin({k}*x{i+1})")

                        self.expanded_basis.append(("cos", i, k))
                        if k == 1:
                            self.expand_names.append(f"cos(x{i+1})")
                        else:
                            self.expand_names.append(f"cos({k}*x{i+1})")
        else:
            raise ValueError("expansion_type must be 'general' or 'specific'")

        self.expanded_dim = len(self.expand_names)

        # Precompile specific basis functions once
        if self.expansion_type == "specific":
            self._compiled_basis = [self._compile_basis(expr) for expr in self.expand_names]

        # --------------------------------------------------
        # Track where the ORIGINAL state variables are located
        # inside the expanded basis
        # --------------------------------------------------
        if self.expansion_type == "general":
            target_names = [f"x{i+1}" for i in range(self.state_dim)]
        else:
            var_names = ["x", "y", "z", "w", "v", "u"]
            target_names = var_names[:self.state_dim]

        missing = [name for name in target_names if name not in self.expand_names]
        if missing:
            raise ValueError(
                f"Could not locate original state variables {missing} in expand_names = {self.expand_names}"
            )

        self.state_indices = [self.expand_names.index(name) for name in target_names]

    def _compile_basis(self, expr):
        """
        Turn strings like 'x^2*y' or 'sin(x)' into callable functions.
        Supports constants like '1' by broadcasting them to batch shape.
        """
        expr_py = expr.replace("^", "**")

        allowed_names = {
            "sin": torch.sin,
            "cos": torch.cos,
        }

        def basis_fn(var_dict):
            local_dict = {**allowed_names, **var_dict}
            out = eval(expr_py, {"__builtins__": {}}, local_dict)

            # Broadcast scalar constants like "1" to shape (batch,)
            if not torch.is_tensor(out):
                ref = next(iter(var_dict.values()))
                out = torch.full_like(ref, float(out))

            return out
        return basis_fn
    
    def expand(self, x):
        expanded_features = []

        if self.expansion_type == "specific":
            var_names = ["x", "y", "z", "w", "v", "u"]
            var_dict = {}

            for i in range(self.state_dim):
                if i < len(var_names):
                    var_dict[var_names[i]] = x[:, i]
                else:
                    var_dict[f"x{i+1}"] = x[:, i]

            for fn in self._compiled_basis:
                expanded_features.append(fn(var_dict))

        else:
            for basis in self.expanded_basis:
                if isinstance(basis[0], int):
                    term = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

                    for dim, power in enumerate(basis):
                        if power > 0:
                            # Bound polynomial base to avoid float overflow at high degree.
                            base = torch.clamp(
                                x[:, dim],
                                min=-self.max_poly_base_abs,
                                max=self.max_poly_base_abs,
                            )
                            term = term * (base ** power)

                    expanded_features.append(term)

                else:
                    func, dim, k = basis

                    if func == "sin":
                        expanded_features.append(torch.sin(k * x[:, dim]))
                    elif func == "cos":
                        expanded_features.append(torch.cos(k * x[:, dim]))

        return torch.stack(expanded_features, dim=1)

    def de_expand(self, x_expanded):
        """
        Recover original state variables using their actual indices
        in the expanded basis.
        """
        return x_expanded[:, self.state_indices]