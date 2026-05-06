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

import math
from typing import Optional

import torch
import torch.nn as nn


class RBFExpansion(nn.Module):
    """
    Radial-basis-function expansion with the same public interface style as ManualExpansion.

    Feature vector:
        [optional bias, raw state coordinates, RBF features]

    This keeps compatibility with the existing EDMD-style models, which expect:
        - expanded_dim
        - expand_names
        - state_indices
        - expand(x)
        - de_expand(x_expanded)

    Notes
    -----
    - The RBF dictionary is data-dependent and must be fitted on training states.
    - Centers are chosen from the training set.
    - Widths (sigmas) can be global or per-center.
    """

    def __init__(
        self,
        state_dim: int,
        n_centers: int = 50,
        bias: bool = True,
        include_state: bool = True,
        center_selection: str = "farthest",
        bandwidth_mode: str = "knn",
        knn_k: int = 5,
        global_sigma_scale: float = 1.0,
        min_sigma: float = 1e-6,
    ):
        super().__init__()

        if state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if n_centers <= 0:
            raise ValueError("n_centers must be positive.")
        if center_selection not in {"random", "farthest"}:
            raise ValueError("center_selection must be one of {'random', 'farthest'}.")
        if bandwidth_mode not in {"global", "knn"}:
            raise ValueError("bandwidth_mode must be one of {'global', 'knn'}.")
        if knn_k <= 0:
            raise ValueError("knn_k must be positive.")

        self.state_dim = state_dim
        self.n_centers = n_centers
        self.bias = bias
        self.include_state = include_state
        self.center_selection = center_selection
        self.bandwidth_mode = bandwidth_mode
        self.knn_k = knn_k
        self.global_sigma_scale = global_sigma_scale
        self.min_sigma = min_sigma

        # Data-fit attributes
        self.is_fitted = False

        # Buffers so they move with device and get saved in checkpoints.
        # Important: initialize with final shapes so load_state_dict can restore
        # trained RBF checkpoints without shape mismatch.
        self.register_buffer("centers", torch.zeros(n_centers, state_dim, dtype=torch.float32))
        self.register_buffer("sigmas", torch.ones(n_centers, dtype=torch.float32))

        # Public interface expected by current models
        self.expand_names = []
        self.state_indices = []
        self.expanded_dim = 0

        self._build_feature_metadata()

    def _build_feature_metadata(self):
        names = []
        state_indices = []

        if self.bias:
            names.append("1")

        if self.include_state:
            for i in range(self.state_dim):
                state_indices.append(len(names))
                names.append(f"x{i+1}")

        for j in range(self.n_centers):
            names.append(f"rbf_{j}")

        self.expand_names = names
        self.state_indices = state_indices
        self.expanded_dim = len(names)

        if self.include_state and len(self.state_indices) != self.state_dim:
            raise RuntimeError("Failed to assign state_indices correctly.")

    def _to_2d_tensor(self, x) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(dtype=torch.float32)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        if x.ndim != 2 or x.shape[1] != self.state_dim:
            raise ValueError(f"Expected input shape (N, {self.state_dim}), got {tuple(x.shape)}.")

        return x

    def _pairwise_sq_dists(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # X: (N, d), C: (M, d)
        # returns (N, M) with squared Euclidean distances
        return torch.cdist(X, C, p=2) ** 2

    def _select_centers_random(self, X: torch.Tensor) -> torch.Tensor:
        n = X.shape[0]
        if self.n_centers > n:
            raise ValueError(f"n_centers={self.n_centers} exceeds number of training points={n}.")
        perm = torch.randperm(n, device=X.device)
        idx = perm[: self.n_centers]
        return X[idx].clone()

    def _select_centers_farthest(self, X: torch.Tensor) -> torch.Tensor:
        """
        Greedy farthest-point sampling from the training set.
        Spreads centers out over the data cloud.
        """
        n = X.shape[0]
        if self.n_centers > n:
            raise ValueError(f"n_centers={self.n_centers} exceeds number of training points={n}.")

        # Start from a random point
        first_idx = torch.randint(low=0, high=n, size=(1,), device=X.device).item()
        selected = [first_idx]

        min_d2 = self._pairwise_sq_dists(X, X[first_idx:first_idx + 1]).squeeze(1)

        for _ in range(1, self.n_centers):
            next_idx = torch.argmax(min_d2).item()
            selected.append(next_idx)

            d2_new = self._pairwise_sq_dists(X, X[next_idx:next_idx + 1]).squeeze(1)
            min_d2 = torch.minimum(min_d2, d2_new)

        idx = torch.tensor(selected, device=X.device, dtype=torch.long)
        return X[idx].clone()

    def _compute_global_sigma(self, centers: torch.Tensor) -> torch.Tensor:
        """
        One shared scale from center-center distances.
        Stored as a per-center vector for uniform downstream handling.
        """
        if centers.shape[0] == 1:
            sigma = torch.tensor([1.0], device=centers.device, dtype=centers.dtype)
            return sigma

        D = torch.cdist(centers, centers, p=2)
        mask = ~torch.eye(D.shape[0], device=D.device, dtype=torch.bool)
        nonzero = D[mask]

        sigma_val = torch.median(nonzero)
        sigma_val = torch.clamp(self.global_sigma_scale * sigma_val, min=self.min_sigma)

        return torch.full(
            (centers.shape[0],),
            fill_value=sigma_val.item(),
            device=centers.device,
            dtype=centers.dtype,
        )

    def _compute_knn_sigmas(self, centers: torch.Tensor) -> torch.Tensor:
        """
        Per-center sigma_j from the distance to the k-th nearest center.
        """
        m = centers.shape[0]
        if m == 1:
            return torch.tensor([1.0], device=centers.device, dtype=centers.dtype)

        D = torch.cdist(centers, centers, p=2)

        # Ignore self-distance safely
        eye_mask = torch.eye(m, device=D.device, dtype=torch.bool)
        D = D.masked_fill(eye_mask, float("inf"))

        k = min(self.knn_k, m - 1)
        knn_dists, _ = torch.topk(D, k=k, largest=False, dim=1)

        # Use the k-th nearest distance
        sigmas = knn_dists[:, -1]
        sigmas = torch.clamp(sigmas, min=self.min_sigma)
        return sigmas

    def fit(self, X_train) -> "RBFExpansion":
        """
        Fit centers and bandwidths from training states.

        Parameters
        ----------
        X_train : array-like or tensor, shape (N, d)
            Training states used to define the RBF dictionary.
        """
        X = self._to_2d_tensor(X_train)

        if self.center_selection == "random":
            centers = self._select_centers_random(X)
        elif self.center_selection == "farthest":
            centers = self._select_centers_farthest(X)
        else:
            raise RuntimeError("Unexpected center_selection branch.")

        if self.bandwidth_mode == "global":
            sigmas = self._compute_global_sigma(centers)
        elif self.bandwidth_mode == "knn":
            sigmas = self._compute_knn_sigmas(centers)
        else:
            raise RuntimeError("Unexpected bandwidth_mode branch.")

        if not torch.isfinite(centers).all():
            raise ValueError("RBF centers contain non-finite values after fitting.")
        if not torch.isfinite(sigmas).all():
            raise ValueError("RBF sigmas contain non-finite values after fitting.")
        
        self.centers.copy_(centers.to(self.centers.dtype))
        self.sigmas.copy_(sigmas.to(self.sigmas.dtype))
        self.is_fitted = True
        return self

    def _rbf_features(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("RBFExpansion must be fitted before calling expand().")

        # squared distances: (N, M)
        d2 = self._pairwise_sq_dists(x, self.centers)

        # sigmas: (M,) -> (1, M)
        sigma2 = torch.clamp(self.sigmas, min=self.min_sigma).unsqueeze(0) ** 2

        # Gaussian RBFs
        return torch.exp(-0.5 * d2 / sigma2)

    def expand(self, x) -> torch.Tensor:
        """
        Build expanded features:
            [optional bias, raw state, gaussian RBF features]
        """
        x = self._to_2d_tensor(x)
        feats = []

        if self.bias:
            feats.append(torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device))

        if self.include_state:
            feats.append(x)

        feats.append(self._rbf_features(x))

        return torch.cat(feats, dim=1)

    def de_expand(self, x_expanded: torch.Tensor) -> torch.Tensor:
        """
        Recover original state variables from the expanded vector.
        This mirrors the ManualExpansion interface used by current models.
        """
        if not self.include_state:
            raise RuntimeError("de_expand() requires include_state=True.")

        return x_expanded[:, self.state_indices]

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, n_centers={self.n_centers}, "
            f"bias={self.bias}, include_state={self.include_state}, "
            f"center_selection='{self.center_selection}', "
            f"bandwidth_mode='{self.bandwidth_mode}', knn_k={self.knn_k}"
        )

def build_expander(
    *,
    state_dim: int,
    expansion_type: str,
    expansion_degree: int = 3,
    bias: bool = True,
    sine_cosine_expansion: bool = False,
    system: Optional[str] = None,
    rbf_n_centers: int = 50,
    rbf_center_selection: str = "farthest",
    rbf_bandwidth_mode: str = "knn",
    rbf_knn_k: int = 5,
):
    if expansion_type == "rbf":
        return RBFExpansion(
            state_dim=state_dim,
            n_centers=rbf_n_centers,
            bias=bias,
            include_state=True,
            center_selection=rbf_center_selection,
            bandwidth_mode=rbf_bandwidth_mode,
            knn_k=rbf_knn_k,
        )

    return ManualExpansion(
        state_dim=state_dim,
        expansion_degree=expansion_degree,
        bias=bias,
        sine_cosine_expansion=sine_cosine_expansion,
        expansion_type=expansion_type,
        system=system,
    )