import os
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import textwrap


from src.eval.model_io import predict_rollout_from_x0
from src.eval.model_io import supports_mode_subset_rollout
from src.eval.noise_robustness import compute_mode_diagnostics
from src.eval.delay_utils import (
    get_model_delay_depth,
    make_backward_delay_x0_from_current_states,
)
from src.data_generation.data_simulation import (
    simulate,
    linear_system,
    vanderpol_system,
    lotka_volterra_system,
    pendulum_system,
    lorenz_system,
    duffing_system,
    closed_small_system,
    closed_large_system,
    closed_trig_small_system,
    closed_trig_medium_system,
    closed_trig_large_system,
)


# ============================================================
# Basic helpers
# ============================================================

def parse_int_list(text: str) -> List[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("At least one integer must be provided.")
    return sorted(set(values))


def _np_scalar(data, key: str, default=None):
    if key not in data:
        if default is not None:
            return default
        raise KeyError(f"Missing key '{key}' in dataset.")
    arr = np.asarray(data[key])
    return arr.item() if arr.shape == () else arr


def _pretty_system_name(system: str) -> str:
    special = {
        "vanderpol": "Van der Pol",
        "lotka_volterra": "Lotka–Volterra",
        "closed_small": "Closed Small",
        "closed_large": "Closed Large",
        "closed_trig_small": "Closed Trig Small",
        "closed_trig_medium": "Closed Trig Medium",
        "closed_trig_large": "Closed Trig Large",
    }
    return special.get(system, system.replace("_", " ").title())


def get_phase_dims(system: str, state_dim: int) -> Tuple[int, int]:
    if system == "lorenz" and state_dim >= 3:
        return 0, 2
    if state_dim < 2:
        raise ValueError("Heatmaps require state_dim >= 2.")
    return 0, 1


def _infer_mode_count(model_name: str, model, extras: Dict[str, Any]) -> Optional[int]:
    """Best-effort inference of the available modal dimension for mode-subset comparisons."""
    if model_name not in {"regression_dmd", "ml_dmd", "ml_dmd_drop"}:
        return None

    candidates = [
        getattr(model, "expanded_dim", None),
        getattr(model, "latent_dim", None),
        getattr(model, "rank", None),
    ]

    if hasattr(model, "Phi_lift_fitted") and getattr(model, "Phi_lift_fitted", None) is not None:
        try:
            candidates.append(int(model.Phi_lift_fitted.shape[1]))
        except Exception:
            pass

    train_args = _get_train_args(extras)
    for key in ("rank", "latent_dim", "expanded_dim", "expansion_degree"):
        if key in train_args:
            candidates.append(train_args.get(key))

    for value in candidates:
        try:
            if value is None:
                continue
            count = int(value)
            if count > 0:
                return count
        except Exception:
            continue

    return None


def _get_expanded_indices(mode_indices, model):
    if mode_indices is None or len(mode_indices) == 0:
        return mode_indices
        
    expanded_idx = set(mode_indices)
    
    # 1. Complex diagonal models (Regression DMD)
    if hasattr(model, "Lambda_fitted"):
        L = model.Lambda_fitted.detach().cpu().numpy()
        L_diag = np.diag(L) if L.ndim == 2 else L
        if np.iscomplexobj(L_diag):
            for i in mode_indices:
                if abs(L_diag[i].imag) > 1e-6:
                    diffs = np.abs(L_diag - L_diag[i].conj())
                    diffs[i] = np.inf
                    conj_idx = int(np.argmin(diffs))
                    if diffs[conj_idx] < 1e-4:
                        expanded_idx.add(conj_idx)
                        
    # 2. Real block matrices (ML DMD)
    elif hasattr(model, "Lambda"):
        L = model.Lambda.detach().cpu().numpy()
        if L.ndim == 2:
            # ---> REAL FIX: Stop grouping the entire tridiagonal matrix! <---
            # Look ONLY for adjacent 2x2 rotation blocks (complex pairs) by 
            # checking for skew-symmetric off-diagonals and similar diagonals.
            for i in list(expanded_idx):
                # Check forward pair
                if i < L.shape[0] - 1:
                    a, b = L[i, i], L[i, i+1]
                    c, d = L[i+1, i], L[i+1, i+1]
                    if abs(b) > 1e-3 and abs(c) > 1e-3 and np.sign(b) != np.sign(c) and abs(a - d) < 1e-3:
                        expanded_idx.add(i+1)
                # Check backward pair
                if i > 0:
                    a, b = L[i-1, i-1], L[i-1, i]
                    c, d = L[i, i-1], L[i, i]
                    if abs(b) > 1e-3 and abs(c) > 1e-3 and np.sign(b) != np.sign(c) and abs(a - d) < 1e-3:
                        expanded_idx.add(i-1)
            
    return sorted(list(expanded_idx))

def _mode_subset_indices_for_fraction(diag, fraction, total_modes, model):
    if fraction is None or total_modes <= 0:
        return None, None, None

    raw_fraction = float(fraction)
    pct_value = raw_fraction * 100.0 if raw_fraction <= 1.0 else raw_fraction
    pct_value = float(np.clip(pct_value, 0.0, 100.0))
    frac = pct_value / 100.0

    if np.isclose(pct_value, round(pct_value)):
        pct_label = str(int(round(pct_value)))
    else:
        pct_label = (f"{pct_value:.3f}".rstrip("0").rstrip(".")).replace(".", "p")

    n_modes = int(np.ceil(frac * total_modes))
    n_modes = min(max(n_modes, 1), total_modes)
    
    # --- FIX: Pass model instead of diag.get("lambdas") ---
    raw_idx = np.asarray(diag["order_contrib"][:n_modes], dtype=int)
    expanded_idx = _get_expanded_indices(raw_idx, model)  # <-- Changed
    
    return np.asarray(expanded_idx, dtype=int), pct_label, len(expanded_idx)


def _mode_subset_specs_for_fractions(diag, fractions, total_modes, model):
    specs = []
    by_mode_count = {}
    contrib = np.asarray(diag.get("state_contribution", []), dtype=float)
    total_contrib = float(np.sum(contrib)) if contrib.size > 0 else 0.0

    for fraction in fractions or []:
        contrib_idx, pct_label, n_modes = _mode_subset_indices_for_fraction(diag, fraction, total_modes, model)
        if contrib_idx is None or pct_label is None:
            continue

        if total_contrib > 0 and np.isfinite(total_contrib):
            actual_score = float(np.sum(contrib[np.asarray(contrib_idx, dtype=int)]) / total_contrib)
        else:
            actual_score = None

        spec = by_mode_count.get(n_modes)
        if spec is None:
            spec = {
                "pct_labels": [pct_label],
                "mode_indices": contrib_idx,
                "n_modes": n_modes,
                "actual_score": actual_score,
            }
            by_mode_count[n_modes] = spec
            specs.append(spec)
        else:
            spec["pct_labels"].append(pct_label)
            if spec.get("actual_score") is None and actual_score is not None:
                spec["actual_score"] = actual_score

    return specs


def _build_mode_subset_heatmap_specs(
    *,
    model_name: str,
    model,
    extras: Dict[str, Any],
    mode_subset_thresholds: Optional[List[float]],
    mode_subset_indices: Optional[List[int]],
    X_states: Optional[np.ndarray] = None,
) -> List[Dict[str, object]]:
    """Build heatmap columns for mode-subset comparisons and append the full model last."""
    specs: List[Dict[str, object]] = []
    total_modes = _infer_mode_count(model_name, model, extras)

    if total_modes is None or total_modes <= 0:
        return [{"name": "all", "title": "Full model", "mode_indices": None}]

    requested_thresholds = [float(t) for t in (mode_subset_thresholds or []) if float(t) > 0]

    if mode_subset_indices:
        if mode_subset_indices:
            idx = [int(i) for i in mode_subset_indices if 0 <= int(i) < total_modes]
            if idx:
                specs.append({
                    "name": "manual",
                    "title": f"Manual modes ({len(idx)})",
                    "mode_indices": np.asarray(idx, dtype=int),
                })
        return specs + [{"name": "all", "title": "Full model", "mode_indices": None}]

    if not requested_thresholds:
        requested_thresholds = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]

    contrib_order = None
    diag = None
    try:
        if X_states is not None:
            diag = compute_mode_diagnostics(model, X_states)
            contrib_order = diag.get("order_contrib", None)
    except Exception:
        diag = None
        contrib_order = None

    if diag is None or contrib_order is None:
        return [{"name": "all", "title": "Full model", "mode_indices": None}]

    for spec in _mode_subset_specs_for_fractions(diag, requested_thresholds, total_modes, model):
        pct_text = ", ".join(f"{pct}%" for pct in spec["pct_labels"])
        actual_score = spec.get("actual_score")
        if actual_score is not None:
            title = f"{spec['n_modes']} Modes ({pct_text}) | C={actual_score:.3f}"
        else:
            title = f"{spec['n_modes']} Modes ({pct_text})"
        specs.append({
            "name": f"pct_{spec['n_modes']}",
            "title": title,
            "mode_indices": np.asarray(spec["mode_indices"], dtype=int),
        })

    if not specs or specs[-1]["name"] != "all":
        # FIXED: Add the total_modes count to the "Full model" title
        specs.append({
            "name": "all", 
            "title": f"Full model ({total_modes} modes)", 
            "mode_indices": None
        })

    return specs


def _pretty_model_name(model_name: str) -> str:
    names = {
        "linear_baseline": "LSTSQ",
        "dmd_baseline": "DMD",
        "regression_dmd": "EDMD",
        "ml_lineardynamics": "NN-LINOP",
        "ml_dmd": "NN-EDMD",
        "ml_dmd_drop": "NN-EDMD",
        "hardcoded_dmd": "DMD",
        "mlp_baseline": "MLP",
        "sindy_baseline": "SINDy",
    }
    return names.get(model_name, model_name.replace("_", " ").title())


def _clean_label_value(value):
    """
    Convert checkpoint/model values to compact printable values.
    Handles numpy scalars, torch scalars, strings, booleans, and None.
    """
    if value is None:
        return None

    try:
        import torch
        if torch.is_tensor(value):
            if value.numel() == 1:
                value = value.detach().cpu().item()
            else:
                return None
    except Exception:
        pass

    try:
        arr = np.asarray(value)
        if arr.shape == ():
            value = arr.item()
    except Exception:
        pass

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    if isinstance(value, str):
        if value.strip() == "":
            return None
        return value

    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return f"{float(value):.2g}"

    return str(value)


def _get_train_args(extras: Dict[str, Any]) -> Dict[str, Any]:
    """
    ML checkpoints are loaded into extras['ckpt'] in model_io.py.
    Their hyperparameters usually live under ckpt['train_args'].
    """
    ckpt = extras.get("ckpt", None)
    if isinstance(ckpt, dict):
        train_args = ckpt.get("train_args", {})
        if isinstance(train_args, dict):
            return train_args
    return {}


def _first_nonempty(*values):
    for value in values:
        value = _clean_label_value(value)
        if value is not None:
            return value
    return None


def _pretty_expansion_type(expansion_type: Any) -> Optional[str]:
    value = _first_nonempty(expansion_type)
    if value is None:
        return None

    normalized = str(value).strip().lower()
    mapping = {
        "general": "General",
        "specific": "Specific",
        "rbf": "RBF",
        "hankel": "Hankel",
        "hankel_svd": "Hankel",
    }
    return mapping.get(normalized, str(value).replace("_", " ").title())


def _format_expansion_parameters(model_name: str, model, train_args: Dict[str, Any], expansion_type: Optional[str]) -> List[str]:
    params: List[str] = []
    # Normalize expansion_type for case-insensitive checks
    expansion_type_norm = str(expansion_type).strip().lower() if expansion_type is not None else ""

    if expansion_type_norm in {"general", "specific"}:
        degree = _first_nonempty(
            getattr(model, "expansion_degree", None),
            train_args.get("expansion_degree", None),
            train_args.get("degree", None),
        )
        if degree is not None:
            params.append(f"Degree {degree}")
    if expansion_type_norm == "rbf":
        bandwidth_mode = _first_nonempty(
            getattr(model, "rbf_bandwidth_mode", None),
            train_args.get("rbf_bandwidth_mode", None),
        )
        if bandwidth_mode is not None:
            normalized_mode = str(bandwidth_mode).strip().lower()
            if normalized_mode == "global":
                params.append("Global")
            elif normalized_mode == "knn":
                knn_k = _first_nonempty(
                    getattr(model, "rbf_knn_k", None),
                    train_args.get("rbf_knn_k", None),
                )
                params.append(f"KNN K{knn_k}" if knn_k is not None else "KNN")
            else:
                params.append(str(bandwidth_mode).replace("_", " ").title())

        centers = _first_nonempty(
            getattr(model, "rbf_n_centers", None),
            train_args.get("rbf_n_centers", None),
        )
        if centers is not None:
            params.append(f"N Centers {centers}")

    if expansion_type_norm in {"hankel", "hankel_svd"}:
        depth = _first_nonempty(
            getattr(model, "delay_depth", None),
            train_args.get("delay_depth", None),
        )
        rank = _first_nonempty(
            getattr(model, "hankel_rank", None),
            train_args.get("hankel_rank", None),
        )
        if depth is not None:
            params.append(f"Depth {depth}")
        if rank is not None:
            params.append(f"Rank {rank}")

    if model_name == "mlp_baseline":
        hidden_dim = _first_nonempty(
            getattr(model, "hidden_dim", None),
            train_args.get("hidden_dim", None),
        )
        num_layers = _first_nonempty(
            getattr(model, "num_layers", None),
            train_args.get("num_layers", None),
        )
        if hidden_dim is not None:
            params.append(f"Hidden Dim {hidden_dim}")
        if num_layers is not None:
            params.append(f"Num Layers {num_layers}")

    if model_name == "sindy_baseline":
        library_type = _first_nonempty(
            getattr(model, "library_type", None),
            train_args.get("sindy_library_type", None),
        )
        if library_type is not None:
            params.append(f"Library type {library_type}")

    if model_name in {"ml_dmd", "ml_dmd_free", "ml_dmd_band", "ml_dmd_drop"}: # <-- CHANGE TO INCLUDE ml_dmd_drop
        l1_weight = _first_nonempty(
            train_args.get("l1_weight", None),
            getattr(model, "l1_weight", None),
        )
        if l1_weight is not None:
            try:
                params.append(f"L1 Weight {float(l1_weight):.3g}")
            except (TypeError, ValueError):
                params.append(f"L1 Weight {l1_weight}")
                
        # Handle Biorth parameter plotting
        biorth_weight = _first_nonempty(
            train_args.get("biorth_weight", None),
            getattr(model, "biorth_weight", None),
        )
        if biorth_weight is not None:
            try:
                params.append(f"Biorth Weight {float(biorth_weight):.3g}")
            except (TypeError, ValueError):
                params.append(f"Biorth Weight {biorth_weight}")

    if model_name == "regression_dmd":
        rollout_mode = _first_nonempty(
            getattr(model, "rollout_mode", None),
            train_args.get("regression_rollout_mode", None),
            train_args.get("rollout_mode", None),
        )
        if rollout_mode is not None:
            params.append(f"Rollout mode {rollout_mode}")

    return params


def format_model_label(model_name: str, model, extras: Dict[str, Any], system: Optional[str] = None) -> str:
    pieces = [_pretty_model_name(model_name)]
    train_args = _get_train_args(extras)

    system_name = _first_nonempty(
        system,
        getattr(model, "system", None),
        train_args.get("system", None),
        train_args.get("system_name", None),
    )
    if system_name is not None:
        pieces.append(_pretty_system_name(str(system_name)))

    # --- FIX: Skip expansion formatting for MLP ---
    if model_name != "mlp_baseline":
        expansion_type = _first_nonempty(
            getattr(model, "expansion_type", None),
            train_args.get("expansion_type", None),
        )
        
        # --- FIX: Safe degree extraction (Use None as default so we don't shadow train_args) ---
        try:
            deg_val = _first_nonempty(
                getattr(model, "expansion_degree", None), 
                train_args.get("expansion_degree", None)
            )
            deg = int(deg_val) if deg_val is not None else 1
        except (TypeError, ValueError):
            deg = 1

        # Intercept General Expansion with Degree <= 1
        if str(expansion_type).lower() == "general" and deg <= 1:
            pieces.append("No Expansion")
            
            # Pass a dummy string so _format_expansion_parameters skips "deg 1"
            # but STILL successfully checks for "+ Trig" or "Delay" tags!
            pieces.extend(_format_expansion_parameters(model_name, model, train_args, "linear_override"))
        else:
            pretty_expansion_type = _pretty_expansion_type(expansion_type)
            if pretty_expansion_type is not None:
                pieces.append(pretty_expansion_type)
    
            pieces.extend(_format_expansion_parameters(model_name, model, train_args, expansion_type))
    else:
        pieces.extend(_format_expansion_parameters(model_name, model, train_args, None))

    return " | ".join(pieces)

# ============================================================
# True dynamics reconstruction
# ============================================================

def build_true_dynamics_from_dataset(data_path: str):
    data = np.load(data_path, allow_pickle=True)
    system = str(_np_scalar(data, "system"))

    if system in {"linear", "inward_spiral", "harmonic_oscillator", "saddle_point", "degenerate_node"}:
        return linear_system(np.asarray(data["A"], dtype=float))

    if system == "vanderpol":
        return vanderpol_system(mu=float(_np_scalar(data, "mu")))

    if system == "lotka_volterra":
        return lotka_volterra_system(
            alpha=float(_np_scalar(data, "alpha")),
            beta=float(_np_scalar(data, "beta")),
            delta=float(_np_scalar(data, "delta")),
            gamma=float(_np_scalar(data, "gamma")),
        )

    if system == "pendulum":
        return pendulum_system(
            g=float(_np_scalar(data, "g")),
            L=float(_np_scalar(data, "L")),
        )

    if system == "lorenz":
        return lorenz_system(
            sigma=float(_np_scalar(data, "sigma")),
            rho=float(_np_scalar(data, "rho")),
            beta=float(_np_scalar(data, "beta")),
        )

    if system == "duffing":
        return duffing_system(
            alpha=float(_np_scalar(data, "alpha")),
            beta=float(_np_scalar(data, "beta")),
            delta=float(_np_scalar(data, "delta")),
            gamma=float(_np_scalar(data, "gamma")),
            omega=float(_np_scalar(data, "omega")),
        )

    if system == "closed_small":
        return closed_small_system(
            mu=float(_np_scalar(data, "mu")),
            alpha=float(_np_scalar(data, "alpha")),
        )

    if system == "closed_large":
        return closed_large_system(
            mu=float(_np_scalar(data, "mu")),
            alpha=float(_np_scalar(data, "alpha")),
            beta=float(_np_scalar(data, "beta")),
            gamma=float(_np_scalar(data, "gamma")),
            delta=float(_np_scalar(data, "delta")),
        )

    if system == "closed_trig_small":
        return closed_trig_small_system(
            omega=float(_np_scalar(data, "omega")),
            alpha=float(_np_scalar(data, "alpha")),
            beta_s1=float(_np_scalar(data, "beta_s1")),
            beta_c1=float(_np_scalar(data, "beta_c1")),
            beta_x=float(_np_scalar(data, "beta_x")),
            beta_x2=float(_np_scalar(data, "beta_x2")),
        )

    if system == "closed_trig_medium":
        return closed_trig_medium_system(
            omega=float(_np_scalar(data, "omega")),
            alpha=float(_np_scalar(data, "alpha")),
            beta_s1=float(_np_scalar(data, "beta_s1")),
            beta_c1=float(_np_scalar(data, "beta_c1")),
            beta_s2=float(_np_scalar(data, "beta_s2")),
            beta_c2=float(_np_scalar(data, "beta_c2")),
            beta_x=float(_np_scalar(data, "beta_x")),
            beta_x2=float(_np_scalar(data, "beta_x2")),
        )

    if system == "closed_trig_large":
        return closed_trig_large_system(
            omega=float(_np_scalar(data, "omega")),
            alpha=float(_np_scalar(data, "alpha")),
            beta_s1=float(_np_scalar(data, "beta_s1")),
            beta_c1=float(_np_scalar(data, "beta_c1")),
            beta_s2=float(_np_scalar(data, "beta_s2")),
            beta_c2=float(_np_scalar(data, "beta_c2")),
            beta_s3=float(_np_scalar(data, "beta_s3")),
            beta_c3=float(_np_scalar(data, "beta_c3")),
            beta_x=float(_np_scalar(data, "beta_x")),
            beta_x2=float(_np_scalar(data, "beta_x2")),
        )

    raise ValueError(f"Unsupported system '{system}' for dense-grid diagnostics.")


# ============================================================
# Grid bounds
# ============================================================

def _auto_grid_bounds(X: np.ndarray, i: int, j: int, pad_frac: float = 0.08):
    xi = X[..., i].reshape(-1)
    xj = X[..., j].reshape(-1)

    x_lo, x_hi = np.percentile(xi, [1.0, 99.0])
    y_lo, y_hi = np.percentile(xj, [1.0, 99.0])

    dx = max(x_hi - x_lo, 1e-8)
    dy = max(y_hi - y_lo, 1e-8)

    return (
        (x_lo - pad_frac * dx, x_hi + pad_frac * dx),
        (y_lo - pad_frac * dy, y_hi + pad_frac * dy),
    )


def _default_grid_bounds_from_dataset(data, X: np.ndarray, i: int, j: int):
    system = str(_np_scalar(data, "system"))

    if system in {"linear", "inward_spiral", "harmonic_oscillator", "degenerate_node"}:
        return (-1.5, 1.5), (-1.5, 1.5)

    if system == "saddle_point":
        return (-0.55, 0.55), (-1.65, 1.65)

    if system == "vanderpol":
        return (-3.5, 3.5), (-3.5, 3.5)

    if system == "pendulum":
        return (-2.8, 2.8), (-3.5, 3.5)

    if system == "lotka_volterra":
        alpha = float(_np_scalar(data, "alpha"))
        beta = float(_np_scalar(data, "beta"))
        delta = float(_np_scalar(data, "delta"))
        gamma = float(_np_scalar(data, "gamma"))
        x_star = gamma / delta
        y_star = alpha / beta
        return (max(0.3, x_star - 3.0), x_star + 3.0), (max(0.3, y_star - 3.0), y_star + 3.0)

    if system == "duffing":
        alpha = float(_np_scalar(data, "alpha"))
        beta = float(_np_scalar(data, "beta"))
        if alpha < 0 and beta > 0:
            x_eq = np.sqrt(-alpha / beta)
            return (-(x_eq + 0.6), x_eq + 0.6), (-1.2, 1.2)

    if system == "closed_small":
        return (-1.0, 1.0), (-1.0, 1.5)

    if system == "closed_large":
        return (-1.0, 1.0), (-1.0, 1.0)

    if system in {"closed_trig_small", "closed_trig_medium", "closed_trig_large"}:
        return (-2.0, 2.0), (-1.0, 1.0)

    return _auto_grid_bounds(X, i, j)


# ============================================================
# Error scaling / colorbar formatting
# ============================================================

def _make_error_norm(errors: np.ndarray, force_linear: bool = False):
    finite_errors = np.asarray(errors, dtype=float)
    finite_errors = finite_errors[np.isfinite(finite_errors)]

    if finite_errors.size == 0:
        vmin, vmax = 1e-16, 1.0
        return mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True), vmin, vmax, False

    positive = finite_errors[finite_errors > 0]

    if positive.size == 0:
        vmin, vmax = 0.0, max(float(np.max(finite_errors)), 1.0)
        return mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True), vmin, vmax, False

    if force_linear:
        vmin = max(0.0, float(np.percentile(finite_errors, 1.0)))
        vmax = float(np.percentile(finite_errors, 99.5))
        if vmax <= vmin:
            vmax = max(float(np.max(finite_errors)), vmin + 1e-12)
        return mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True), vmin, vmax, False

    vmin = max(float(np.percentile(positive, 1.0)), 1e-16)
    vmax = float(np.percentile(finite_errors, 99.5))
    if vmax <= vmin:
        vmax = max(float(np.max(positive)), vmin * 10.0)

    ratio = vmax / vmin
    if ratio >= 50.0:
        return mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=True), vmin, vmax, True

    vmin_lin = max(0.0, float(np.percentile(finite_errors, 1.0)))
    vmax_lin = float(np.percentile(finite_errors, 99.5))
    if vmax_lin <= vmin_lin:
        vmax_lin = max(float(np.max(finite_errors)), vmin_lin + 1e-12)

    return mcolors.Normalize(vmin=vmin_lin, vmax=vmax_lin, clip=True), vmin_lin, vmax_lin, False


def _format_three_tick_colorbar(cbar, vmin: float, vmax: float, use_log: bool):
    tick_mid = np.sqrt(vmin * vmax) if use_log else 0.5 * (vmin + vmax)
    ticks = [vmin, tick_mid, vmax]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1e}" for t in ticks])
    cbar.minorticks_off()


# ============================================================
# Simple RMSE summary plots
# ============================================================

def plot_error_vs_horizon(
    horizon_metrics: Dict[str, np.ndarray],
    figdir: str,
    model_label: str,
    logy: bool = True,
) -> None:
    horizons = np.asarray(horizon_metrics["horizons"], dtype=int)
    rmse = np.asarray(horizon_metrics["horizon_rmse"], dtype=float)

    plt.figure(figsize=(10.8, 5.2))
    plt.plot(horizons, rmse, marker="o", linewidth=1.8, markersize=4)
    plt.xlabel("Prediction horizon")
    plt.ylabel("RMSE")
    plt.title(f"Error vs prediction horizon\n{_wrap_model_label(model_label, width=72)}")
    if logy:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=(0, 0, 1, 0.955))
    plt.savefig(os.path.join(figdir, "error_vs_horizon.png"), dpi=220)
    plt.close()


def plot_rollout_error_summary(
    rollout_metrics: Dict[str, np.ndarray],
    figdir: str,
    model_label: str,
) -> None:
    horizons = np.asarray(rollout_metrics["rollout_horizons"], dtype=int)
    rmse = np.asarray(rollout_metrics["rollout_rmse"], dtype=float)

    plt.figure(figsize=(10.8, 5.2))
    plt.plot(horizons, rmse, marker="o", linewidth=1.8, markersize=4)
    plt.xlabel("Rollout horizon")
    plt.ylabel("RMSE")
    plt.title(f"Full-rollout error summary\n{_wrap_model_label(model_label, width=72)}")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=(0, 0, 1, 0.955))
    plt.savefig(os.path.join(figdir, "rollout_error_distribution.png"), dpi=220)
    plt.close()


# ============================================================
# Dense-grid compute helpers
# ============================================================

def _terminal_rmse_with_bad_mask(
    pred_terminal: np.ndarray,
    true_terminal: np.ndarray,
    overflow_threshold: float = 1e150,
) -> Tuple[np.ndarray, int]:
    pred_terminal = np.asarray(pred_terminal, dtype=np.float64)
    true_terminal = np.asarray(true_terminal, dtype=np.float64)

    diff = pred_terminal - true_terminal
    invalid_mask = ~np.isfinite(diff).all(axis=1)
    large_mask = np.max(np.abs(diff), axis=1) > overflow_threshold
    bad_mask = invalid_mask | large_mask

    errors_flat = np.empty(diff.shape[0], dtype=np.float64)
    errors_flat.fill(np.inf)

    good_mask = ~bad_mask
    if np.any(good_mask):
        diff_good = diff[good_mask]
        errors_flat[good_mask] = np.sqrt(np.mean(diff_good * diff_good, axis=1))

    return errors_flat, int(np.sum(bad_mask))


def _predict_rollout_batch_safely(
    *,
    x0_batch: np.ndarray,
    steps: int,
    model_name: str,
    model,
    extras: Dict[str, Any],
    mode_indices: Optional[np.ndarray] = None,
    initial_chunk_size: int = 8192,
) -> np.ndarray:
    x0_batch = np.asarray(x0_batch)

    try:
        rollout = predict_rollout_from_x0(
            x0=x0_batch,
            steps=steps,
            model_name=model_name,
            model=model,
            extras=extras,
            mode_indices=mode_indices,
        )
        rollout = np.asarray(rollout)
        if rollout.ndim == 2:
            rollout = rollout[:, None, :]
        return rollout

    except Exception as e:
        print(f"[diagnostics] Full batched rollout failed ({type(e).__name__}: {e}). Falling back to chunking.")

    n = x0_batch.shape[0]
    chunks = []
    start = 0
    chunk_size = max(1, min(int(initial_chunk_size), n))

    while start < n:
        end = min(start + chunk_size, n)
        x0_chunk = x0_batch[start:end]

        try:
            rollout_chunk = predict_rollout_from_x0(
                x0=x0_chunk,
                steps=steps,
                model_name=model_name,
                model=model,
                extras=extras,
                mode_indices=mode_indices,
            )
            rollout_chunk = np.asarray(rollout_chunk)
            if rollout_chunk.ndim == 2:
                rollout_chunk = rollout_chunk[:, None, :]
            chunks.append(rollout_chunk)
            start = end

        except Exception as e:
            if chunk_size > 1:
                chunk_size = max(1, chunk_size // 2)
                print(f"[diagnostics] Retrying with chunk_size={chunk_size} after failure: {e}")
                continue

            rollout_single = predict_rollout_from_x0(
                x0=x0_batch[start],
                steps=steps,
                model_name=model_name,
                model=model,
                extras=extras,
                mode_indices=mode_indices,
            )
            rollout_single = np.asarray(rollout_single)
            if rollout_single.ndim != 2:
                raise RuntimeError(f"Unexpected single rollout shape: {rollout_single.shape}")
            chunks.append(rollout_single[:, None, :])
            start += 1

    return np.concatenate(chunks, axis=1)


def compute_true_grid_heatmap_grid(
    *,
    data_path: str,
    X: np.ndarray,
    horizons: List[int],
    heatmap_specs: List[Dict[str, object]],
    model_name: str,
    model,
    extras: Dict[str, Any],
    grid_resolution: int = 100,
) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
    """
    Fast dense-grid evaluation:
    - grid once
    - true rollout once to max_horizon
    - model rollout once to max_horizon per mode spec
    - slice requested horizons
    """
    horizons = sorted(set(int(h) for h in horizons))
    if len(horizons) == 0:
        raise ValueError("At least one horizon must be provided.")

    max_horizon = max(horizons)

    data = np.load(data_path, allow_pickle=True)
    dt = float(_np_scalar(data, "dt"))
    method = str(_np_scalar(data, "method", "rk4"))
    system = str(_np_scalar(data, "system"))

    state_dim = X.shape[-1]
    i, j = get_phase_dims(system, state_dim)
    xlim, ylim = _default_grid_bounds_from_dataset(data, X, i, j)

    xs = np.linspace(xlim[0], xlim[1], grid_resolution)
    ys = np.linspace(ylim[0], ylim[1], grid_resolution)
    XX, YY = np.meshgrid(xs, ys)

    fixed_state = X.reshape(-1, state_dim).mean(axis=0)

    grid_points = np.tile(fixed_state[None, :], (XX.size, 1))
    grid_points[:, i] = XX.ravel()
    grid_points[:, j] = YY.ravel()

    f_true = build_true_dynamics_from_dataset(data_path)

    _, X_true_grid = simulate(
        f_true,
        x0=grid_points,
        dt=dt,
        T=max_horizon * dt,
        method=method,
    )

    delay_depth = get_model_delay_depth(model_name, model)
    model_grid_x0 = make_backward_delay_x0_from_current_states(
        current_states=grid_points,
        f_true=f_true,
        dt=dt,
        delay_depth=delay_depth,
    )

    base_grid = {
        "XX": XX,
        "YY": YY,
        "dims": np.array([i, j], dtype=int),
        "fixed_state": fixed_state,
        "xlim": np.array(xlim, dtype=float),
        "ylim": np.array(ylim, dtype=float),
    }

    grid_results: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {h: {} for h in horizons}

    for spec in heatmap_specs:
        spec_name = str(spec["name"])
        mode_indices = spec["mode_indices"]

        print(f"[diagnostics] Predicting dense-grid rollout once for column '{spec_name}' up to h={max_horizon}...")

        rollout_batch = _predict_rollout_batch_safely(
            x0_batch=model_grid_x0,
            steps=max_horizon,
            model_name=model_name,
            model=model,
            extras=extras,
            mode_indices=mode_indices,
            initial_chunk_size=8192,
        )

        for h in horizons:
            pred_terminal = rollout_batch[h]
            true_terminal = X_true_grid[h]

            errors_flat, n_bad = _terminal_rmse_with_bad_mask(pred_terminal, true_terminal)
            if n_bad > 0:
                print(f"[diagnostics] Warning: {n_bad}/{errors_flat.size} bad grid points at h={h}, column='{spec_name}'.")

            grid_results[h][spec_name] = {
                **base_grid,
                "errors": errors_flat.reshape(XX.shape),
            }

    return grid_results


# ============================================================
# Dense-grid plotting
# ============================================================

def select_overlay_trajectories(
    *,
    X: np.ndarray,
    split_idx: np.ndarray,
    traj_id: int,
    n_trajs: int,
) -> List[np.ndarray]:
    if n_trajs <= 0:
        return []

    split_idx = list(split_idx)
    if traj_id not in split_idx:
        split_idx = [traj_id] + split_idx

    selected = [traj_id]
    if n_trajs > 1:
        remaining = [tid for tid in split_idx if tid != traj_id]
        k = min(n_trajs - 1, len(remaining))
        if k > 0:
            positions = np.linspace(0, len(remaining) - 1, k, dtype=int)
            selected.extend([remaining[p] for p in positions])

    return [X[:, tid, :] for tid in selected]


def _build_reference_trajectory_color_info(
    trajectory: np.ndarray,
    dims: Tuple[int, int],
):
    i, j = dims
    pts = np.asarray(trajectory[:, [i, j]], dtype=float)

    if pts.shape[0] < 2:
        return None

    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    step_disp = np.linalg.norm(np.diff(pts, axis=0), axis=1)

    finite = step_disp[np.isfinite(step_disp)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
        values = np.zeros_like(step_disp)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        positive = finite[finite > 0]
        if positive.size > 0:
            vmin = np.percentile(positive, 5.0)
            vmax = np.percentile(positive, 95.0)
            if vmax <= vmin:
                vmax = max(positive.max(), vmin + 1e-12)
        else:
            vmin = 0.0
            vmax = max(finite.max(), 1e-12)

        values = np.clip(step_disp, vmin, vmax)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    return {
        "points": pts,
        "segments": segments,
        "values": values,
        "norm": norm,
        "vmin": vmin,
        "vmax": vmax,
    }

def _wrap_model_label(model_label: str, width: int = 65) -> str:
    """
    Wrap long model labels onto multiple lines so the suptitle does not overflow.
    """
    if model_label.startswith("MLP"):
        return model_label

    return "\n".join(textwrap.wrap(model_label, width=width, break_long_words=False))


def plot_true_grid_heatmap_grid(
    *,
    grid_results: Dict[int, Dict[str, Dict[str, np.ndarray]]],
    horizons: List[int],
    heatmap_specs: List[Dict[str, object]],
    system: str,
    model_label: str,
    figdir: str,
    trajectory_overlay: Optional[np.ndarray] = None,
    trajectory_overlays: Optional[List[np.ndarray]] = None,
    filename: str = "true_grid_error_heatmap_grid.png",
    force_linear_error_scale: bool = False,
    data_path: Optional[str] = None,
    wspace: float = 0.24,
    title_fontsize: int = 17,
    subtitle_fontsize: int = 13,
    cbar_label_fontsize: int = 12,
    top_margin: float = 0.88,
    colorbar_pad_cols: tuple[float, ...] = (0.03, 0.07, 0.03),
    colorbar_axis_widths: tuple[float, float] = (0.05, 0.05),
    legend_y: float = 0.91,
) -> None:
    n_rows = len(horizons)
    n_cols = len(heatmap_specs)

    # --------------------------------------------------
    # Collect all finite errors for shared color scaling
    # --------------------------------------------------
    all_errors = []
    for h in horizons:
        for spec in heatmap_specs:
            errs = np.asarray(grid_results[h][str(spec["name"])]["errors"], dtype=float)
            finite = errs[np.isfinite(errs)]
            if finite.size > 0:
                all_errors.append(finite)

    concat_errors = np.concatenate(all_errors) if len(all_errors) > 0 else np.array([1e-16])
    norm, vmin, vmax, use_log = _make_error_norm(
        concat_errors,
        force_linear=force_linear_error_scale,
    )

    # --------------------------------------------------
    # Figure sizing:
    # make it scale sensibly with rows/cols
    # --------------------------------------------------
    panel_w = 4.8
    panel_h = 3.9
    fig_width = max(10.5, panel_w * n_cols + 4.0)
    fig_height = max(5.8, panel_h * n_rows + 2.3)

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Dedicated columns:
    # - heatmap columns
    # - error colorbar
    # - trajectory colorbar
    effective_wspace = max(wspace, 0.40 if n_cols >= 3 else 0.24)

    if len(colorbar_pad_cols) == 2:
        cbar_pad_left = float(colorbar_pad_cols[0])
        cbar_pad_middle = float(colorbar_pad_cols[1])
        cbar_pad_right = cbar_pad_left
    else:
        cbar_pad_left = float(colorbar_pad_cols[0])
        cbar_pad_middle = float(colorbar_pad_cols[1])
        cbar_pad_right = float(colorbar_pad_cols[2])

    gs = GridSpec(
        nrows=n_rows,
        ncols=n_cols + 5,
        figure=fig,
        # heatmaps | spacer | error cbar | spacer | trajectory cbar | right spacer
        width_ratios=[1.0] * n_cols + [cbar_pad_left, colorbar_axis_widths[0], cbar_pad_middle, colorbar_axis_widths[1], cbar_pad_right],
        wspace=effective_wspace,
        hspace=0.25,
    )

    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c])

    # Dedicated colorbar axes, with spacer columns around them
    cax_err = fig.add_subplot(gs[:, n_cols + 1])
    cax_traj = fig.add_subplot(gs[:, n_cols + 3])

    first_spec_name = str(heatmap_specs[0]["name"])
    first_grid = grid_results[horizons[0]][first_spec_name]
    dims = tuple(first_grid["dims"].tolist())

    traj_cmap = mcolors.LinearSegmentedColormap.from_list(
        "traj_overlay_pink",
        ["#f8d4ff", "#f39cf6", "#ec5be8", "#d81b9c", "#a0006d"],
        N=256,
    )

    mesh_for_cbar = None
    traj_line_for_cbar = None

    # FIXED: Handle a list of multiple real trajectories
    traj_color_infos = []
    if trajectory_overlay is not None:
        if isinstance(trajectory_overlay, list):
            for t in trajectory_overlay:
                traj_color_infos.append(_build_reference_trajectory_color_info(t, dims))
        else:
            traj_color_infos.append(_build_reference_trajectory_color_info(trajectory_overlay, dims))

    mesh_for_cbar = None
    traj_line_for_cbar = None

    # --------------------------------------------------
    # Plot panels
    # --------------------------------------------------
    for row, h in enumerate(horizons):
        dx_grid, dy_grid = None, None

        if data_path is not None:
            try:
                f_true = build_true_dynamics_from_dataset(data_path)
                g_data = grid_results[h][first_spec_name]
                XX_row, YY_row = g_data["XX"], g_data["YY"]
                i_idx, j_idx = g_data["dims"]

                grid_pts = np.tile(g_data["fixed_state"][None, :], (XX_row.size, 1))
                grid_pts[:, i_idx] = XX_row.ravel()
                grid_pts[:, j_idx] = YY_row.ravel()

                vf = f_true(0.0, grid_pts)
                dx_grid = vf[:, i_idx].reshape(XX_row.shape)
                dy_grid = vf[:, j_idx].reshape(XX_row.shape)
            except Exception:
                pass

        for col, spec in enumerate(heatmap_specs):
            ax = axes[row, col]
            spec_name = str(spec["name"])
            spec_title = str(spec["title"])

            grid_data = grid_results[h][spec_name]
            XX, YY = grid_data["XX"], grid_data["YY"]
            errors = np.asarray(grid_data["errors"], dtype=float)
            i, j = grid_data["dims"]

            plot_errors = np.where(np.isfinite(errors), errors, np.nan)
            if use_log:
                plot_errors = np.where(
                    np.isfinite(plot_errors),
                    np.maximum(plot_errors, vmin),
                    np.nan,
                )
            else:
                plot_errors = np.where(
                    np.isfinite(plot_errors),
                    np.clip(plot_errors, vmin, vmax),
                    np.nan,
                )

            mesh = ax.pcolormesh(
                XX, YY, plot_errors,
                shading="auto",
                cmap="viridis",
                norm=norm,
            )
            if mesh_for_cbar is None:
                mesh_for_cbar = mesh

            if trajectory_overlays is not None:
                for traj in trajectory_overlays:
                    ax.plot(
                        traj[:, i], traj[:, j],
                        linewidth=0.7,
                        alpha=0.15,
                        color="white",
                        zorder=2,
                    )

            # FIXED: Loop through and plot all 20 colored trajectories
            if traj_color_infos:
                for t_info in traj_color_infos:
                    ax.plot(
                        t_info["points"][:, 0],
                        t_info["points"][:, 1],
                        color="white",
                        linewidth=2.0,
                        alpha=0.10,
                        zorder=3,
                    )
                    lc = LineCollection(
                        t_info["segments"],
                        cmap=traj_cmap,
                        norm=t_info["norm"],
                        linewidth=1.7,
                        alpha=0.95,
                        zorder=4,
                    )
                    lc.set_array(t_info["values"])
                    ax.add_collection(lc)
                    if traj_line_for_cbar is None:
                        traj_line_for_cbar = lc

            if dx_grid is not None and dy_grid is not None:
                ax.contour(
                    XX, YY, dx_grid,
                    levels=[0],
                    colors="red",
                    linewidths=1.5,
                    linestyles="--",
                    alpha=0.85,
                    zorder=5,
                )
                ax.contour(
                    XX, YY, dy_grid,
                    levels=[0],
                    colors="orange",
                    linewidths=1.5,
                    linestyles="--",
                    alpha=0.85,
                    zorder=5,
                )

            ax.set_xlim(grid_data["xlim"])
            ax.set_ylim(grid_data["ylim"])

            if row == 0:
                ax.set_title(spec_title, fontsize=subtitle_fontsize)

            if col == 0:
                ax.set_ylabel(f"pred_horizon = {h}\n\nx{i + 1}", fontsize=12)
            else:
                ax.set_ylabel("")

            if row == n_rows - 1:
                ax.set_xlabel(f"x{j + 1}", fontsize=12)
            else:
                ax.set_xlabel("")

    # --------------------------------------------------
    # Suptitle with wrapped model label
    # --------------------------------------------------
    wrapped_label = _wrap_model_label(model_label, width=70)
    fig.suptitle(f"True-grid error heatmaps\n{wrapped_label}", fontsize=title_fontsize, y=0.965)

    # --------------------------------------------------
    # Error colorbar
    # --------------------------------------------------
    cbar_err = fig.colorbar(mesh_for_cbar, cax=cax_err)
    cbar_err.set_label("Terminal h-step RMSE", fontsize=cbar_label_fontsize, labelpad=12)
    cbar_err.ax.yaxis.set_label_position("right")
    cbar_err.ax.yaxis.tick_right()
    cbar_err.ax.tick_params(labelleft=False, labelright=True, left=False, right=True)
    _format_three_tick_colorbar(cbar_err, vmin, vmax, use_log)

    # TIGHTEN PADDING: Reduce padding on the colorbar axes to bring them left
    cax_err.tick_params(pad=1, axis='y')
    cax_traj.tick_params(pad=1, axis='y')

    # --------------------------------------------------
    # Trajectory colorbar
    # --------------------------------------------------
    # FIXED: Check the new list length, and grab vmin/vmax from the first trajectory
    if traj_line_for_cbar is not None and len(traj_color_infos) > 0:
        cbar_traj = fig.colorbar(traj_line_for_cbar, cax=cax_traj)
        cbar_traj.set_label(
            "Reference trajectory\nper-step displacement",
            fontsize=12,
            labelpad=12,
        )
        cbar_traj.ax.yaxis.set_label_position("right")
        cbar_traj.ax.yaxis.tick_right()
        cbar_traj.ax.tick_params(labelleft=False, labelright=True, left=False, right=True)
        _format_three_tick_colorbar(
            cbar_traj,
            traj_color_infos[0]["vmin"],
            traj_color_infos[0]["vmax"],
            False,
        )
    else:
        # hide unused axis cleanly
        cax_traj.axis("off")

    # --------------------------------------------------
    # Final layout:
    # use explicit margins instead of tight_layout to avoid
    # warnings with the custom GridSpec + colorbar axes.
    # --------------------------------------------------

    # ADDED: Global legend placed neatly in the middle above the plots
    if dx_grid is not None and dy_grid is not None:
        legend_elements = [
            Line2D([0], [0], color="red", lw=1.5, linestyle="--", label=r"$\dot{x}_1 = 0$"),
            Line2D([0], [0], color="orange", lw=1.5, linestyle="--", label=r"$\dot{x}_2 = 0$"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=2,
            framealpha=0.95,
        )

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.07, top=top_margin)

    out_path = os.path.join(figdir, filename)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Minimal diagnostics runner
# ============================================================

def run_diagnostics(
    *,
    X: np.ndarray,
    split_idx: np.ndarray,
    traj_id: int,
    model_name: str,
    model,
    extras: Dict[str, Any],
    system: str,
    figdir: str,
    horizon_metrics: Dict[str, np.ndarray],
    rollout_metrics: Dict[str, np.ndarray],
    data_path: str,
    run_true_grid_heatmap: bool = False,
    grid_resolution: int = 100,
    true_grid_heatmap_horizons: Optional[List[int]] = None,
    overlay_true_trajectory_on_grid: bool = True,
    grid_overlay_n_trajs: int = 1,
    force_linear_true_grid_error_scale: bool = False,
    mode_subset_thresholds: Optional[List[float]] = None,
    mode_subset_indices: Optional[List[int]] = None,
    linear_error_scale: bool = False,
    **kwargs,
) -> None:
    os.makedirs(figdir, exist_ok=True)

    model_label = format_model_label(model_name, model, extras, system=system)

    plot_error_vs_horizon(
        horizon_metrics=horizon_metrics,
        figdir=figdir,
        model_label=model_label,
        logy=not linear_error_scale,
    )

    plot_rollout_error_summary(
        rollout_metrics=rollout_metrics,
        figdir=figdir,
        model_label=model_label,
    )

    if not run_true_grid_heatmap:
        return

    supports_subset_rollout = supports_mode_subset_rollout(model_name, model, extras)

    horizons_to_plot = [1] if true_grid_heatmap_horizons is None else list(true_grid_heatmap_horizons)

    heatmap_specs = _build_mode_subset_heatmap_specs(
        model_name=model_name,
        model=model,
        extras=extras,
        mode_subset_thresholds=mode_subset_thresholds,
        mode_subset_indices=mode_subset_indices,
        X_states=X,
    )

    # --------------------------------------------------
    # First: always produce a "full model" only figure
    # --------------------------------------------------
    full_specs = [s for s in heatmap_specs if str(s.get("name")) == "all"]
    if not full_specs and len(heatmap_specs) > 0:
        # fallback: use the last spec as full
        full_specs = [heatmap_specs[-1]]

    grid_results_full = compute_true_grid_heatmap_grid(
        data_path=data_path,
        X=X,
        horizons=horizons_to_plot,
        heatmap_specs=full_specs,
        model_name=model_name,
        model=model,
        extras=extras,
        grid_resolution=grid_resolution,
    )
    
    # Safely extract the base directory if data_path is already a .npz file
    if data_path.endswith('.npz'):
        base_dir = os.path.dirname(data_path)
        normal_path = data_path
    else:
        base_dir = data_path
        normal_path = os.path.join(data_path, "test.npz")
    
    # Check potential paths for the long dataset
    long_path_t10 = os.path.join(base_dir + "_T10", "test.npz")
    long_path_sub = os.path.join(base_dir, "long", "test.npz")
    
    if os.path.exists(long_path_t10):
        raw_data = np.load(long_path_t10, allow_pickle=True)
    elif os.path.exists(long_path_sub):
        raw_data = np.load(long_path_sub, allow_pickle=True)
    else:
        raw_data = np.load(normal_path, allow_pickle=True)

    X_raw = raw_data["X"]
    if X_raw.ndim == 2:
        X_raw = X_raw[:, None, :]
    n_real = min(10, X_raw.shape[1])
    X_trajs = [X_raw[:, i, :] for i in range(n_real)]

    overlay_trajs = (
        select_overlay_trajectories(
            X=X,
            split_idx=split_idx,
            traj_id=traj_id,
            n_trajs=grid_overlay_n_trajs,
        )
        if overlay_true_trajectory_on_grid and grid_overlay_n_trajs > 1
        else None
    )

    # plot: full-model only
    plot_true_grid_heatmap_grid(
        grid_results=grid_results_full,
        horizons=horizons_to_plot,
        heatmap_specs=full_specs,
        system=system,
        model_label=model_label,
        figdir=figdir,
        trajectory_overlay=X_trajs if overlay_true_trajectory_on_grid else None, # pass X_trajs here
        trajectory_overlays=overlay_trajs,
        force_linear_error_scale=force_linear_true_grid_error_scale,
        data_path=data_path,
        filename="true_grid_error_heatmap_grid_full.png",
        legend_y=0.925,
    )

    # If there are additional mode-subset specs, produce a second combined figure.
    # Exclude the full-model column here because it was already rendered above.
    subset_specs = [spec for spec in heatmap_specs if str(spec.get("name")) != "all"]
    if subset_specs and not supports_subset_rollout:
        rollout_mode = extras.get("rollout_mode", "DMD")
        print(
            f"[diagnostics] Skipping mode-subset true-grid heatmaps for rollout mode '{rollout_mode}' "
            f"because mode_indices are not supported."
        )
        subset_specs = []

    if subset_specs:
        # --- FIX: Use model object directly ---
        for spec in subset_specs:
            if "mode_indices" in spec:
                spec["mode_indices"] = _get_expanded_indices(spec["mode_indices"], model)
        # ------------------------------------
        
        grid_results_all = compute_true_grid_heatmap_grid(
            data_path=data_path,
            X=X,
            horizons=horizons_to_plot,
            heatmap_specs=subset_specs,
            model_name=model_name,
            model=model,
            extras=extras,
            grid_resolution=grid_resolution,
        )

        plot_true_grid_heatmap_grid(
            grid_results=grid_results_all,
            horizons=horizons_to_plot,
            heatmap_specs=subset_specs,
            system=system,
            model_label=model_label,
            figdir=figdir,
            trajectory_overlay=X_trajs if overlay_true_trajectory_on_grid else None,
            trajectory_overlays=overlay_trajs,
            force_linear_error_scale=force_linear_true_grid_error_scale,
            data_path=data_path,
            filename="true_grid_error_heatmap_grid_with_subsets.png",
            wspace=0.16,
            colorbar_pad_cols=(0.001, 0.006, 0.001),
            colorbar_axis_widths=(0.08, 0.08),
            subtitle_fontsize=11,
            legend_y=0.928,
        )
