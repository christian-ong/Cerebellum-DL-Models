import os
from typing import Dict, Any, Optional

import numpy as np
import torch


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_eig(M: Optional[np.ndarray]):
    if M is None:
        return None, None
    try:
        eigvals, eigvecs = np.linalg.eig(M)
        return eigvals, eigvecs
    except Exception:
        return None, None


def extract_transition_matrix(model_name: str, model, extras: Dict[str, Any]):
    if model_name == "linear_baseline":
        return _to_numpy(extras.get("M"))

    if model_name == "dmd_baseline":
        Phi = _to_numpy(extras.get("Phi"))
        Lambda = _to_numpy(extras.get("Lambda"))
        if Phi is None or Lambda is None:
            return None
        try:
            return Phi @ np.diag(Lambda) @ np.linalg.pinv(Phi)
        except Exception:
            return None

    if model_name == "regression_dmd":
        if hasattr(model, "K_fitted") and model.K_fitted is not None:
            return _to_numpy(model.K_fitted)
        return _to_numpy(extras.get("K"))

    if model is not None and hasattr(model, "get_K_true"):
        try:
            return _to_numpy(model.get_K_true())
        except Exception:
            return None

    return None


def extract_eigendecomposition(model_name: str, model, extras: Dict[str, Any]):
    if model_name == "dmd_baseline":
        eigvals = _to_numpy(extras.get("Lambda"))
        eigvecs = _to_numpy(extras.get("Phi"))
        return eigvals, eigvecs

    if model_name == "regression_dmd":
        if hasattr(model, "Lambda_fitted") and model.Lambda_fitted is not None:
            eigvals = _to_numpy(model.Lambda_fitted)
            eigvecs = _to_numpy(getattr(model, "Phi_lift_fitted", None))
            return eigvals, eigvecs

    if model is not None and hasattr(model, "get_Lambda"):
        try:
            eigvals = _to_numpy(model.get_Lambda())
            eigvecs = None
            if hasattr(model, "get_Phi_true"):
                try:
                    eigvecs = _to_numpy(model.get_Phi_true())
                except Exception:
                    pass
            return eigvals, eigvecs
        except Exception:
            pass

    K = extract_transition_matrix(model_name, model, extras)
    return _safe_eig(K)


def eigenvalue_summary(eigvals: Optional[np.ndarray], dt: Optional[float] = None):
    if eigvals is None or len(eigvals) == 0:
        return {}

    absvals = np.abs(eigvals)
    angles = np.angle(eigvals)

    summary = {
        "eigvals": eigvals,
        "spectral_radius": np.array(float(np.max(absvals))),
        "spectral_radius_min": np.array(float(np.min(absvals))),
        "n_unstable_outside_unit_circle": np.array(int(np.sum(absvals > 1.0 + 1e-12))),
        "n_near_unit_circle": np.array(int(np.sum(np.isclose(absvals, 1.0, atol=1e-3)))),
        "angles": angles,
    }

    if dt is not None and dt > 0:
        summary["angular_frequency_rad_per_step"] = angles
        summary["frequency_cycles_per_step"] = angles / (2 * np.pi)
        summary["frequency_hz"] = angles / (2 * np.pi * dt)

        with np.errstate(divide="ignore", invalid="ignore"):
            cont_eigs = np.log(eigvals) / dt
        summary["continuous_eigs"] = cont_eigs
        summary["continuous_growth_real"] = np.real(cont_eigs)
        summary["continuous_angular_frequency"] = np.imag(cont_eigs)

    return summary


def save_spectral_summary_npz(
    out_path: str,
    *,
    matrix: Optional[np.ndarray],
    eigvals: Optional[np.ndarray],
    eigvecs: Optional[np.ndarray],
    extra_summary: Optional[Dict[str, np.ndarray]] = None,
):
    save_kwargs = {}
    if matrix is not None:
        save_kwargs["transition_matrix"] = matrix
    if eigvals is not None:
        save_kwargs["eigvals"] = eigvals
    if eigvecs is not None:
        save_kwargs["eigvecs"] = eigvecs
    if extra_summary:
        save_kwargs.update(extra_summary)

    np.savez(out_path, **save_kwargs)


def maybe_extract_dt(data) -> Optional[float]:
    if "dt" not in data:
        return None
    try:
        return float(np.asarray(data["dt"]).item())
    except Exception:
        return None