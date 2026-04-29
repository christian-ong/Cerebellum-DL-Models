import os
from typing import Dict, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.linear_baseline import rollout_linear_map
from src.models.dmd_baseline import rollout_dmd_eig
# from src.models.deprecated.ml_dmd import ML_DMD
# from src.models.deprecated.ml_eigen_dmd import MLEigenDMD
from src.models.regression_dmd import Regression_DMD
from src.models.ml_linear_dynamics import ML_LinearDynamics
from src.models.ml_dmd import ML_DMD
from src.models.sindy_baseline import SINDyBaseline
from src.data_generation.load_data import OneStepTrajectoryDataset, resolve_split_npz_path


def infer_system_name_from_data_path(data_path: str) -> str:
    base = os.path.basename(data_path)
    return base.replace("_trajectory.npz", "")


def infer_run_name(model_path: str, explicit_name: str = None) -> str:
    if explicit_name:
        return explicit_name
    return os.path.basename(os.path.dirname(model_path))


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    if isinstance(value, str):
        return value.lower() == "true"

    arr = np.asarray(value)
    if arr.shape == ():
        return bool(arr.item())

    return bool(value)


def _to_optional_str(value: Any):
    if value is None:
        return None

    arr = np.asarray(value)
    if arr.shape == ():
        s = str(arr.item())
    else:
        s = str(value)

    return None if s == "" else s

def _load_sindy_from_saved_config(model_path: str, data_path: str, state_dim: int) -> SINDyBaseline:
    model_data = np.load(model_path, allow_pickle=True)

    specific_system = _to_optional_str(model_data["specific_system"]) if "specific_system" in model_data else None

    specific_basis_size = None
    if "specific_basis_size" in model_data:
        specific_basis_size = int(np.asarray(model_data["specific_basis_size"]).item())
        if specific_basis_size < 0:
            specific_basis_size = None

    model = SINDyBaseline(
        discrete_time=_to_bool(model_data["discrete_time"], default=True),
        poly_order=int(np.asarray(model_data["poly_order"]).item()),
        include_bias=_to_bool(model_data["include_bias"], default=True),
        include_interaction=_to_bool(model_data["include_interaction"], default=True),
        threshold=float(np.asarray(model_data["threshold"]).item()),
        alpha=float(np.asarray(model_data["alpha"]).item()),
        differentiation_method=str(np.asarray(model_data["diff_method"]).item()),
        library_type=str(np.asarray(model_data["library_type"]).item()),
        fourier_n_frequencies=int(np.asarray(model_data["fourier_n_frequencies"]).item()),
        specific_system=specific_system,
        specific_basis_size=specific_basis_size,
    )

    # -------------------------------------------------------------
    # THE FIX: Load frozen coefficients directly. DO NOT REFIT!
    # -------------------------------------------------------------
    if "coefficients" in model_data:
        model.load_model(model_data["coefficients"], state_dim)
        
        # Fetch the dt for continuous models directly from the dataset
        if not model.discrete_time:
            train_split_path = resolve_split_npz_path(data_path, "train")
            meta = np.load(train_split_path)
            model.dt = float(meta["dt"])
    else:
        raise ValueError("No coefficients found in SINDy checkpoint! Please retrain the model.")

    return model


def load_model(
    *,
    model_name: str,
    model_path: str,
    data_path: str,
    state_dim: int,
    system: str,
    device: str,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Returns
    -------
    model : object or None
        Torch model instance for learned models, or None for pure numpy baselines.
    extras : dict
        Additional matrices / objects needed for rollout and evaluation.
    """
    model = None
    extras: Dict[str, Any] = {}

    if model_name == "linear_baseline":
        model_data = np.load(model_path)
        extras["M"] = model_data["M"]
        return model, extras

    if model_name == "dmd_baseline":
        model_data = np.load(model_path)
        extras["Lambda"] = model_data["Lambda"]
        extras["Phi"] = model_data["Phi"]
        return model, extras

    if model_name == "regression_dmd":
        model_data = np.load(model_path, allow_pickle=True)

        rollout_mode = str(model_data["rollout_mode"]) if "rollout_mode" in model_data else "DMD"
        extras["rollout_mode"] = rollout_mode

        rank_val = int(np.asarray(model_data["rank"]).item()) if "rank" in model_data else -1
        rank = None if rank_val < 0 else rank_val

        model = Regression_DMD(
            state_dim=state_dim,
            expansion_degree=int(model_data["expansion_degree"]),
            bias=_to_bool(model_data["bias"], default=True),
            sine_cosine_expansion=_to_bool(model_data["sine_cosine_expansion"], default=False),
            expansion_type=str(model_data["expansion_type"]),
            system=_to_optional_str(model_data["system_basis"]),
            normalize_state=_to_bool(model_data["normalize_state"], default=False),
            normalize_lifted=_to_bool(model_data["normalize_lifted"], default=True),
            rollout_mode=rollout_mode,
            ridge=float(np.asarray(model_data["ridge"]).item()) if "ridge" in model_data else 0.0,
            rank=rank,
        ).to(device)

        model.x_mean = torch.tensor(model_data["x_mean"], dtype=torch.float64)
        model.x_scale = torch.tensor(model_data["x_scale"], dtype=torch.float64)
        model.psi_scale = torch.tensor(model_data["psi_scale"], dtype=torch.float64)

        model.K_fitted = torch.tensor(model_data["K"], dtype=torch.float64)
        model.C_fitted = torch.tensor(model_data["C"], dtype=torch.float64)

        if "K_tilde" in model_data:
            model.K_tilde_fitted = torch.tensor(model_data["K_tilde"], dtype=torch.float64)
        if "U_r" in model_data:
            model.U_r_fitted = torch.tensor(model_data["U_r"], dtype=torch.float64)
        if "W_reduced" in model_data:
            model.W_reduced_fitted = torch.tensor(model_data["W_reduced"], dtype=torch.complex128)
        if "Lambda" in model_data:
            model.Lambda_fitted = torch.tensor(model_data["Lambda"], dtype=torch.complex128)
        if "Phi_lift" in model_data:
            model.Phi_lift_fitted = torch.tensor(model_data["Phi_lift"], dtype=torch.complex128)
            model.Phi_fitted = model.Phi_lift_fitted
            model.Phi_pinv_fitted = torch.linalg.pinv(model.Phi_lift_fitted)
        if "Phi_state" in model_data:
            model.Phi_state_fitted = torch.tensor(model_data["Phi_state"], dtype=torch.complex128)

        extras["K"] = model_data["K"]
        if "Lambda" in model_data:
            extras["Lambda"] = model_data["Lambda"]

        model.eval()
        return model, extras

    if model_name == "ml_lineardynamics":
        ckpt = torch.load(model_path, map_location=device)
        train_args = ckpt["train_args"]

        model = ML_LinearDynamics(
            state_dim=ckpt["state_dim"],
            expansion_degree=train_args["expansion_degree"],
            expansion_type=train_args["expansion_type"],
            bias=_to_bool(train_args.get("bias", "true"), default=True),
            sine_cosine_expansion=_to_bool(train_args.get("sine_cosine_expansion", "false"), default=False),
            system=ckpt["system"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        extras["ckpt"] = ckpt
        return model, extras

    if model_name == "ml_dmd":
        ckpt = torch.load(model_path, map_location=device)
        train_args = ckpt["train_args"]

        model = ML_DMD(
            state_dim=ckpt["state_dim"],
            expansion_degree=train_args["expansion_degree"],
            bias=_to_bool(train_args.get("bias", "true"), default=True),
            sine_cosine_expansion=_to_bool(train_args.get("sine_cosine_expansion", "false"), default=False),
            expansion_type=train_args["expansion_type"],
            system=ckpt["system"] if train_args["expansion_type"] == "specific" else None,
        ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        extras["ckpt"] = ckpt
        return model, extras

    if model_name == "sindy_baseline":
        model = _load_sindy_from_saved_config(model_path=model_path, data_path=data_path, state_dim=state_dim)
        return model, extras

    raise ValueError(f"Unknown model: {model_name}")


def predict_rollout_from_x0(*, x0, steps, model_name, model, extras, mode_indices=None):
    if model_name == "linear_baseline":
        return rollout_linear_map(extras["M"], x0=x0, steps=steps)

    if model_name == "dmd_baseline":
        return rollout_dmd_eig(extras["Lambda"], extras["Phi"], x0=x0, steps=steps)

    if model_name == "regression_dmd":
            rollout_mode = extras.get("rollout_mode", "DMD")
            rollout = model.rollout(
                x0=x0,
                steps=steps,
                mode=rollout_mode,
                mode_indices=mode_indices,
            )
            # CONVERSION FIX: Ensure we return NumPy to satisfy metrics.py
            if torch.is_tensor(rollout):
                return rollout.detach().cpu().numpy()
            return rollout

    if model_name == "sindy_baseline":
        return model.rollout(x0, steps=steps)

    with torch.inference_mode():
        return model.rollout(x0=x0, steps=steps).detach().cpu().numpy()
    
def supports_mode_subset_rollout(model_name: str, model, extras: Dict[str, Any]) -> bool:
    if model_name == "regression_dmd":
        rollout_mode = extras.get("rollout_mode", "DMD")
        return rollout_mode in {"DMD", "projected_DMD"}
    return False