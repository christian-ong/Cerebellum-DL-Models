import os
from typing import Dict, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.linear_baseline import rollout_linear_map
# from src.models.deprecated.dmd_baseline import rollout_dmd_eig
# from src.models.deprecated.ml_dmd import ML_DMD
# from src.models.deprecated.ml_eigen_dmd import MLEigenDMD
from src.models.regression_dmd import Regression_DMD
from src.models.ml_linear_dynamics import ML_LinearDynamics
from src.models.ml_dmd import ML_DMD
from src.models.sindy_baseline import SINDyBaseline
from src.data_generation.load_data import OneStepTrajectoryDataset


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


def _fit_sindy_from_saved_config(model_path: str, data_path: str) -> SINDyBaseline:
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

    if model.discrete_time:
        train_ds = OneStepTrajectoryDataset(data_path, split="train")
        train_loader = DataLoader(train_ds, batch_size=4096, shuffle=False)

        x_list, y_list = [], []
        for x, y in train_loader:
            x_list.append(x.numpy())
            y_list.append(y.numpy())

        x_train = np.vstack(x_list)
        y_train = np.vstack(y_list)
        model.fit_discrete_pairs(x_train, y_train)
    else:
        meta = np.load(data_path)
        x_all = meta["X"]
        train_idx = meta["train_idx"]
        dt = float(meta["dt"])
        x_train = x_all[:, train_idx, :]
        model.fit_continuous_trajectories(x_train, dt=dt)

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
        extras["K"] = model_data["K"]

        if "C" not in model_data:
            raise ValueError(
                "Checkpoint is missing decoder matrix C. "
                "Please retrain regression_dmd with the updated EDMD-style implementation."
            )
        extras["C"] = model_data["C"]

        degree = int(model_data["expansion_degree"]) if "expansion_degree" in model_data else 3

        # Backward-compatible handling:
        # prefer current training-side naming "bias"
        if "bias" in model_data:
            bias = _to_bool(model_data["bias"], default=True)
        elif "include_bias" in model_data:
            bias = _to_bool(model_data["include_bias"], default=True)
        elif "constant_expansion" in model_data:
            bias = _to_bool(model_data["constant_expansion"], default=True)
        else:
            bias = True

        if "sine_cosine_expansion" in model_data:
            sine_cosine_expansion = _to_bool(model_data["sine_cosine_expansion"], default=False)
        else:
            sine_cosine_expansion = False

        expansion_type = str(model_data["expansion_type"]) if "expansion_type" in model_data else "general"

        if "system_basis" in model_data:
            system_basis = _to_optional_str(model_data["system_basis"])
        else:
            system_basis = system if expansion_type == "specific" else None

        model = Regression_DMD(
            state_dim=state_dim,
            expansion_degree=degree,
            bias=bias,
            sine_cosine_expansion=sine_cosine_expansion,
            expansion_type=expansion_type,
            system=system_basis,
        ).to(device)
        model.eval()
        return model, extras

    # if model_name == "ml_dmd":
    #     ckpt = torch.load(model_path, map_location=device)
    #     model = ML_DMD(state_dim=ckpt["state_dim"]).to(device)
    #     model.load_state_dict(ckpt["model_state_dict"])
    #     model.eval()
    #     extras["ckpt"] = ckpt
    #     return model, extras

    # if model_name == "ml_eigen_dmd":
    #     ckpt = torch.load(model_path, map_location=device)
    #     model = MLEigenDMD(state_dim=ckpt["state_dim"]).to(device)
    #     model.load_state_dict(ckpt["model_state_dict"])
    #     model.eval()
    #     extras["ckpt"] = ckpt
    #     return model, extras

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
        model = _fit_sindy_from_saved_config(model_path=model_path, data_path=data_path)
        return model, extras

    raise ValueError(f"Unknown model: {model_name}")


def predict_rollout_from_x0(
    *,
    x0: np.ndarray,
    steps: int,
    model_name: str,
    model,
    extras: Dict[str, Any],
) -> np.ndarray:
    if model_name == "linear_baseline":
        return rollout_linear_map(extras["M"], x0=x0, steps=steps)

    if model_name == "dmd_baseline":
        return rollout_dmd_eig(extras["Lambda"], extras["Phi"], x0=x0, steps=steps)

    if model_name == "regression_dmd":
        return model.rollout(
            K=extras["K"],
            C=extras["C"],
            x0=x0,
            steps=steps,
        ).detach().cpu().numpy()

    if model_name == "sindy_baseline":
        return model.rollout(x0, steps=steps)

    return model.rollout(x0=x0, steps=steps).detach().cpu().numpy()