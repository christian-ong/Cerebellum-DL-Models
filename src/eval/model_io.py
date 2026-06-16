import os
from typing import Dict, Tuple, Any

import numpy as np
import torch
import warnings

from src.models.linear_baseline import rollout_linear_map
from src.models.dmd_baseline import rollout_dmd_eig
from src.models.regression_dmd import Regression_DMD
from src.models.ml_linear_dynamics import ML_LinearDynamics
from src.models.ml_dmd import ML_DMD
from src.models.mlp_baseline import MLP_BlackBox
from src.models.sindy_baseline import SINDyBaseline
from src.data_generation.load_data import resolve_split_npz_path


def _canonical_model_name(model_name: str) -> str:
    alias_map = {
        "ml_linear_dynamics": "ml_lineardynamics",
    }
    return alias_map.get(model_name, model_name)

def _finalize_loaded_expander(model, train_args):
    exp_type = str(train_args.get("expansion_type", "general"))

    if exp_type in {"rbf", "hankel_svd"} and hasattr(model, "expander"):
        model.expander.is_fitted = True

    if hasattr(model, "expander"):
        if hasattr(model.expander, "expand_names"):
            model.expand_names = model.expander.expand_names
        if hasattr(model.expander, "state_indices"):
            model.state_indices = model.expander.state_indices
        if hasattr(model.expander, "expanded_dim"):
            model.expanded_dim = model.expander.expanded_dim
            model.latent_dim = model.expander.expanded_dim

def _ensure_device_marker(model, device: str):
    if not isinstance(model, torch.nn.Module):
        return

    try:
        if len(list(model.buffers())) == 0:
            model.register_buffer(
                "_device_marker",
                torch.empty(0, dtype=torch.float32, device=torch.device(device)),
                persistent=False,
            )
    except Exception:
        pass

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


def _to_optional_int(value: Any):
    if value is None:
        return None

    arr = np.asarray(value)
    item = arr.item() if arr.shape == () else value

    if item is None:
        return None

    try:
        int_value = int(item)
    except (TypeError, ValueError):
        return None

    return None if int_value < 0 else int_value


def _unwrap_train_args(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, dict):
        return value

    try:
        return dict(value)
    except Exception:
        return {}

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
    model_name = _canonical_model_name(model_name)
    rollout_mode_override = os.environ.get("EVAL_REGRESSION_ROLLOUT_MODE")

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
        if str(model_path).endswith('.npz'):
            # ==========================================================
            # LEGACY .npz LOADING (Keeps your old sweeps working)
            # ==========================================================
            model_data = np.load(model_path, allow_pickle=True)

            train_args = _unwrap_train_args(model_data["train_args"] if "train_args" in model_data else {})

            rollout_mode = rollout_mode_override or (
                str(train_args.get("regression_rollout_mode", train_args.get("rollout_mode", model_data.get("rollout_mode", "DMD"))))
                if ("rollout_mode" in model_data or "regression_rollout_mode" in train_args or "rollout_mode" in train_args)
                else "DMD"
            )
            extras["rollout_mode"] = rollout_mode

            rank_source = train_args.get("rank", model_data["rank"] if "rank" in model_data else None) if ("rank" in model_data or "rank" in train_args) else None
            rank = _to_optional_int(rank_source)
            hankel_rank_source = train_args.get("hankel_rank", model_data["hankel_rank"] if "hankel_rank" in model_data else None) if ("hankel_rank" in model_data or "hankel_rank" in train_args) else None
            hankel_rank = _to_optional_int(hankel_rank_source)

            model = Regression_DMD(
                state_dim=state_dim,
                expansion_degree=int(train_args.get("expansion_degree", model_data.get("expansion_degree", 3))),
                bias=_to_bool(train_args.get("bias", model_data.get("bias", True)), default=True),
                sine_cosine_expansion=_to_bool(train_args.get("sine_cosine_expansion", model_data.get("sine_cosine_expansion", False)), default=False),
                expansion_type=str(train_args.get("expansion_type", model_data.get("expansion_type", "general"))),
                system=_to_optional_str(model_data.get("system_basis")),
                delay_depth=int(train_args.get("delay_depth", np.asarray(model_data["delay_depth"]).item())) if "delay_depth" in model_data or "delay_depth" in train_args else 1,
                hankel_rank=hankel_rank,
                normalize_state=_to_bool(train_args.get("normalize_state", model_data.get("normalize_state", False)), default=False),
                normalize_lifted=_to_bool(train_args.get("normalize_lifted", model_data.get("normalize_lifted", True)), default=True),
                rollout_mode=rollout_mode,
                ridge=float(train_args.get("ridge", np.asarray(model_data["ridge"]).item())) if "ridge" in model_data or "ridge" in train_args else 0.0,
                rank=rank,
                rbf_n_centers=int(train_args.get("rbf_n_centers", np.asarray(model_data["rbf_n_centers"]).item())) if "rbf_n_centers" in model_data or "rbf_n_centers" in train_args else 50,
                rbf_center_selection=str(train_args.get("rbf_center_selection", np.asarray(model_data["rbf_center_selection"]).item())) if "rbf_center_selection" in model_data or "rbf_center_selection" in train_args else "farthest",
                rbf_bandwidth_mode=str(train_args.get("rbf_bandwidth_mode", np.asarray(model_data["rbf_bandwidth_mode"]).item())) if "rbf_bandwidth_mode" in model_data or "rbf_bandwidth_mode" in train_args else "knn",
                rbf_knn_k=int(train_args.get("rbf_knn_k", np.asarray(model_data["rbf_knn_k"]).item())) if "rbf_knn_k" in model_data or "rbf_knn_k" in train_args else 5,
            ).to(device)

            _ensure_device_marker(model, device)

            if rollout_mode_override:
                model.rollout_mode = rollout_mode_override

            dev = torch.device(device)

            # --- SAFE TENSOR LOADING FOR LEGACY SCALES ---
            def _safe_tensor(val, dtype):
                if val is None or (isinstance(val, np.ndarray) and val.dtype == object and val.item() is None):
                    return None
                return torch.tensor(val, dtype=dtype, device=dev)

            model.x_mean = _safe_tensor(model_data.get("x_mean"), torch.float64)
            model.x_scale = _safe_tensor(model_data.get("x_scale"), torch.float64)
            model.psi_scale = _safe_tensor(model_data.get("psi_scale"), torch.float64)

            if hasattr(model.expander, "state_scale") and "expander_state_scale" in model_data:
                model.expander.state_scale.copy_(
                    torch.as_tensor(model_data["expander_state_scale"], dtype=model.expander.state_scale.dtype, device=dev)
                )
            if hasattr(model.expander, "history_scale") and "expander_history_scale" in model_data:
                model.expander.history_scale.copy_(
                    torch.as_tensor(model_data["expander_history_scale"], dtype=model.expander.history_scale.dtype, device=dev)
                )

            if str(model_data.get("expansion_type", "general")) == "rbf":
                if "rbf_centers" not in model_data or "rbf_sigmas" not in model_data:
                    raise ValueError(
                        "RBF regression_dmd checkpoint is missing rbf_centers/rbf_sigmas. "
                        "Please retrain and resave the model with the updated train.py."
                    )

                model.expander.centers = torch.tensor(model_data["rbf_centers"], dtype=torch.float32, device=dev)
                model.expander.sigmas = torch.tensor(model_data["rbf_sigmas"], dtype=torch.float32, device=dev)
                model.expander.is_fitted = True

                model.expand_names = model.expander.expand_names
                model.state_indices = model.expander.state_indices
                model.expanded_dim = model.expander.expanded_dim

            if str(model_data.get("expansion_type", "general")) == "hankel_svd":
                required = ["hankel_mean", "hankel_components", "hankel_singular_values"]
                missing = [k for k in required if k not in model_data]
                if missing:
                    raise ValueError(
                        f"Hankel-SVD regression_dmd checkpoint is missing {missing}. "
                        "Please retrain and resave the model."
                    )

                h_device = model.expander.mean.device

                model.expander.mean.copy_(
                    torch.as_tensor(model_data["hankel_mean"], dtype=torch.float64, device=h_device)
                )
                model.expander.components.copy_(
                    torch.as_tensor(model_data["hankel_components"], dtype=torch.float64, device=h_device)
                )
                model.expander.singular_values.copy_(
                    torch.as_tensor(model_data["hankel_singular_values"], dtype=torch.float64, device=h_device)
                )

                model.expander.is_fitted = True

                model.expand_names = model.expander.expand_names
                model.state_indices = model.expander.state_indices
                model.expanded_dim = model.expander.expanded_dim

            model.K_fitted = torch.tensor(model_data["K"], dtype=torch.float64, device=dev)
            model.C_fitted = torch.tensor(model_data["C"], dtype=torch.float64, device=dev)

            if "K_tilde" in model_data:
                model.K_tilde_fitted = torch.tensor(model_data["K_tilde"], dtype=torch.float64, device=dev)
            if "U_r" in model_data:
                model.U_r_fitted = torch.tensor(model_data["U_r"], dtype=torch.float64, device=dev)
            if "W_reduced" in model_data:
                model.W_reduced_fitted = torch.tensor(model_data["W_reduced"], dtype=torch.complex128, device=dev)
            if "Lambda" in model_data:
                model.Lambda_fitted = torch.tensor(model_data["Lambda"], dtype=torch.complex128, device=dev)
            if "Phi_lift" in model_data:
                model.Phi_lift_fitted = torch.tensor(model_data["Phi_lift"], dtype=torch.complex128, device=dev)
                model.Phi_fitted = model.Phi_lift_fitted
                model.Phi_pinv_fitted = torch.linalg.pinv(model.Phi_lift_fitted)
            if "Phi_state" in model_data:
                model.Phi_state_fitted = torch.tensor(model_data["Phi_state"], dtype=torch.complex128, device=dev)

            extras["K"] = model_data["K"]
            if "Lambda" in model_data:
                extras["Lambda"] = model_data["Lambda"]

            model.eval()
            return model, extras
            
        else:
            # ==========================================================
            # NEW .pt LOADING (The clean PyTorch way)
            # ==========================================================
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            train_args = checkpoint.get("train_args", {})
            dev = torch.device(device)

            rollout_mode = rollout_mode_override or train_args.get("regression_rollout_mode", train_args.get("rollout_mode", "DMD"))
            extras["rollout_mode"] = rollout_mode
            
            model = Regression_DMD(
                state_dim=state_dim,
                expansion_degree=int(train_args.get("expansion_degree", 3)),
                bias=_to_bool(train_args.get("bias", True), default=True),
                sine_cosine_expansion=_to_bool(train_args.get("sine_cosine_expansion", False), default=False),
                expansion_type=str(train_args.get("expansion_type", "general")),
                system=_to_optional_str(checkpoint.get("system")),
                delay_depth=int(train_args.get("delay_depth", 1)),
                hankel_rank=_to_optional_int(train_args.get("hankel_rank", None)),
                normalize_state=_to_bool(train_args.get("normalize_state", False), default=False),
                normalize_lifted=_to_bool(train_args.get("normalize_lifted", True), default=True),
                rollout_mode=rollout_mode,
                ridge=float(train_args.get("ridge", 0.0)),
                rank=_to_optional_int(train_args.get("rank", None)),
                rbf_n_centers=int(train_args.get("rbf_n_centers", 50)),
                rbf_center_selection=str(train_args.get("rbf_center_selection", "farthest")),
                rbf_bandwidth_mode=str(train_args.get("rbf_bandwidth_mode", "knn")),
                rbf_knn_k=int(train_args.get("rbf_knn_k", 5)),
            ).to(device)

            _ensure_device_marker(model, device)
            
            # Load PyTorch state dict (Restores the Expander parameters all at once!)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            
            # Finalize expander state to ensure correct properties
            if hasattr(model.expander, "is_fitted"):
                model.expander.is_fitted = True
            model.expand_names = model.expander.expand_names
            model.state_indices = model.expander.state_indices
            model.expanded_dim = model.expander.expanded_dim
            
            # Load the exact DMD matrices packed inside train_sweep.py
            if "dmd_matrices" in checkpoint:
                dmd_mats = checkpoint["dmd_matrices"]
                
                # --- SAFE TENSOR LOADING FOR NEW CHECKPOINTS ---
                def _to_tensor(val, dtype):
                    if val is None or (isinstance(val, np.ndarray) and val.dtype == object and val.item() is None):
                        return None
                    return torch.tensor(val, dtype=dtype, device=dev)

                model.K_fitted = _to_tensor(dmd_mats.get("K_full", dmd_mats.get("K")), torch.float64)
                model.C_fitted = _to_tensor(dmd_mats.get("C"), torch.float64)
                model.K_tilde_fitted = _to_tensor(dmd_mats.get("K_tilde"), torch.float64)
                model.U_r_fitted = _to_tensor(dmd_mats.get("U_r"), torch.float64)
                model.W_reduced_fitted = _to_tensor(dmd_mats.get("W_reduced"), torch.complex128)
                model.Lambda_fitted = _to_tensor(dmd_mats.get("Lambda"), torch.complex128)
                
                phi_lift = _to_tensor(dmd_mats.get("Phi_lift"), torch.complex128)
                if phi_lift is not None:
                    model.Phi_lift_fitted = phi_lift
                    model.Phi_fitted = phi_lift
                    model.Phi_pinv_fitted = torch.linalg.pinv(phi_lift)
                
                model.Phi_state_fitted = _to_tensor(dmd_mats.get("Phi_state"), torch.complex128)

                # State scaling params (backed up manually here to perfectly mirror legacy)
                model.x_mean = _to_tensor(dmd_mats.get("x_mean"), torch.float64)
                model.x_scale = _to_tensor(dmd_mats.get("x_scale"), torch.float64)
                model.psi_scale = _to_tensor(dmd_mats.get("psi_scale"), torch.float64)
                
                extras["K"] = dmd_mats.get("K_full", dmd_mats.get("K"))
                if "Lambda" in dmd_mats:
                    extras["Lambda"] = dmd_mats.get("Lambda")

            model.eval()
            return model, extras

    if model_name == "ml_dmd" or model_name =="hardcoded_dmd":
        ckpt = torch.load(model_path, map_location=device)
        train_args = ckpt["train_args"]

        kwargs = {
            "state_dim": ckpt["state_dim"],
            "expansion_degree": train_args["expansion_degree"],
            "expansion_type": train_args["expansion_type"],
            "bias": _to_bool(train_args.get("bias", "true"), default=True),
            "sine_cosine_expansion": _to_bool(train_args.get("sine_cosine_expansion", "false"), default=False),
            "system": ckpt["system"] if train_args["expansion_type"] == "specific" else None,
            "delay_depth": int(train_args.get("delay_depth", 1)),
            "rbf_n_centers": int(train_args.get("rbf_n_centers", 50)),
            "rbf_center_selection": str(train_args.get("rbf_center_selection", "farthest")),
            "rbf_bandwidth_mode": str(train_args.get("rbf_bandwidth_mode", "knn")),
            "rbf_knn_k": int(train_args.get("rbf_knn_k", 5)),
            "hankel_rank": train_args.get("hankel_rank", None),
            "l1_weight": float(train_args.get("l1_weight", 1e-3)),
            "biorth_weight": float(train_args.get("biorth_weight", 0.1))
        }

        model = ML_DMD(**kwargs).to(device)

        model.load_state_dict(ckpt["model_state_dict"])
        _finalize_loaded_expander(model, train_args)
        # Ensure expander/buffer tensors are on the same device as the model
        dev = torch.device(device)
        try:
            # common scale/buffer names that may exist
            if hasattr(model, "x_mean"):
                model.x_mean = model.x_mean.to(dev)
            if hasattr(model, "x_scale"):
                model.x_scale = model.x_scale.to(dev)
            if hasattr(model, "lift_scale"):
                model.lift_scale = model.lift_scale.to(dev)
            if hasattr(model, "psi_scale"):
                model.psi_scale = model.psi_scale.to(dev)
        except Exception:
            pass

        # Place expander internals onto device if present
        if hasattr(model, "expander"):
            for attr in ("centers", "sigmas", "mean", "components", "singular_values"):
                if hasattr(model.expander, attr):
                    try:
                        val = getattr(model.expander, attr)
                        if isinstance(val, torch.Tensor):
                            setattr(model.expander, attr, val.to(dev))
                    except Exception:
                        pass

        # Minimal ML_DMD compatibility aliases used by older diagnostic code
        try:
            if not hasattr(model, "_normalize_x") and hasattr(model, "_normalize"):
                model._normalize_x = lambda x: model._normalize(x)
            if not hasattr(model, "_denormalize_x") and hasattr(model, "_unnormalize"):
                model._denormalize_x = lambda x: model._unnormalize(x)
            if not hasattr(model, "psi_scale") and hasattr(model, "lift_scale"):
                model.psi_scale = getattr(model, "lift_scale")
        except Exception:
            pass

        model.eval()
        extras["ckpt"] = ckpt
        return model, extras
        
    if model_name == "mlp_baseline":
        ckpt = torch.load(model_path, map_location=device)
        train_args = ckpt.get("train_args", {}) if isinstance(ckpt, dict) else {}

        # Defaults when older checkpoints don't include these args
        hidden_dim = int(train_args.get("hidden_dim", 64))
        num_layers = int(train_args.get("num_layers", 4))

        # Try to infer dimensions from the saved state dict if available
        state_dim = ckpt.get("state_dim", None)
        state_dict = ckpt.get("model_state_dict", {}) if isinstance(ckpt, dict) else {}

        try:
            if state_dim is None and "head.weight" in state_dict:
                w = state_dict["head.weight"]
                state_dim = int(w.shape[0])
        except Exception:
            state_dim = state_dim

        try:
            if "hidden_dim" not in train_args and "head.weight" in state_dict:
                w = state_dict["head.weight"]
                hidden_dim = int(w.shape[1])
        except Exception:
            pass

        if state_dim is None:
            state_dim = 2

        model = MLP_BlackBox(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)

        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])

        model.eval()
        extras["ckpt"] = ckpt
        return model, extras

    if model_name == "sindy_baseline":
        model = _load_sindy_from_saved_config(model_path=model_path, data_path=data_path, state_dim=state_dim)
        return model, extras

    raise ValueError(f"Unknown model: {model_name}")


def supports_mode_subset_rollout(model_name: str, model, extras: Dict[str, Any]) -> bool:
    model_name = _canonical_model_name(model_name)
    if model_name == "regression_dmd":
        rollout_mode = extras.get("rollout_mode", "DMD")
        return rollout_mode in {"DMD", "projected_DMD"}
    
    # --- FIX: Add ml_dmd_drop ---
    if model_name in {"ml_dmd"}:
        return True
        
    return False

def predict_rollout_from_x0(*, x0, steps, model_name, model, extras, mode_indices=None):
    model_name = _canonical_model_name(model_name)
    # Safety cap to avoid extremely long blocking rollouts during large sweeps
    MAX_SAFE_ROLLOUT = 5000
    if steps is not None and steps > MAX_SAFE_ROLLOUT:
        warnings.warn(f"Requested rollout steps={steps} exceeds MAX_SAFE_ROLLOUT={MAX_SAFE_ROLLOUT}; capping.", RuntimeWarning)
        steps = MAX_SAFE_ROLLOUT
    if model_name == "linear_baseline":
        return rollout_linear_map(extras["M"], x0=x0, steps=steps)

    if model_name == "dmd_baseline":
        return rollout_dmd_eig(extras["Lambda"], extras["Phi"], x0=x0, steps=steps)
    
    if model_name == "regression_dmd":
        rollout_mode = extras.get("rollout_mode", "DMD")
        
        # --- FIX: UNIVERSAL MODE COUPLING PROTECTION (REGRESSION DMD) ---
        if mode_indices is not None:
            from src.eval.visualize_modes import _get_expanded_indices
            mode_indices = _get_expanded_indices(mode_indices, model)
        # ----------------------------------------------------------------

        rollout = model.rollout(
            x0=x0,
            steps=steps,
            mode=rollout_mode,
            mode_indices=mode_indices,
        )
        return rollout.detach().cpu().numpy()

    if model_name == "sindy_baseline":
        x0_arr = np.asarray(x0)
        if x0_arr.ndim == 2 and x0_arr.shape[0] == 1:
            x0_arr = x0_arr[0]
        if x0_arr.ndim == 1:
            return model.rollout(x0_arr, steps=steps)

        # SINDy rollout is implemented for single trajectories; batch the grid
        # points explicitly so dense-grid diagnostics can stay vectorized.
        trajs = [np.asarray(model.rollout(x0_arr[i], steps=steps)) for i in range(x0_arr.shape[0])]
        trajs = [traj[:, None, :] if traj.ndim == 2 else traj for traj in trajs]
        return np.concatenate(trajs, axis=1)

    with torch.inference_mode():
        # --- Native Mode Subsetting for ML-DMD Models ---
        # --- FIX: Include ml_dmd_drop in this condition ---
        if model_name in {"ml_dmd"} and mode_indices is not None:
            # Safely infer dtype/device from model parameters when possible
            try:
                p = next(model.parameters())
                param_dtype = p.dtype
                param_device = p.device
            except Exception:
                param_dtype = torch.float32
                param_device = torch.device("cpu")

            x0_t = torch.as_tensor(x0, dtype=param_dtype, device=param_device)
            
            is_1d = x0_t.ndim == 1
            if is_1d:
                x0_t = x0_t.unsqueeze(0)
                
            delay_depth = int(getattr(model.expander, "delay_depth", 1))
            expected_width = model.state_dim * delay_depth

            if delay_depth > 1:
                if x0_t.shape[1] == model.state_dim:
                    raise ValueError(
                        f"Delay ML model received only current state, but delay_depth={delay_depth}. "
                        f"Pass full delay history with width {expected_width}."
                    )
                if x0_t.shape[1] != expected_width:
                    raise ValueError(
                        f"Delay ML model expected width {expected_width}, got {x0_t.shape[1]}."
                    )

            x = x0_t

            # 1. Expand and scale using the NEW standardization methods
            z = model.expander.expand(x)
            z_norm = model._normalize(z)
            
            # 2. Project into modal coordinates (b)
            b = model._get_modal_coords(z_norm)

            # 3. Mask unwanted modes
            if mode_indices is None:
                mask = torch.ones_like(b)
            else:
                from src.eval.visualize_modes import _get_expanded_indices
                
                # Use the bulletproof central function for ALL models
                expanded_idx = _get_expanded_indices(mode_indices, model)
                idx_np = np.asarray(expanded_idx, dtype=np.int64)
                
                mask = torch.zeros_like(b)
                mask[:, idx_np] = 1.0
            
            b = b * mask
            
            # 4. Rollout entirely in the latent space
            trajectory = [x[:, :model.state_dim].squeeze(0).cpu().numpy()]
            
            for _ in range(steps):
                # step in modal coords; guard if missing
                if not hasattr(model, "_step_modal"):
                    warnings.warn("ML-DMD model missing _step_modal(); falling back to model.rollout()", RuntimeWarning)
                    # fallback to standard rollout
                    return model.rollout(x0=x0, steps=steps).detach().cpu().numpy()

                b = model._step_modal(b)
                b = b * mask  # Keep masked modes exactly at 0
                
                # 5. Reconstruct back to physical space using updated pipeline
                z_norm_next = model._modal_to_latent(b)
                z_next_phys = model._unnormalize(z_norm_next) if hasattr(model, "_unnormalize") else z_norm_next
                # ensure real/float tensors for de_expand
                if torch.is_complex(z_next_phys):
                    z_next_phys = z_next_phys.real
                z_next_phys = z_next_phys.to(dtype=torch.float32)
                x_next_head = model.expander.de_expand(z_next_phys)
                
                trajectory.append(x_next_head.squeeze(0).cpu().numpy())
                
            return np.array(trajectory)
            
        else:
            # Standard rollout (already updated in your class files)
            return model.rollout(x0=x0, steps=steps).detach().cpu().numpy()