import os
import torch
import numpy as np
import sympy
from scipy.linalg import expm, schur
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from src.models.ml_dmd import ML_DMD
from src.models.ml_dmd_drop import ML_DMD_DROP
from src.models.ml_linear_dynamics import ML_LinearDynamics
from src.models.regression_dmd import Regression_DMD


def _with_subtitle(title, subtitle=None):
    return f"{title}\n{subtitle}" if subtitle else title


def _set_nonzero_ylim(ax, y_lower, y_upper, *, min_pad=1e-6):
    y_lower = float(y_lower)
    y_upper = float(y_upper)

    if not np.isfinite(y_lower) or not np.isfinite(y_upper):
        y_lower, y_upper = -1.0, 1.0
    elif np.isclose(y_lower, y_upper):
        center = 0.5 * (y_lower + y_upper)
        pad = max(min_pad, 0.1 * max(abs(center), 1.0))
        y_lower, y_upper = center - pad, center + pad

    ax.set_ylim(y_lower, y_upper)


def _as_train_args(value):
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



def safe_expand(model, x):
    """Safe expand wrapper: uses `model.expander.expand` if available, otherwise `model.expand`.
    Accepts numpy arrays or torch tensors and returns a torch Tensor.
    """
    t = x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)
    if hasattr(model, "expander") and hasattr(model.expander, "expand"):
        return model.expander.expand(t)
    if hasattr(model, "expand"):
        return model.expand(t)
    raise AttributeError("Model does not expose an expander or expand() method.")


def _safe_matmul(A, B, clip_abs=1e30):
    """Numeric-safe matrix multiply: replaces NaN/Inf, clips extreme values, and multiplies.

    Returns an array with same dtype as numpy.matmul would produce.
    """
    A = np.asarray(A)
    B = np.asarray(B)

    def sanitize(x):
        if np.iscomplexobj(x):
            real = np.nan_to_num(x.real, posinf=clip_abs, neginf=-clip_abs)
            imag = np.nan_to_num(x.imag, posinf=clip_abs, neginf=-clip_abs)
            # clip magnitudes
            real = np.clip(real, -clip_abs, clip_abs)
            imag = np.clip(imag, -clip_abs, clip_abs)
            return real + 1j * imag
        else:
            y = np.nan_to_num(x, posinf=clip_abs, neginf=-clip_abs)
            return np.clip(y, -clip_abs, clip_abs)

    A_s = sanitize(A)
    B_s = sanitize(B)

    try:
        return A_s @ B_s
    except Exception:
        # On any failure, return zeros of expected shape
        out_shape = (A_s.shape[0], B_s.shape[1]) if A_s.ndim == 2 and B_s.ndim == 2 else np.zeros((0,))
        return np.zeros(out_shape, dtype=complex if np.iscomplexobj(A_s) or np.iscomplexobj(B_s) else float)


def safe_de_expand(model, x):
    """Safe de-expand wrapper: uses `model.expander.de_expand` if available, otherwise `model.de_expand`.
    Returns a torch Tensor.
    """
    t = x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)
    if hasattr(model, "expander") and hasattr(model.expander, "de_expand"):
        return model.expander.de_expand(t)
    if hasattr(model, "de_expand"):
        return model.de_expand(t)
    if hasattr(model, "deexpand"):
        return model.deexpand(t)
    raise AttributeError("Model does not expose a de_expander or de_expand() method.")


def build_model_from_checkpoint(model_path, device="cpu"):

    rollout_mode_override = os.environ.get("EVAL_REGRESSION_ROLLOUT_MODE")

    # ---------------------------------------------------------
    # Regression DMD models
    # ---------------------------------------------------------

    if model_path.endswith(".npz"):
        ckpt = np.load(model_path, allow_pickle=True)
        model_name = str(ckpt["model"].item() if hasattr(ckpt["model"], "item") else ckpt["model"])
            
        if model_name != "regression_dmd":
            raise NotImplementedError("Not implemented")
        
        # Helper to safely extract scalars from numpy bounds
        def get_scalar(key, default=None):
            if key in ckpt:
                val = ckpt[key]
                return val.item() if hasattr(val, 'item') else val
            return default

        # 1. Instantiate the shell
        model = Regression_DMD(
            state_dim=int(get_scalar("state_dim")),
            expansion_degree=int(get_scalar("expansion_degree")),
            bias=bool(get_scalar("bias")),
            sine_cosine_expansion=bool(get_scalar("sine_cosine_expansion")),
            expansion_type=str(get_scalar("expansion_type")),
            system=str(get_scalar("system")) if get_scalar("expansion_type") == "specific" else None,
            delay_depth=int(get_scalar("delay_depth")),
            hankel_rank=None if get_scalar("hankel_rank") == -1 else int(get_scalar("hankel_rank")),
            normalize_state=bool(get_scalar("normalize_state")),
            normalize_lifted=bool(get_scalar("normalize_lifted")),
            rollout_mode=rollout_mode_override or str(get_scalar("rollout_mode")),
            ridge=float(get_scalar("ridge")),
            rank=None if get_scalar("rank") == -1 else int(get_scalar("rank")),
            rbf_n_centers=int(get_scalar("rbf_n_centers")),
            rbf_center_selection=str(get_scalar("rbf_center_selection")),
            rbf_bandwidth_mode=str(get_scalar("rbf_bandwidth_mode")),
            rbf_knn_k=int(get_scalar("rbf_knn_k")),
        ).to(device)

        if rollout_mode_override:
            model.rollout_mode = rollout_mode_override

        # 2. Re-hydrate arrays back to model attributes safely
        real_mappings = {
            "x_mean": "x_mean", 
            "x_scale": "x_scale", 
            "psi_scale": "psi_scale",
            "K_fitted": "K", 
            "C_fitted": "C", 
            "K_tilde_fitted": "K_tilde",
            "U_r_fitted": "U_r",
            # --- FIX: Load the SVD energy statistics ---
            "singular_values_fitted": "svd_singular_values",
            "svd_energy_fitted": "svd_energy_cumulative"
        }
        
        complex_mappings = {
            "W_reduced_fitted": "W_reduced",
            "Lambda_fitted": "Lambda", 
            "Phi_lift_fitted": "Phi_lift", 
            "Phi_state_fitted": "Phi_state"
        }
        
        # Load real-valued parameters as float64 (matching fit precision)
        for attr, np_key in real_mappings.items():
            if np_key in ckpt:
                tensor_val = torch.as_tensor(ckpt[np_key], dtype=torch.float64, device=device)
                setattr(model, attr, tensor_val)
                
        # Load spectral operators as complex128 (matching fit precision)
        for attr, np_key in complex_mappings.items():
            if np_key in ckpt:
                tensor_val = torch.as_tensor(ckpt[np_key], dtype=torch.complex128, device=device)
                setattr(model, attr, tensor_val)

        # --- FIX: Reconstruct missing inversion matrices and aliases ---
        if hasattr(model, "Phi_lift_fitted") and model.Phi_lift_fitted is not None:
            phi_lift = model.Phi_lift_fitted
            model.Phi_fitted = phi_lift # Map the alias used in visualization
            
            # Reconstruct the pseudo-inverse needed for projected rollouts
            if phi_lift.ndim > 1:
                model.Phi_pinv_fitted = torch.linalg.pinv(phi_lift)
            else:
                model.Phi_pinv_fitted = torch.linalg.pinv(phi_lift.unsqueeze(1))

        # Restore RBF / Hankel expander buffers and fitted flag so expand()/de_expand() work
        exp_type = str(ckpt.get("expansion_type", "general"))
        if exp_type == "rbf" and hasattr(model, "expander"):
            if "rbf_centers" in ckpt and "rbf_sigmas" in ckpt:
                model.expander.centers = torch.as_tensor(ckpt["rbf_centers"], dtype=torch.float32, device=device)
                model.expander.sigmas = torch.as_tensor(ckpt["rbf_sigmas"], dtype=torch.float32, device=device)
                model.expander.is_fitted = True

                if hasattr(model.expander, "expand_names"):
                    model.expand_names = model.expander.expand_names
                if hasattr(model.expander, "state_indices"):
                    model.state_indices = model.expander.state_indices
                if "expander_state_scale" in ckpt and hasattr(model.expander, "state_scale"):
                    model.expander.state_scale.copy_(torch.as_tensor(ckpt["expander_state_scale"], dtype=model.expander.state_scale.dtype, device=device))
                if "expander_history_scale" in ckpt and hasattr(model.expander, "history_scale"):
                    model.expander.history_scale.copy_(torch.as_tensor(ckpt["expander_history_scale"], dtype=model.expander.history_scale.dtype, device=device))
                if hasattr(model.expander, "expanded_dim"):
                    model.expanded_dim = model.expander.expanded_dim
                    model.latent_dim = model.expander.expanded_dim

        if exp_type == "hankel_svd" and hasattr(model, "expander"):
            required = ["hankel_mean", "hankel_components", "hankel_singular_values"]
            missing = [k for k in required if k not in ckpt]
            if not missing:
                h_device = model.expander.mean.device
                model.expander.mean.copy_(torch.as_tensor(ckpt["hankel_mean"], dtype=torch.float64, device=h_device))
                model.expander.components.copy_(torch.as_tensor(ckpt["hankel_components"], dtype=torch.float64, device=h_device))
                model.expander.singular_values.copy_(torch.as_tensor(ckpt["hankel_singular_values"], dtype=torch.float64, device=h_device))
                model.expander.is_fitted = True

                if hasattr(model.expander, "expand_names"):
                    model.expand_names = model.expander.expand_names
                if hasattr(model.expander, "state_indices"):
                    model.state_indices = model.expander.state_indices
                if hasattr(model.expander, "expanded_dim"):
                    model.expanded_dim = model.expander.expanded_dim
                    model.latent_dim = model.expander.expanded_dim

        model.is_fitted = True
        return model, model_name, _as_train_args(ckpt.get("train_args", {}))

    # ---------------------------------------------------------
    # ML models
    # ---------------------------------------------------------

    ckpt = torch.load(model_path, map_location="cpu")
    model_name = ckpt.get("model", "ml_dmd_free")
    train_args = _as_train_args(ckpt["train_args"])
    
    kwargs = {
        "state_dim": ckpt["state_dim"],
        "expansion_degree": train_args["expansion_degree"],
        "bias": str(train_args.get("bias", "true")).lower() == "true",
        "sine_cosine_expansion": str(train_args.get("sine_cosine_expansion", "false")).lower() == "true",
        "expansion_type": train_args["expansion_type"],
        "system": ckpt["system"] if train_args["expansion_type"] == "specific" else None,
        "delay_depth": int(train_args.get("delay_depth", 1)),
        "hankel_rank": train_args.get("hankel_rank", None),
        "rbf_n_centers": int(train_args.get("rbf_n_centers", 50)),
        "rbf_center_selection": str(train_args.get("rbf_center_selection", "farthest")),
        "rbf_bandwidth_mode": str(train_args.get("rbf_bandwidth_mode", "knn")),
        "rbf_knn_k": int(train_args.get("rbf_knn_k", 5)),
        "l1_weight": float(train_args.get("l1_weight", 1e-6)),
        "biorth_weight": float(train_args.get("biorth_weight", 0.1)),
    }

    if model_name in {"ml_dmd", "hardcoded_dmd", "ml_dmd_free", "ml_dmd_band"}:
        model = ML_DMD(**kwargs)
    elif model_name == "ml_dmd_drop":
        model = ML_DMD_DROP(**kwargs)
    elif model_name in {"ml_lineardynamics", "ml_linear_dynamics"}:
        model = ML_LinearDynamics(**kwargs)
    else:
        raise ValueError(f"Unsupported: {model_name}")

    # Because ALL your ML parameters and scalers are registered buffers, 
    # load_state_dict captures them automatically and perfectly.
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        model.load_state_dict(state_dict)

    # RBF and Hankel-SVD expanders store fitted buffers in the checkpoint's
    # state_dict, but the runtime flag still needs to be restored explicitly.
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

    model.eval()
    return model, model_name, dict(train_args)


def get_koopman_eigensystem(model):
    """
    Extracts the Koopman modes and eigenfunctions STRICTLY from 
    the neural network's learned parameters (Phi and Lambda).
    """
    # ------------------------------------------------------------------
    # Case 1b: Models that expose getters `get_Phi` / `get_Phi_inv` / `get_Lambda`
    # (handles the updated ML_DMD_BAND implementation)
    # ------------------------------------------------------------------
    if hasattr(model, "get_Phi") and hasattr(model, "get_Phi_inv") and hasattr(model, "get_Lambda"):
        Phi_obj = model.get_Phi()
        Lambda_obj = model.get_Lambda()
        K_obj = model.get_K() if hasattr(model, "get_K") else None

        Phi_true = (
            Phi_obj.detach().cpu().numpy()
            if hasattr(Phi_obj, "detach")
            else np.array(Phi_obj)
        )
        Lambda = (
            Lambda_obj.detach().cpu().numpy()
            if hasattr(Lambda_obj, "detach")
            else np.array(Lambda_obj)
        )
        K_true = (
            K_obj.detach().cpu().numpy()
            if (K_obj is not None and hasattr(K_obj, "detach"))
            else (np.array(K_obj) if K_obj is not None else None)
        )

        Phi_inv_obj = model.get_Phi_inv()
        Phi_inv = (
            Phi_inv_obj.detach().cpu().numpy()
            if hasattr(Phi_inv_obj, "detach")
            else np.array(Phi_inv_obj)
        )

        eigvals, V_inner = np.linalg.eig(Lambda)
        _, W_inner = np.linalg.eig(Lambda.T)

        V = Phi_true @ V_inner

        v_norms = np.linalg.norm(V, axis=0)
        V = V / (v_norms + 1e-12)

        # Calculate W using the learned Encoder (get_Phi_inv)
        W = Phi_inv.T @ W_inner
        W = W * (v_norms + 1e-12)

        dominant_rows = np.argmax(np.abs(V), axis=0)
        sort_idx = np.argsort(dominant_rows)

        eigvals = eigvals[sort_idx]
        V = V[:, sort_idx]
        W = W[:, sort_idx]

        for i in range(V.shape[1]):
            dom_idx = np.argmax(np.abs(V[:, i]))
            if np.real(V[dom_idx, i]) < 0:
                V[:, i] *= -1
                W[:, i] *= -1

        return Phi_true, Lambda, Phi_inv, K_true, V, W

    # Regression DMD
    elif hasattr(model, "Phi_lift_fitted") and hasattr(model, "Lambda_fitted"): 
        Phi_true = model.Phi_lift_fitted.detach().cpu().numpy() if hasattr(model.Phi_lift_fitted, "detach") else np.array(model.Phi_lift_fitted)
        Lambda = model.Lambda_fitted.detach().cpu().numpy() if hasattr(model.Lambda_fitted, "detach") else np.array(model.Lambda_fitted)
        K_true = model.K_fitted.detach().cpu().numpy() if hasattr(model.K_fitted, "detach") else np.array(model.K_fitted)

        if Lambda.ndim == 1:
            Lambda = np.diag(Lambda)

        eigvals, V_inner = np.linalg.eig(Lambda)
        _, W_inner = np.linalg.eig(Lambda.T)

        V = Phi_true @ V_inner

        v_norms = np.linalg.norm(V, axis=0)
        V = V / (v_norms + 1e-12)

        Phi_inv_T = np.linalg.pinv(Phi_true).T
        W = Phi_inv_T @ W_inner

        W = W * (v_norms + 1e-12)

        dominant_rows = np.argmax(np.abs(V), axis=0)
        sort_idx = np.argsort(dominant_rows)

        eigvals = eigvals[sort_idx]
        V = V[:, sort_idx]
        W = W[:, sort_idx]

        for i in range(V.shape[1]):
            dom_idx = np.argmax(np.abs(V[:, i]))
            if np.real(V[dom_idx, i]) < 0:
                V[:, i] *= -1
                W[:, i] *= -1

        return Phi_true, Lambda, K_true, V, W

    else:
        raise ValueError("Model format not recognized for eigensystem extraction.")


def get_system_matrices(system="saddle_point", decomp_type="schur", truncate_dim=None):
    
    # system values
    vp_mu = 1.5 # vanderpol
    lv_al = 1.1 # lotka-volterra
    lv_be = 0.4 
    lv_ga = 0.4 
    lv_de = 0.1
    pe_g = 9.81 # pendulum
    pe_l = 1.0
    du_al = -1.0 # duffing
    du_be = 1.0
    du_de = 0.2
    lo_sigma = 10.0 # lorenz
    lo_rho = 28.0
    lo_beta = 8.0 / 3.0
    cs_mu = 0.1 # closed_small
    cs_al = -1.0
    cl_mu = 0.1 # closed_large
    cl_al = -1.0 
    cl_be = 0.8
    cl_ga = -0.4
    cl_de = 0.2
    ct_om = 1.0 # closed_trig
    ct_alpha = -0.8
    ct_bs1 = 0.7
    ct_bc1 = -0.5
    ct_bs2 = 0.4
    ct_bc2 = 0.2
    ct_bs3 = -0.25
    ct_bc3 = 0.15
    ct_bx = 0.3
    ct_bx2 = -0.08

    # calculate
    pe_c = pe_g / pe_l

    A_cs = {
        "saddle_point": np.array([
            [0.2, 0], 
            [0, -0.2]]),
        "degenerate_node": np.array([
            [-0.7, 0.7], 
            [0, -0.7]]),
        "inward_spiral": np.array([
            [-0.5, -2], 
            [2, -0.5]]),
        "inward_spiral_cw": np.array([
            [-0.5, 2], 
            [-2, -0.5]]),
        "harmonic_oscillator": np.array([
            [0, 1.3], 
            [-1.3, 0]]),

        "vanderpol": np.array([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, vp_mu, -vp_mu, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, vp_mu, 2, -1, -vp_mu, 0, 0, 0, 0],
            [0, 0, -2, 2*vp_mu, 0, 0, 1, -2*vp_mu, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, vp_mu, 0, 4, -1, -vp_mu],
            [0, 0, 0, -3, 0, 0, 3*vp_mu, 0, 0, 0],
            [0, 0, 0, 0, 0, -2, 0, 2*vp_mu, 0, 0],
            [0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, vp_mu]]),
        "lotka_volterra": np.array([
            [lv_al, 0, -lv_be, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -lv_ga, lv_de, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, lv_al-lv_ga, -lv_be, lv_de, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, lv_al-2*lv_ga, 0, -lv_be, 2*lv_de, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2*lv_al-lv_ga, 0, -2*lv_be, lv_de, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, lv_al-3*lv_ga, 0, 0, -lv_be, 3*lv_de, 0, 0],
            [0, 0, 0, 0, 0, 0, 2*lv_al-2*lv_ga, 0, 0, -2*lv_be, 2*lv_de, 0],
            [0, 0, 0, 0, 0, 0, 0, 3*lv_al-lv_ga, 0, 0, -3*lv_be, lv_de],
            [0, 0, 0, 0, 0, 0, 0, 0, lv_al-4*lv_ga, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 2*lv_al-3*lv_ga, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3*lv_al-2*lv_ga, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4*lv_al-lv_ga]]),
        "pendulum": np.array([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -pe_c, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -pe_c/2, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, -pe_c, 0, 0, 0, 0, pe_c, 1, 0, 0, 0],
            [0, 0, pe_c/2, 0, 0, 0, 0, 0, -pe_c/2, -2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -3*pe_c/2, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -pe_c, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        "duffing": np.array([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-du_al, -du_de, -du_be, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -du_al, -du_de, 2, -du_be, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -2*du_al, -2*du_de, 0, 1, -2*du_be, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
            [0, 0, 0, 0, -3*du_al, 0, -3*du_de, 0, -3*du_be, 0, 0, 0],
            [0, 0, 0, 0, 0, -du_al, 0, -du_de, 4, -du_be, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -2*du_al, -2*du_de, 0, 3, -2*du_be],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7],
            [0, 0, 0, 0, 0, 0, 0, -3*du_al, -3*du_de, 0, 0, -3*du_be],
            [0, 0, 0, 0, 0, 0, 0, 0, 6, -du_al, 0, -du_de]]),
        "lorenz": np.array([
            [-lo_sigma, lo_sigma, 0, 0, 0, 0, 0, 0, 0, 0],
            [lo_rho, -1, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, -lo_beta, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, -(lo_sigma+lo_beta), 0, lo_sigma, 1, 0, 0, 0],
            [0, 0, 0, 0, -(lo_sigma+1), 0, 0, lo_sigma, lo_rho, -1],
            [0, 0, 0, lo_rho, 0, -(lo_beta+1), 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -(2*lo_sigma+1), 0, 0, 0],
            [0, 0, 0, 0, 2*lo_sigma, 0, 0, -2*lo_sigma, 0, 0],
            [0, 0, 0, 0, 2*lo_rho, 0, 0, 0, -2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -(2*lo_sigma+lo_beta)]]),

        "closed_small": np.array([
            [cs_mu, 0, 0],
            [0, cs_al, -cs_al],
            [0, 0, 2*cs_mu]]),
        "closed_large": np.array([
            [cl_mu, 0, 0, 0, 0],
            [0, cl_al, cl_be, cl_ga, cl_de],
            [0, 0, 2*cl_mu, 0, 0],
            [0, 0, 0, 3*cl_mu, 0],
            [0, 0, 0, 0, 4*cl_mu]]),
        "closed_trig_small": np.array([
            [0, 0, 0, 0, 0, 0], 
            [ct_om, 0, 0, 0, 0,0],
            [0, ct_bx, ct_alpha, ct_bx2, ct_bs1, ct_bc1],
            [0, 2*ct_om, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, ct_om], 
            [0, 0, 0, 0, -ct_om, 0]]),
        "closed_trig_medium": np.array([
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [ct_om, 0, 0, 0, 0, 0, 0, 0],
            [0, ct_bx, ct_alpha, ct_bx2, ct_bs1, ct_bc1, ct_bs2, ct_bc2],
            [0, 2*ct_om, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, ct_om, 0, 0], 
            [0, 0, 0, 0, -ct_om, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 2*ct_om], 
            [0, 0, 0, 0, 0, 0, -2*ct_om, 0]]),
        "closed_trig_large": np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [ct_om, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, ct_bx, ct_alpha, ct_bx2, ct_bs1, ct_bc1, ct_bs2, ct_bc2, ct_bs3, ct_bc3],
            [0, 2*ct_om, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, ct_om, 0, 0, 0, 0], 
            [0, 0, 0, 0, -ct_om, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 2*ct_om, 0, 0], 
            [0, 0, 0, 0, 0, 0, -2*ct_om, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3*ct_om], 
            [0, 0, 0, 0, 0, 0, 0, 0, -3*ct_om, 0]])
    }

    expansion_names = {
        "saddle_point": ["x", "y"],
        "degenerate_node": ["x", "y"],
        "inward_spiral": ["x", "y"],
        "inward_spiral_cw": ["x", "y"],
        "harmonic_oscillator": ["x", "y"],

        "vanderpol": ["x", "y", "x^2y", "xy^2", "x^3", "x^4y", "y^3", "x^3y^2", "x^5", "x^6y"],
        "lotka_volterra": ["x", "y", "xy", "xy^2", "x^2y", "xy^3", "x^2y^2", "x^3y", "xy^4", "x^2y^3", "x^3y^2", "x^4y"],
        "pendulum": ["x", "y", "sin(x)", "y cos(x)", "sin(2x)", "y^2 sin(x)", "y cos(2x)", "y^3 cos(x)", "sin(3x)", "y^2 sin(2x)", "y^4 sin(x)"],
        "duffing": ["x", "y", "x^3", "x^2y", "xy^2", "x^5", "y^3", "x^4y", "x^3y^2", "x^7", "x^2y^3", "x^6y"],
        "lorenz": ["x", "y", "z", "xz", "xy", "yz", "x^2y", "x^2", "y^2", "x^2z"],

        "closed_small": ["x", "y", "x^2"],
        "closed_large": ["x", "y", "x^2", "x^3", "x^4"],
        "closed_trig_small": ["1", "x", "y", "x^2", "sin(x)", "cos(x)"],
        "closed_trig_medium": ["1", "x", "y", "x^2", "sin(x)", "cos(x)", "sin(2x)", "cos(2x)"],
        "closed_trig_large": ["1", "x", "y", "x^2", "sin(x)", "cos(x)", "sin(2x)", "cos(2x)", "sin(3x)", "cos(3x)"]
    }

    if system not in A_cs:
        raise ValueError(f"System '{system}' not found. Available systems: {list(A_cs.keys())}")

    # ==========================================================

    # Truncate
    if truncate_dim is not None:
        A_cs[system] = A_cs[system][:truncate_dim, :truncate_dim]
        expansion_names[system] = expansion_names[system][:truncate_dim]

    A_c = A_cs[system]
    system_expansion_names = expansion_names[system]

    # # OBS : try to swap y and x^2
    # A_c[2], A_c[3] = A_c[3], A_c[2]
    # system_expansion_names[2], system_expansion_names[3] = system_expansion_names[3], system_expansion_names[2]

    # Compute A_d
    dt = 0.01
    A_d = expm(A_c * dt)

    # Compute eigendecomposition
    if decomp_type == "numpy":
        eigvals, Phi = np.linalg.eig(A_d)
        Lambda = np.diag(eigvals)
    elif decomp_type == "jordan":
        Lambda, Phi = get_sorted_jordan_form(A_d)
    elif decomp_type == "schur":
        Lambda, Phi = schur(A_d)
    else:
        raise ValueError(f"Invalid decomposition type: {decomp_type}")

    return A_c, A_d, Lambda, Phi, system_expansion_names


def sort_block_diagonal_modes(Lambda, Phi, block_tol=1e-12):
    """Sort contiguous diagonal blocks without splitting Jordan chains."""

    block_slices = []
    start_idx = 0
    while start_idx < Lambda.shape[0]:
        end_idx = start_idx + 1
        while end_idx < Lambda.shape[0] and abs(Lambda[end_idx - 1, end_idx]) > block_tol:
            end_idx += 1
        block_slices.append(slice(start_idx, end_idx))
        start_idx = end_idx

    block_keys = []
    for block_slice in block_slices:
        block_phi = Phi[:, block_slice]
        dominant_rows = np.argmax(np.abs(block_phi), axis=0)
        block_keys.append((int(np.min(dominant_rows)), block_slice.start))

    sort_idx = [i for i, _ in sorted(enumerate(block_keys), key=lambda item: item[1])]
    sorted_indices = np.concatenate([
        np.arange(block_slices[i].start, block_slices[i].stop) for i in sort_idx
    ])

    return Lambda[sorted_indices][:, sorted_indices], Phi[:, sorted_indices]


def get_sorted_jordan_form(K_d_analytic, block_tol=1e-12):
    """Return the analytic Jordan form with contiguous blocks kept intact."""

    sympy_mat = sympy.Matrix(K_d_analytic)
    P, J = sympy_mat.jordan_form()

    Lambda_jordan = np.array(J).astype(np.complex128)
    V_theory = np.array(P).astype(np.complex128)

    block_slices = []
    start_idx = 0
    while start_idx < Lambda_jordan.shape[0]:
        end_idx = start_idx + 1
        while end_idx < Lambda_jordan.shape[0] and abs(Lambda_jordan[end_idx - 1, end_idx]) > block_tol:
            end_idx += 1
        block_slices.append(slice(start_idx, end_idx))
        start_idx = end_idx

    block_keys = []
    for block_slice in block_slices:
        dominant_rows = np.argmax(np.abs(V_theory[:, block_slice]), axis=0)
        block_keys.append((int(np.min(dominant_rows)), block_slice.start))

    sort_idx = [i for i, _ in sorted(enumerate(block_keys), key=lambda item: item[1])]
    sorted_indices = np.concatenate([
        np.arange(block_slices[i].start, block_slices[i].stop) for i in sort_idx
    ])

    Lambda_jordan = Lambda_jordan[sorted_indices][:, sorted_indices]
    V_theory = V_theory[:, sorted_indices]

    return Lambda_jordan, V_theory


def find_complex_pairs(
        Lambda, 
        rtol=1e-2,       # 1% relative error allowed
        atol=1e-5,       # Absolute noise floor
        include_jordan_blocks=True):
    """
    Detects pairs of complex conjugate modes (and optionally Jordan blocks).
    Uses RELATIVE thresholding and global search to find pairs anywhere in the matrix.
    """
    
    if np.iscomplexobj(Lambda):
        block_indices = []
        block_scores = []
        paired = set()
        
        L_diag = np.diag(Lambda) if Lambda.ndim == 2 else Lambda

        for i in range(len(L_diag)):
            if i in paired:
                continue
            
            mag_i = abs(L_diag[i])
            if abs(L_diag[i].imag) > atol:
                diffs = np.abs(L_diag - L_diag[i].conj())
                
                diffs[i] = np.inf
                for p in paired:
                    diffs[p] = np.inf
                    
                best_match = int(np.argmin(diffs))
                dynamic_threshold = atol + rtol * mag_i
                
                if diffs[best_match] < dynamic_threshold:
                    block_indices.append((i, best_match))
                    score = 1.0 / (diffs[best_match] + 1e-12)
                    block_scores.append(score)
                    
                    paired.add(i)
                    paired.add(best_match)

    elif np.isrealobj(Lambda):
        block_indices = []
        block_scores = []
        n_modes = len(Lambda)

        # ---> FIX: Global search across ALL possible (i, j) pairs!
        for i in range(n_modes):
            for j in range(i + 1, n_modes):
                a = Lambda[i, i]
                b = Lambda[i, j]
                c = Lambda[j, i]
                d = Lambda[j, j]

                # Scale of this specific 2x2 interaction
                block_scale = max(abs(a), abs(b), abs(c), abs(d), atol)
                
                dynamic_diag_thresh = atol + rtol * block_scale
                dynamic_off_thresh = atol + (rtol * 0.5) * block_scale 

                is_rotation = (
                    (abs(b) > dynamic_off_thresh) and 
                    (abs(c) > dynamic_off_thresh) and 
                    (np.sign(b) != np.sign(c))
                )
                
                is_jordan = False
                if include_jordan_blocks:
                    is_jordan = (
                        (max(abs(b), abs(c)) > dynamic_off_thresh) and 
                        (min(abs(b), abs(c)) < dynamic_off_thresh)
                    )

                diagonals_match = abs(a - d) < dynamic_diag_thresh

                if (is_rotation or is_jordan) and diagonals_match:
                    block_indices.append((i, j))

                    magnitude = abs(b) + abs(c)
                    diag_penalty = abs(a - d)
                    
                    if is_rotation:
                        skew_penalty = abs(b + c) 
                    else:
                        skew_penalty = 0.0 

                    score = magnitude - diag_penalty - skew_penalty
                    block_scores.append(score)

    sorted_scores_idx = np.argsort(block_scores)[::-1] 
    sorted_block_scores = [block_scores[i] for i in sorted_scores_idx]
    sorted_block_indices = [block_indices[i] for i in sorted_scores_idx]

    final_blocks_idx = []
    final_idxs = set()
    
    for idx, score in zip(sorted_block_indices, sorted_block_scores):
        if not any(k in final_idxs for k in idx): 
            final_blocks_idx.append(idx)
            final_idxs.update(idx)

    return final_blocks_idx
    

def rotation_blocks_to_complex(Lambda, Phi, complex_pair_idx, W=None):
    Lambda_complex = Lambda.copy().astype(np.complex128)
    Phi_complex = Phi.copy().astype(np.complex128)
    if W is not None:
        W_complex = W.copy().astype(np.complex128)

    for idx in complex_pair_idx:
        i, j = idx
        
        a = Lambda[i, i]
        c = Lambda[i, j]
        b = Lambda[j, i]
        d = Lambda[j, j]

        real_part = (a + d) / 2.0
        det = a * d - b * c
        discriminant = real_part**2 - det
        
        if discriminant >= 0:
            # Leave Jordan blocks and real pairs strictly alone
            continue
            
        # ---> FIX: Exact Similarity Transform (P^-1 Lambda P) <---
        # Applying this to the full row/column guarantees any off-diagonal 
        # couplings the network learned are preserved perfectly in complex space!
        
        # 1. Transform Lambda (Left multiply by P^-1 on rows)
        row_i = Lambda_complex[i, :].copy()
        row_j = Lambda_complex[j, :].copy()
        Lambda_complex[i, :] = 0.5 * (row_i - 1j * row_j)
        Lambda_complex[j, :] = 0.5 * (row_i + 1j * row_j)
        
        # 2. Transform Lambda (Right multiply by P on cols)
        col_i = Lambda_complex[:, i].copy()
        col_j = Lambda_complex[:, j].copy()
        Lambda_complex[:, i] = col_i + 1j * col_j
        Lambda_complex[:, j] = col_i - 1j * col_j

        # 3. Transform Phi (Right multiply by P)
        phi_i = Phi_complex[:, i].copy()
        phi_j = Phi_complex[:, j].copy()
        Phi_complex[:, i] = phi_i + 1j * phi_j
        Phi_complex[:, j] = phi_i - 1j * phi_j

        # 4. Transform W (Right multiply by (P^-1)^T )
        if W is not None:
            w_i = W_complex[:, i].copy()
            w_j = W_complex[:, j].copy()
            W_complex[:, i] = 0.5 * (w_i - 1j * w_j)
            W_complex[:, j] = 0.5 * (w_i + 1j * w_j)

    if W is not None:
        return Lambda_complex, Phi_complex, W_complex
    return Lambda_complex, Phi_complex


def plot_transition_matrices(matrices, title, model_expansion_names, analytic_expansion_names, threshold_include_val = 1e-3, save_path=None, subtitle=None):
    fig = plt.figure(figsize=(21.5, 11.2))
    # Create a 2x3 grid
    num_rows = int(np.ceil(len(matrices) / 3))
    gs = gridspec.GridSpec(num_rows, 3, width_ratios=[1, 1, 1], wspace=0.02, hspace=0.30)
    axes = [fig.add_subplot(gs[i,j]) for i in range(num_rows) for j in range(3)]

    fig.suptitle(_with_subtitle(title, subtitle), fontsize=21, y=0.985, x=0.5, ha="center")

    for i, (M, subtitle) in enumerate(matrices):
        ax = axes[i]
        M_mag = np.abs(M)
        im = ax.imshow(M_mag, vmin=0) # color by magnitude, but show values as real/imaginary parts
        ax.set_title(subtitle, fontsize=13)
        fig.colorbar(im, ax=ax, fraction=0.038, pad=0.018)
        
        # Model or analytic expansion names (for x/y ticks)
        expansion_names = model_expansion_names if "model" in subtitle.lower() else analytic_expansion_names

        # --- Corrected X Ticks ---
        if "K" in subtitle:
            # K maps basis functions to basis functions (Square Matrix)
            ax.set_xticks(range(len(expansion_names)))
            ax.set_xticklabels(expansion_names, rotation=60, fontsize=8)
        else:
            # Remove x-labels for Phi (modes) and Lambda (modes)
            ax.set_xticks([])

        # --- Corrected Y Ticks ---
        if "K" in subtitle or "Phi" in subtitle:
            # Both K and Phi rows index the basis functions (Observables)
            ax.set_yticks(range(len(expansion_names)))
            ax.set_yticklabels(expansion_names, fontsize=8)
        else:
            # Remove y-labels for Lambda (modes)
            ax.set_yticks([])

        # Dynamically scale font size based on matrix dimension
        n_cols = M.shape[1]
        f_size = 8

        for (row, col), v in np.ndenumerate(M):
            if abs(v) > threshold_include_val:
                re_val, im_val = np.real(v), np.imag(v)
                
                if abs(im_val) < threshold_include_val:
                    txt = f"{re_val:.4f}"
                elif abs(re_val) < threshold_include_val:
                    txt = f"{im_val:.4f}j"
                else:
                    sign = "+" if im_val > 0 else "-"
                    txt = f"{re_val:.4f}\n{sign}{abs(im_val):.4f}j"

                ax.text(
                    col, row, txt,
                    ha="center", va="center",
                    fontsize=f_size, color="red"
                )

    # if len(matrices) != 5:
    #     for i in range(len(matrices), len(axes)):
    #         axes[i].axis("off")

    fig.subplots_adjust(left=0.055, right=0.985, top=0.88, bottom=0.08)
        
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def get_data_bounds_and_grid_points(trajectories, grid_res=100, state_dim=2):
    # Find min and max dynamically for ALL dimensions
    state_bounds = []
    for dim in range(state_dim):
        dim_min, dim_max = trajectories[:, :, dim].min(), trajectories[:, :, dim].max()
        state_bounds.append((dim_min, dim_max))

    # Dynamic Grid setup based on true boundaries (taking 2D slice for 3D+ systems)
    
    x_range = np.linspace(state_bounds[0][0], state_bounds[0][1], grid_res)
    y_range = np.linspace(state_bounds[1][0], state_bounds[1][1], grid_res)
    X, Y = np.meshgrid(x_range, y_range)
    grid_cols = [X.ravel(), Y.ravel()]

    # Pad higher dimensions with their mean trajectory values
    for dim in range(2, state_dim):
        dim_mean = trajectories[:, :, dim].mean()
        grid_cols.append(np.full_like(X.ravel(), dim_mean))
    grid_points = np.column_stack(grid_cols)

    return state_bounds, grid_points


def get_real_representation(V, eigvals, jordan_value=1.0, threshold_imag=1e-5, threshold_jordan=1, W=None):
    V_real = np.copy(np.real(V).astype(np.float64))
    Lambda_real = np.copy(np.real(eigvals).astype(np.float64))
    if W is not None:
        W_real = np.copy(np.real(W).astype(np.float64))

    i = 0
    while i < len(eigvals):
        if abs(np.imag(eigvals[i,i])) < threshold_imag: 
            V_real[:, i], Lambda_real[i, i] = np.real(V[:, i]), np.real(eigvals[i, i])
            if W is not None: W_real[:, i] = np.real(W[:, i])
            
            if jordan_value != 0 and (i + 2 <= len(eigvals) and (abs(eigvals[i+1,i]) > threshold_jordan or abs(eigvals[i,i+1]) > threshold_jordan)): 
                Lambda_real[i, i+1] = jordan_value 
            i += 1
        else: 
            if i + 1 < len(eigvals):
                V_real[:, i], V_real[:, i+1] = np.real(V[:, i]), np.imag(V[:, i]) 
                if W is not None: W_real[:, i], W_real[:, i+1] = np.real(W[:, i]), np.imag(W[:, i]) 
                
                a, b, c, d = np.real(eigvals[i,i]), np.imag(eigvals[i,i]), np.imag(eigvals[i+1,i+1]), np.real(eigvals[i+1,i+1])
                Lambda_real[i, i], Lambda_real[i, i+1] = a, c
                Lambda_real[i+1, i], Lambda_real[i+1, i+1] = b, d
                i += 2
            else:
                V_real[:, i], Lambda_real[i, i] = np.real(V[:, i]), np.real(eigvals[i, i])
                if W is not None: W_real[:, i] = np.real(W[:, i])
                i += 1
                
    if W is not None:
        return V_real, Lambda_real, W_real
    return V_real, Lambda_real


def plot_koopman_mode_rollout(
    model, Phi, Lambda, real_traj, save_path=None, 
    model_type="regression_dmd", subtitle=None, 
    main_title="Koopman Eigenfunction Rollout (Real Part)",
    W=None
):
    n_trajs = real_traj.shape[1]
    state_dim = real_traj.shape[2]
    n_modes = Phi.shape[1]
    plot_n_modes = min(n_modes, 10)
    lifted_dim = Phi.shape[0]
    
    delay_depth = int(getattr(model.expander, "delay_depth", 1)) if hasattr(model, "expander") else 1
    
    if delay_depth > 1:
        q = delay_depth
        T = real_traj.shape[0]
        history_traj = []
        for t in range(q-1, T):
            window = real_traj[t-q+1 : t+1, :, :] 
            window = window[::-1, :, :]
            history_traj.append(window.transpose(1, 0, 2).reshape(n_trajs, -1))
        real_traj_input = np.concatenate(history_traj, axis=0)
        real_traj_valid = real_traj[q-1:, :, :]
    else:
        real_traj_input = torch.tensor(real_traj.reshape(-1, state_dim), dtype=torch.float32)
        real_traj_valid = real_traj
        
    n_steps_valid = real_traj_valid.shape[0]
        
    device = "cpu"
    if hasattr(model, "psi_scale"): device = model.psi_scale.device
    elif hasattr(model, "lift_scale"): device = model.lift_scale.device
    elif hasattr(model, "parameters") and list(model.parameters()): device = next(model.parameters()).device
        
    real_traj_input_t = torch.as_tensor(real_traj_input, dtype=torch.float32, device=device)
    
    # ---> FIX: Normalize physical state before expansion <---
    if hasattr(model, "_normalize_x"):
        real_traj_input_scaled = model._normalize_x(real_traj_input_t)
    else:
        real_traj_input_scaled = real_traj_input_t
        
    # 1. Expand and NORMALIZE the true trajectory
    expanded_traj_flattened = safe_expand(model, real_traj_input_scaled)
    if hasattr(model, "_normalize"):
        expanded_traj_flattened = model._normalize(expanded_traj_flattened)
    elif hasattr(model, "psi_scale"):
        expanded_traj_flattened = expanded_traj_flattened / model.psi_scale.to(expanded_traj_flattened.device)
        
    expanded_traj = expanded_traj_flattened.cpu().numpy().reshape(n_steps_valid, n_trajs, -1)
    
    init_conditions = torch.as_tensor(real_traj_valid[0, :, :], dtype=torch.float32) 
    expanded_init_conditions = expanded_traj[0, :, :]

    # We project using the provided (truncated & sorted) W
    real_proj = (expanded_traj @ W).real

    # 2. Get latent rollout directly from the model
    if model is None or not hasattr(model, "rollout"):
        model_proj = np.repeat(real_proj[0:1, :, :], n_steps_valid, axis=0)
    else:
        if delay_depth > 1:
            window = real_traj[0:delay_depth, :, :]
            window = window[::-1, :, :]
            init_hist = window.transpose(1, 0, 2).reshape(n_trajs, -1)
            init_input = torch.as_tensor(init_hist, dtype=torch.float32).to(init_conditions.device)
        else:
            init_input = init_conditions
        
        # USE THE NEW FLAG: return_latent=True
        rollout_latent = model.rollout(init_input, steps=n_steps_valid-1, return_latent=True)
        z_rollout = rollout_latent.detach() if hasattr(rollout_latent, "detach") else torch.as_tensor(rollout_latent)
        
        # Ensure the latent states are normalized before projection
        # (Regression DMD natively returns normalized, ML DMD natively returns physical)
        if hasattr(model, "expansion_type") and hasattr(model, "_normalize"):
             z_rollout = model._normalize(z_rollout)
             
        z_rollout_np = z_rollout.cpu().numpy()
        
        # Multiply by W to get directly into the sorted modal coordinates
        model_proj = (z_rollout_np @ W).real

    # --- DYNAMIC W LOGIC ---
    # Unify mode evolution: independently evolve the requested modes using the 
    # specific Lambda (Real Block-Diagonal or Complex Diagonal) passed to the function.
    W_mat = W if W is not None else np.linalg.pinv(Phi).T
    z0 = expanded_init_conditions @ W_mat
    lambdas = np.diag(Lambda) if Lambda.ndim == 2 else Lambda
    mode_evolution = np.zeros((n_steps_valid, n_trajs, n_modes), dtype=complex)
    mode_evolution[0, :, :] = z0

    for t in range(1, n_steps_valid):
        mode_evolution[t, :, :] = mode_evolution[t-1, :, :] * lambdas
        
    mode_evolution = mode_evolution.real

    # 1. Base width is slim (2.5 per mode). Minimum 6.5 inches just so the legend fits.
    fig_width = max(6.5, 2.5 * plot_n_modes)
    fig, axes = plt.subplots(n_trajs, plot_n_modes, figsize=(fig_width, 10), sharex=True)
    
    # Safely ensure axes is a 2D array even if plot_n_modes=1 or n_trajs=1
    if n_trajs == 1 and plot_n_modes == 1:
        axes = np.array([[axes]])
    elif n_trajs == 1:
        axes = axes[np.newaxis, :]
    elif plot_n_modes == 1:
        axes = axes[:, np.newaxis]

    time = np.arange(n_steps_valid)

    for i in range(plot_n_modes): 
        for traj_idx in range(n_trajs): 
            if i == 0: axes[traj_idx,i].set_ylabel(f"Trajectory {traj_idx + 1}")
            if traj_idx == 0: axes[traj_idx,i].set_title(f"Mode {i + 1}", fontsize=10)

            axes[traj_idx,i].plot(time, real_proj[:, traj_idx, i], 'k-', label='Ground Truth (Proj)', alpha=0.6)
            axes[traj_idx,i].plot(time, model_proj[:, traj_idx, i], 'r--', label='Model Rollout')
            axes[traj_idx,i].plot(time, mode_evolution[:, traj_idx, i], 'b:', label='$\lambda$ Evolution')

            y_min_real, y_max_real = real_proj[:, traj_idx, i].min(), real_proj[:, traj_idx, i].max()
            range_real = max(y_max_real - y_min_real, 1e-12)
            y_lower_bound_real, y_upper_bound_real = y_min_real - range_real * 0.1, y_max_real + range_real * 0.1
            y_range_real = y_upper_bound_real - y_lower_bound_real

            y_min_other = min(model_proj[:, traj_idx, i].min(), mode_evolution[:, traj_idx, i].min())
            y_max_other = max(model_proj[:, traj_idx, i].max(), mode_evolution[:, traj_idx, i].max())
            range_other = max(y_max_other - y_min_other, 1e-12)
            y_lower_bound_other, y_upper_bound_other = y_min_other - range_other * 0.1, y_max_other + range_other * 0.1
            y_range_other = y_upper_bound_other - y_lower_bound_other
            
            if y_range_other > 10 * y_range_real:
                _set_nonzero_ylim(axes[traj_idx, i], y_lower_bound_real - y_range_real * 2, y_upper_bound_real + y_range_real * 2)
            else:
                y_lower_bound_final, y_upper_bound_final = min(y_lower_bound_real, y_lower_bound_other), max(y_upper_bound_real, y_upper_bound_other)
                if np.isclose(y_lower_bound_final, y_upper_bound_final):
                    pad = max(1.0, abs(y_upper_bound_final) * 0.1)
                    _set_nonzero_ylim(axes[traj_idx, i], y_lower_bound_final - pad, y_upper_bound_final + pad)
                elif abs(y_upper_bound_final) > 1e-12 and not 0.9 < abs(y_lower_bound_final / y_upper_bound_final) < 1.1:
                    _set_nonzero_ylim(axes[traj_idx, i], y_lower_bound_final, y_upper_bound_final)
                elif abs(y_upper_bound_final) <= 1e-12:
                    _set_nonzero_ylim(axes[traj_idx, i], y_lower_bound_final - 1.0, y_upper_bound_final + 1.0)

            axes[traj_idx, i].grid(True, linestyle="--", alpha=0.5)
            if traj_idx == n_trajs - 1: axes[traj_idx, i].set_xlabel("Time Steps")
        
    # 2. Dynamic Title Sizing
    title_fontsize = min(22, max(14, int(1.5 * fig_width)))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    # 3. Legend ALWAYS at the bottom center, no exceptions
    fig.suptitle(_with_subtitle(main_title, subtitle), fontsize=title_fontsize, y=0.96)
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.02), fontsize=12, frameon=False)
    
    # 4. Protect top and bottom margins for the text and legend
    plt.tight_layout(rect=(0, 0.08, 1, 0.92))
        
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_eigenfunctions(
        grid_points, 
        grid_points_expanded, 
        W,
        scores,
        eigvals,
        score_metric,
        complex_pair_idx,
        cmap="inferno", 
    save_path=None,
    subtitle=None,
    ):
    # Compute eigenfunction values on the grid for the top N modes
    eigenfunction_vals = grid_points_expanded @ W

    # Compute complex pairs for annotation
    complex_pairs = [sorted([int(i+1), int(j+1)]) for i, j in complex_pair_idx] # 1-based index and internal sort

    grid_n = int(np.sqrt(len(grid_points)))
    extent = [grid_points[:,0].min(), grid_points[:,0].max(), grid_points[:,1].min(), grid_points[:,1].max()]
    num_modes = eigenfunction_vals.shape[1]

    # --- FIX 1: Adjust figsize to give the 4 rows enough vertical space to be square ---
    fig_width = max(6.0, 3.0 * num_modes)
    fig, axes = plt.subplots(4, num_modes, figsize=(fig_width, 11.0))
    
    # Safely ensure axes is 2D if num_modes is 1
    if num_modes == 1:
        axes = axes[:, np.newaxis]
    
    # --- FIX: Track visible pairs dynamically ---
    visible_pair_counter = 1
    pair_to_stars = {}
    # --------------------------------------------

    for mode_idx in range(num_modes):
        data_map = {
            "Real": np.real(eigenfunction_vals[:, mode_idx]), 
            "Imag": np.imag(eigenfunction_vals[:, mode_idx]),
            "Mag": np.abs(eigenfunction_vals[:, mode_idx]), 
            "Phase": np.angle(eigenfunction_vals[:, mode_idx])
        }
    
        # Annotate complex pairs
        pair_string = ""
        for pair in complex_pairs:
            if mode_idx+1 in pair: # if this mode is part of a complex pair
                pair_tuple = tuple(pair)
                
                # Assign the next available asterisk count if we haven't seen this pair yet
                if pair_tuple not in pair_to_stars:
                    pair_to_stars[pair_tuple] = '*' * visible_pair_counter
                    visible_pair_counter += 1
                    
                pair_string += f" {pair_to_stars[pair_tuple]}\n"

        for i, (label, data) in enumerate(data_map.items()):
            if i==0:
                label = (
                    pair_string + 
                    rf"$\mathbf{{EF{mode_idx+1}}}$" + 
                    f"\n{score_metric}: {scores[mode_idx]:.3f}" +
                    f"\n$\lambda$: {eigvals[mode_idx]:.3g}" +
                    f"\n\n{label}")
            ax = axes[i, mode_idx]
            im = ax.imshow(data.reshape(grid_n, grid_n), extent=extent, origin="lower", cmap=cmap, aspect='auto')
            
            # --- FIX 2: Force the subplot box to be a perfect square ---
            ax.set_box_aspect(1)
            
            ax.set_title(f"{label}")
            
            # Adjust colorbar fraction so it matches the newly squared plot
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(_with_subtitle(f"{num_modes} first eigenfunctions", subtitle), fontsize=20, y=0.99)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def modes_by_mse(model, Phi, Lambda, real_traj, W=None):
    n_trajs = real_traj.shape[1]
    state_dim = real_traj.shape[2]
    n_modes = Phi.shape[1]
    
    delay_depth = int(getattr(model.expander, "delay_depth", 1)) if hasattr(model, "expander") else 1
    
    if delay_depth > 1:
        q = delay_depth
        T = real_traj.shape[0]
        history_traj = []
        # Consume the first q-1 steps strictly as history
        for t in range(q-1, T):
            window = real_traj[t-q+1 : t+1, :, :] 
            window = window[::-1, :, :] 
            history_traj.append(window.transpose(1, 0, 2).reshape(n_trajs, -1))
        real_traj_input = np.concatenate(history_traj, axis=0)
        # The valid trajectory we evaluate against starts AFTER the history
        real_traj_valid = real_traj[q-1:, :, :]
    else:
        real_traj_input = real_traj.reshape(-1, state_dim)
        real_traj_valid = real_traj
        
    n_steps_valid = real_traj_valid.shape[0]
    
    device = "cpu"
    if hasattr(model, 'parameters') and list(model.parameters()): device = next(model.parameters()).device
    elif hasattr(model, 'buffers') and list(model.buffers()): device = next(model.buffers()).device
        
    real_traj_input_t = torch.as_tensor(real_traj_input, dtype=torch.float32, device=device)
    
    # ---> FIX: Normalize physical state before expansion <---
    if hasattr(model, "_normalize_x"):
        real_traj_input_scaled = model._normalize_x(real_traj_input_t)
    else:
        real_traj_input_scaled = real_traj_input_t
        
    expanded_traj_flattened = safe_expand(model, real_traj_input_scaled).cpu().numpy()
    expanded_traj = expanded_traj_flattened.reshape(n_steps_valid, n_trajs, -1)
    
    # --- Normalize initial conditions so W projection is accurate ---
    expanded_init_conditions = torch.as_tensor(expanded_traj[0, :, :], dtype=torch.float32)
    if hasattr(model, "_normalize"):
        expanded_init_conditions = model._normalize(expanded_init_conditions)
    elif hasattr(model, "psi_scale"):
        expanded_init_conditions = expanded_init_conditions / model.psi_scale.to(expanded_init_conditions.device)
    expanded_init_conditions = expanded_init_conditions.cpu().numpy()
    
    if W is not None:
        W_mat = W
    else:
        W_mat = np.linalg.pinv(Phi).T

    z0 = expanded_init_conditions @ W_mat
    mode_evolution = np.zeros((n_steps_valid, n_trajs, n_modes), dtype=complex)
    mode_evolution[0, :, :] = z0

    # Step forward in modal space
    for t in range(1, n_steps_valid):
        mode_evolution[t, :, :] = mode_evolution[t-1, :, :] @ Lambda.T
        
    mode_mses = np.zeros(n_modes)
    
    # Evaluate physical reconstruction error for each mode individually
    device = "cpu"
    if hasattr(model, 'parameters') and list(model.parameters()):
        device = next(model.parameters()).device
    elif hasattr(model, 'buffers') and list(model.buffers()):
        device = next(model.buffers()).device
        
    for i in range(n_modes):
        # 1. Isolate the modal trajectory for mode i: shape (T, N_traj)
        b_i = mode_evolution[:, :, i] 
        
        # 2. Map back to latent space: z_i(t) = b_i(t) * Phi[:, i]
        phi_i = Phi[:, i] 
        z_i = np.einsum('tn,d->tnd', b_i, phi_i) 
        
        if np.iscomplexobj(z_i):
            # FIX: Multiply genuinely complex modes by 2 to account for the conjugate partner
            if np.abs(np.imag(Lambda[i, i])) > 1e-6:
                z_i = 2 * z_i.real
            else:
                z_i = z_i.real
            
        # 3. Unnormalize the latent features
        z_i_tensor = torch.as_tensor(z_i, dtype=torch.float32, device=device)
        if hasattr(model, "_unnormalize"):
            z_i_unnorm = model._unnormalize(z_i_tensor)
        elif hasattr(model, "psi_scale"):
            # FIX: Regression DMD's C_fitted natively expects normalized inputs. 
            # Do NOT multiply by psi_scale!
            z_i_unnorm = z_i_tensor 
        else:
            z_i_unnorm = z_i_tensor
            
        # 4. Map back to physical space correctly depending on the model type
        T_steps, N_traj, D_latent = z_i_unnorm.shape
        
        if hasattr(model, 'C_fitted'):
            # Regression_DMD: uses a learned matrix C_fitted to decode, then denormalizes
            C_mat = model.C_fitted.detach().to(device) if hasattr(model.C_fitted, 'detach') else torch.as_tensor(model.C_fitted, device=device)
            x_i_pred_t = torch.matmul(z_i_unnorm.to(C_mat.dtype), C_mat.T).to(torch.float32)
            
            if hasattr(model, "_denormalize_x"):
                x_i_pred_t_flat = model._denormalize_x(x_i_pred_t.reshape(-1, C_mat.shape[0]))
                x_i_pred_t = x_i_pred_t_flat.reshape(T_steps, N_traj, -1)
        else:
            # ML_DMD: requires de_expand to slice the physical state from the FULL latent vector
            z_i_flat = z_i_unnorm.reshape(-1, D_latent)
            x_i_pred_t_flat = safe_de_expand(model, z_i_flat)
            x_i_pred_t = x_i_pred_t_flat.reshape(T_steps, N_traj, -1)
            
        x_i_pred = x_i_pred_t.detach().cpu().numpy()
        
        # ---> FIX: Strip delay history from Regression DMD prediction <---
        x_i_pred = x_i_pred[:, :, :state_dim]
        
        # 5. MSE against the true physical trajectory
        error = np.mean((x_i_pred - real_traj_valid)**2)
        mode_mses[i] = error

    # Sort lowest error to highest
    ranked_indices = np.argsort(mode_mses)[::1]
    
    # ---> THIS RETURN WAS MISSING <---
    return ranked_indices, mode_mses

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


def truncated_rollout(
    model, real_traj, n_modes=2, save_path=None, Phi=None, Lambda=None, W=None,
    mode_indices=None, subtitle=None, save_name=None, plot=True
):
    n_trajs = real_traj.shape[1]
    state_dim = real_traj.shape[2]
    
    delay_depth = int(getattr(model.expander, "delay_depth", 1)) if hasattr(model, "expander") else 1
    
    if delay_depth > 1:
        q = delay_depth
        window = real_traj[0:q, :, :]
        window = window[::-1, :, :]
        x0_hist = window.transpose(1, 0, 2).reshape(n_trajs, -1)
        plot_real_traj = real_traj[q-1:, :, :]
    else:
        x0_hist = real_traj[0, :, :]
        plot_real_traj = real_traj
        
    n_steps_valid = plot_real_traj.shape[0]
    
    device = "cpu"
    if hasattr(model, 'parameters') and list(model.parameters()): device = next(model.parameters()).device
    elif hasattr(model, 'buffers') and list(model.buffers()): device = next(model.buffers()).device
    
    # ---------------------------------------------------------
    # 1. Project Initial State into Sorted Modal Space
    # ---------------------------------------------------------
    x0_t = torch.as_tensor(x0_hist, dtype=torch.float32, device=device)
    
    # Normalize physical state
    if hasattr(model, "_normalize_x"):
        x0_n = model._normalize_x(x0_t)
    else:
        x0_n = x0_t
        
    # Expand to latent space
    with torch.no_grad():
        if hasattr(model, "expander"):
            z0 = model.expander.expand(x0_n)
        else:
            z0 = model.expand(x0_n)
            
        # Normalize latent state
        if hasattr(model, "_normalize"):
            z0_norm = model._normalize(z0)
        elif hasattr(model, "psi_scale"):
            z0_norm = z0 / model.psi_scale.to(device)
        else:
            z0_norm = z0
            
    z0_norm_np = z0_norm.detach().cpu().numpy()
    
    # Project using the PASSED-IN, sorted W matrix!
    b_t = z0_norm_np @ W 
    
    # ---------------------------------------------------------
    # 2. Apply Truncation Mask to Sorted Modes
    # ---------------------------------------------------------
    latent_dim = Phi.shape[1]
    mask = np.zeros(latent_dim, dtype=complex)
    
    if mode_indices is None:
        mode_indices = list(range(min(n_modes, latent_dim)))
    
    # We no longer need _get_expanded_indices because Phi/Lambda/W are already 
    # perfectly formatted as diagonal complex pairs by the main script!
    mask[mode_indices] = 1.0
    b_t = b_t * mask

    # ---------------------------------------------------------
    # 3. Evolve in Pure Modal Space
    # ---------------------------------------------------------
    Lambda_mat = np.diag(Lambda) if Lambda.ndim == 1 else Lambda
    mode_evolution = np.zeros((n_steps_valid, n_trajs, latent_dim), dtype=complex)
    
    for t in range(n_steps_valid):
        mode_evolution[t, :, :] = b_t
        b_t = (b_t @ Lambda_mat.T) * mask
        
    # ---------------------------------------------------------
    # 4. Map Back to Physical Space
    # ---------------------------------------------------------
    # Back to normalized latent space
    z_t_norm = (mode_evolution @ Phi.T).real
    
    truncated_trajectory = np.zeros_like(plot_real_traj)
    for t in range(n_steps_valid):
        with torch.no_grad():
            z_norm_tensor = torch.as_tensor(z_t_norm[t], dtype=torch.float32, device=device)
            
            # Unnormalize latent
            if hasattr(model, "_unnormalize"):
                # ML DMD requires un-scaling before de-expanding
                z_unnorm = model._unnormalize(z_norm_tensor)
            elif hasattr(model, "psi_scale"):
                # FIX: Regression DMD's C_fitted natively expects normalized inputs. 
                # Do NOT multiply by psi_scale!
                z_unnorm = z_norm_tensor 
            else:
                z_unnorm = z_norm_tensor
                
            # Decode to physical state
            if hasattr(model, "C_fitted"):
                # Regression DMD path
                C_mat = model.C_fitted.detach().to(device) if hasattr(model.C_fitted, 'detach') else torch.as_tensor(model.C_fitted, device=device)
                x_pred_n = torch.matmul(z_unnorm.to(C_mat.dtype), C_mat.T).to(torch.float32)
            else:
                # ML DMD path
                x_pred_n = model.expander.de_expand(z_unnorm)
                
            # Unnormalize physical state
            if hasattr(model, "_denormalize_x"):
                x_pred = model._denormalize_x(x_pred_n)
            else:
                x_pred = x_pred_n
                
            truncated_trajectory[t, :, :] = x_pred.cpu().numpy()[:, :state_dim]

    # ---------------------------------------------------------
    # Evaluation (Best/Median/Worst)
    # ---------------------------------------------------------
    mse_all = np.mean((truncated_trajectory - plot_real_traj[:, :, :state_dim]) ** 2, axis=(0, 2))
    rmse_all = np.sqrt(mse_all)
    
    results = [(rmse_all[i], i) for i in range(n_trajs)]
    results.sort(key=lambda x: x[0])
    
    best_idx = results[0][1]
    median_idx = results[len(results) // 2][1]
    worst_idx = results[-1][1]
    
    selected_cases = [
        ("Best Case", best_idx),
        ("Median Case", median_idx),
        ("Worst Case", worst_idx)
    ]
    
    if plot:
        # ---------------------------------------------------------
        # Plotting
        # ---------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for ax, (case_name, idx) in zip(axes, selected_cases):
            rmse = rmse_all[idx]
            
            ax.plot(plot_real_traj[:, idx, 0], plot_real_traj[:, idx, 1], 'k-', label='Ground Truth', alpha=0.7)
            ax.plot(truncated_trajectory[:, idx, 0], truncated_trajectory[:, idx, 1], 'b--', label=f'Mode rollout (n={len(mode_indices)})')
            
            ax.scatter(plot_real_traj[0, idx, 0], plot_real_traj[0, idx, 1], color='black', marker='o', s=40, label='Ground Truth Start', zorder=10)
            ax.scatter(truncated_trajectory[0, idx, 0], truncated_trajectory[0, idx, 1], color='blue', marker='x', s=50, label='Model Start', zorder=10)
            
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_title(f"{case_name}\nRMSE: {rmse:.2e}", fontsize=12)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(loc="best", fontsize=8, frameon=True)

        avg_rmse = np.mean(rmse_all)
        display_subtitle = f"{subtitle}\nAvg Test RMSE: {avg_rmse:.2e}"
        fig.suptitle(_with_subtitle(f"State Space Rollout Comparison (Top {len(mode_indices)} Koopman Modes)", display_subtitle), fontsize=14, y=0.985)
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        
        if save_path:
            filename = save_name or f"truncated_rollout_{len(mode_indices)}_modes.png"
            plt.savefig(os.path.join(save_path, filename), bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    return truncated_trajectory


def plot_rmse_contribution(mode_counts, rmses, contributions, save_path=None, subtitle=None):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot RMSE (Left Axis)
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Modes')
    ax1.set_ylabel('RMSE', color=color1, fontsize=12)
    ax1.plot(mode_counts, rmses, color=color1, marker='o', linestyle='-', linewidth=2, label='RMSE')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot Contribution (Right Axis)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Contribution (%)', color=color2, fontsize=12)
    ax2.plot(mode_counts, contributions, color=color2, marker='s', linestyle='--', linewidth=2, label='Contribution')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 105) # Contribution is a percentage

    plt.title(_with_subtitle("Model Performance vs. Mode Truncation", subtitle), fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_eigenvalue_spectrum(eigvals, mode_scores, score_metric, save_path=None, subtitle=None):
    fig, ax = plt.subplots(figsize=(6,4))
    
    circle_color = "#9ca3af"
    
    scatter = ax.scatter(
        eigvals.real, 
        eigvals.imag, 
        c=mode_scores, 
        cmap='viridis', 
        s=40 + (mode_scores * 60), 
        edgecolors='white',
        linewidths=0.5,
        alpha=0.9,
        zorder=3
    )
    
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "--", color=circle_color, linewidth=1.2, zorder=1)

    x_bounds = (
        eigvals.real.min() - max(0.1*abs(eigvals.real.min()), 0.02),
        eigvals.real.max() + max(0.1*abs(eigvals.real.max()), 0.02)
    )
    y_bounds = (
        eigvals.imag.min() - max(0.1*abs(eigvals.imag.min()), 0.02),
        eigvals.imag.max() + max(0.1*abs(eigvals.imag.max()), 0.02)
    )
    ax.set_xlim(x_bounds)
    _set_nonzero_ylim(ax, y_bounds[0], y_bounds[1])
    
    ax.axhline(0, color=circle_color, linewidth=0.8, alpha=0.5)
    ax.axvline(0, color=circle_color, linewidth=0.8, alpha=0.5)
    # ax.set_aspect("equal", adjustable="box")
    ax.set_title(_with_subtitle(f"Eigenvalue Spectrum", subtitle), fontsize=10)
    ax.set_xlabel("$\mathbb{R}(\lambda)$")
    ax.set_ylabel("$\mathbb{I}(\lambda)$")

    # Add legend for unit circle line and dot for eigenvalues
    ax.plot([], [], "--", color=circle_color, linewidth=1.2, label="Unit Circle")
    ax.scatter([], [], color='black', s=20, label=f"Eigenvalues")
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'Mode {score_metric}', rotation=270, labelpad=15)
    
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_freq_magnitude(eigvals, mode_scores, score_metric, save_path=None, subtitle=None):
    magnitudes = np.abs(eigvals)
    frequencies = np.abs(np.angle(eigvals)) / np.pi 
    
    fig, ax = plt.subplots(figsize=(8, 5))
    circle_color = "#9ca3af"
    
    scatter = ax.scatter(
        frequencies, 
        magnitudes, 
        c=mode_scores, 
        cmap='viridis', 
        s=50, 
        alpha=0.8,
        edgecolors='white',
        linewidths=0.5
    )

    # Group only nearly identical points together so isolated points still get labels.
    rounded_positions = np.column_stack((np.round(frequencies, 3), np.round(magnitudes, 3)))
    plot_checklist = np.zeros(len(eigvals), dtype=bool)  # keep track of annotated points
    y_offset = 0.012 * max(magnitudes.max() - magnitudes.min(), 1e-3)
    x_offset = 0.012 * max(frequencies.max() - frequencies.min(), 1e-3)

    for i, (freq, mag) in enumerate(zip(frequencies, magnitudes)):
        if plot_checklist[i]:
            continue

        same_spot = np.where(
            (rounded_positions[:, 0] == rounded_positions[i, 0])
            & (rounded_positions[:, 1] == rounded_positions[i, 1])
        )[0]
        same_spot = same_spot[~plot_checklist[same_spot]]

        if len(same_spot) == 0:
            continue

        label_string = ", ".join(str(idx) for idx in same_spot)
        plot_checklist[same_spot] = True

        x_coord = freq + x_offset
        y_coord = mag - y_offset
        ax.text(
            x_coord,
            y_coord,
            label_string,
            fontsize=8,
            ha='left',
            va='center',
            color='black',
            fontweight='bold',
        )
    
    ax.axhline(1.0, color=circle_color, linestyle='--', alpha=0.6, label="Unit Circle (Stable)")
    
    ax.set_title(_with_subtitle("Eigenvalue Distribution: Frequency vs. Magnitude", subtitle), fontsize=14)
    ax.set_xlabel("Normalized Frequency ($\omega / \pi$)")
    ax.set_ylabel("Magnitude ($|\lambda|$)")
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'Mode {score_metric}')
    
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_mode_contributions_vs_quality(V, phi_traj, best_ids, scores, state_dim, n_top=10, save_path=None, subtitle=None):
    amplitudes = np.mean(np.abs(phi_traj), axis=0)
    v_norms = np.linalg.norm(V[:state_dim, :], axis=0)
    
    mode_energies = amplitudes * v_norms
    total_energy = np.sum(mode_energies)
    relative_contribution = (mode_energies / (total_energy + 1e-12)) * 100
    
    energy_sort_idx = np.argsort(relative_contribution)[::-1]
    n_show = min(n_top, len(energy_sort_idx))
    top_energy_idx = energy_sort_idx[:n_show]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(scores[top_energy_idx])
    
    bars = ax.bar(range(n_show), relative_contribution[top_energy_idx], color=colors)
    
    ax.set_xticks(range(n_show))
    ax.set_xticklabels([f"Mode {i}" for i in top_energy_idx], rotation=45)
    ax.set_ylabel("Contribution to Reconstruction (%)")
    ax.set_title(_with_subtitle("Physical Mode Energy (Weighted by Average Activation)", subtitle))
    
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.2, 
                f"Q:{scores[top_energy_idx[i]]:.2f}", ha='center', va='bottom', fontsize=8)

    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label="Quality Score")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_mode_energy_vs_quality(V, scores, state_dim, n_top=20, save_path=None, subtitle=None):
    mode_energies = np.linalg.norm(V[:state_dim, :], axis=0)
    energy_norm = (mode_energies - mode_energies.min()) / (mode_energies.max() - mode_energies.min() + 1e-12)

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(
        scores, 
        energy_norm, 
        s=100, 
        alpha=0.7, 
        edgecolors='black',
        zorder=3
    )

    top_indices = np.argsort(mode_energies)[::-1][:n_top]
    plot_checklist = np.zeros(len(scores), dtype=bool)
    for idx in top_indices:
        if plot_checklist[idx]:
            continue
        nearby_indices = np.where(np.abs(scores - scores[idx]) < 0.001)[0]
        label_string = f"Mode {idx}"
        for near_idx in nearby_indices:
            if near_idx != idx and not plot_checklist[near_idx]:
                label_string += f", {near_idx}"
                plot_checklist[near_idx] = True
            
        ax.text(scores[idx], energy_norm[idx]+0.02, label_string, fontsize=8, ha='center', va='center', zorder=4)
        plot_checklist[idx] = True

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.3)

    ax.text(0.75, 0.9, "Governing Modes", fontsize=12, fontweight='bold', alpha=0.5, ha='center')
    ax.text(0.75, 0.1, "Math Harmonics", fontsize=12, alpha=0.5, ha='center')
    ax.text(0.25, 0.9, "Overfitted Noise", fontsize=12, alpha=0.5, ha='center')
    ax.text(0.25, 0.1, "Junk Modes", fontsize=12, alpha=0.5, ha='center')

    ax.set_xlabel("Quality Score (Linearity & Stability)", fontsize=12)
    ax.set_ylabel("Normalized Energy (Contribution to Physical States)", fontsize=12)
    ax.set_title(_with_subtitle("Mode Selection: Quality vs. Physical Energy", subtitle), fontsize=14)
    
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()