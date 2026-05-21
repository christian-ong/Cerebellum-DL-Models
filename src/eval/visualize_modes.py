import os
import torch
import numpy as np
import sympy
from scipy.linalg import expm, schur
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from src.models.deprecated.ml_dmd_free import ML_DMD
from src.models.ml_linear_dynamics import ML_LinearDynamics
from src.models.regression_dmd import Regression_DMD



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
            rollout_mode=str(get_scalar("rollout_mode")),
            ridge=float(get_scalar("ridge")),
            rank=None if get_scalar("rank") == -1 else int(get_scalar("rank")),
            rbf_n_centers=int(get_scalar("rbf_n_centers")),
            rbf_center_selection=str(get_scalar("rbf_center_selection")),
            rbf_bandwidth_mode=str(get_scalar("rbf_bandwidth_mode")),
            rbf_knn_k=int(get_scalar("rbf_knn_k")),
        ).to(device)

        # 2. Re-hydrate arrays back to model attributes
        matrix_mappings = {
            "x_mean": "x_mean", "x_scale": "x_scale", "psi_scale": "psi_scale",
            "K_fitted": "K", "C_fitted": "C", "K_tilde_fitted": "K_tilde",
            "U_r_fitted": "U_r", "W_reduced_fitted": "W_reduced",
            "Lambda_fitted": "Lambda", "Phi_lift_fitted": "Phi_lift", "Phi_state_fitted": "Phi_state"
        }
        
        for attr, np_key in matrix_mappings.items():
            if np_key in ckpt:
                tensor_val = torch.as_tensor(ckpt[np_key], dtype=torch.complex64, device=device)
                setattr(model, attr, tensor_val)

        model.is_fitted = True
        return model, model_name

    # ---------------------------------------------------------
    # ML models
    # ---------------------------------------------------------

    ckpt = torch.load(model_path, map_location="cpu")
    model_name = ckpt.get("model", "ml_dmd_free")
    train_args = ckpt["train_args"]
    
    kwargs = {
        "state_dim": ckpt["state_dim"],
        "expansion_degree": train_args["expansion_degree"],
        "bias": str(train_args.get("bias", "true")).lower() == "true",
        "sine_cosine_expansion": str(train_args.get("sine_cosine_expansion", "false")).lower() == "true",
        "expansion_type": train_args["expansion_type"],
        "system": ckpt["system"],
    }

    if model_name == "ml_dmd" or model_name == "hardcoded_dmd":
        model = ML_DMD(**kwargs)
    elif model_name == "ml_lineardynamics":
        model = ML_LinearDynamics(**kwargs)
    else:
        raise ValueError(f"Unsupported: {model_name}")

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        model.load_state_dict(state_dict)

    model.eval()
    return model, model_name


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

        return Phi_true, Lambda, V, W, K_true

    # ------------------------------------------------------------------
    # Case 2: Old scaled models that expose get_Phi_true and get_Lambda
    # ------------------------------------------------------------------
    elif hasattr(model, "get_Phi_true") and hasattr(model, "get_Lambda"):
        Phi_true_obj = model.get_Phi_true()
        Lambda_obj = model.get_Lambda()
        K_obj = model.get_K_true() if hasattr(model, "get_K_true") else None

        Phi_true = (
            Phi_true_obj.detach().cpu().numpy()
            if hasattr(Phi_true_obj, "detach")
            else np.array(Phi_true_obj)
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

        return Phi_true, Lambda, eigvals, V, W, K_true

    # Case 2: Models that expose only the Koopman operator K (e.g., ML_LinearDynamics)
    elif hasattr(model, "get_K_true"):
        K_obj = model.get_K_true()
        K_true = K_obj.detach().cpu().numpy() if hasattr(K_obj, "detach") else np.array(K_obj)

        eigvals, V = np.linalg.eig(K_true)
        _, W = np.linalg.eig(K_true.T)

        # Without an explicit Phi mapping, treat Phi_true as identity in lifted space
        Phi_true = np.eye(K_true.shape[0])
        Lambda = np.diag(eigvals)

        v_norms = np.linalg.norm(V, axis=0)
        V = V / (v_norms + 1e-12)
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

        return Phi_true, Lambda, eigvals, V, W, K_true

    # Regression DMD
    elif hasattr(model, "Phi_lift_fitted") and hasattr(model, "Lambda_fitted"): 
        Phi_true = model.Phi_lift_fitted.detach().cpu().numpy() if hasattr(model.Phi_lift_fitted, "detach") else np.array(model.Phi_lift_fitted)
        Lambda = model.Lambda_fitted.detach().cpu().numpy() if hasattr(model.Lambda_fitted, "detach") else np.array(model.Lambda_fitted)
        K_true = model.K_fitted.detach().cpu().numpy() if hasattr(model.K_fitted, "detach") else np.array(model.K_fitted)

        if Lambda.ndim == 1:
            Lambda = np.diag(Lambda)

        # Transpose to match convention
        Lambda = Lambda.T
        Phi_true = np.linalg.pinv(Phi_true).T

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

        return Phi_true, Lambda, V, W, K_true

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
        threshold_off_diag=1e-3, 
        threshold_diag=1e-3,
        print_info=False):
    """
    Detects pairs of complex conjugate modes in the Lambda matrix.
    Looks for significant (> threshold) off-diagonal values in 2x2 blocks.
    In case of overlaps, keeps highest scoring blocks (sum of off-diagonal elements).
    """
    
    if np.iscomplexobj(Lambda):

        block_indices = []
        block_scores = []

        for i in range(Lambda.shape[0]):
            if (np.iscomplex(Lambda[i, i]) and 
                i < Lambda.shape[0] - 1):
                
                if (Lambda[i,i].real - Lambda[i+1,i+1].real < threshold_diag and
                    Lambda[i,i].imag - (-Lambda[i+1,i+1].imag) < threshold_diag):

                    block_indices.append((i, i+1))
                    score = (Lambda[i,i].real - Lambda[i+1,i+1].real + 
                             Lambda[i,i].imag - (-Lambda[i+1,i+1].imag))
                    block_scores.append(score)

    elif np.isrealobj(Lambda):

        block_indices = []
        block_scores = []

        # Loop through 2x2 diagonal blocks
        for diag_idx in range(len(Lambda)-1):

            # Get 2x2 block <a,b; c,d>
            a = Lambda[diag_idx, diag_idx]
            b = Lambda[diag_idx, diag_idx+1]
            c = Lambda[diag_idx+1, diag_idx]
            d = Lambda[diag_idx+1, diag_idx+1]

            # Detect rotation blocks (significant off-diagonal values)
            if ((abs(b) > threshold_off_diag) and # off-diag val 1 significant
                (abs(c) > threshold_off_diag) and # off-diag val 2 significant
                (np.sign(b) != np.sign(c)) and # off-diag vals have opposite signs, indicating rotation
                (abs(a - d) < threshold_diag)): # diag vals similar 

                block_indices.append((diag_idx, diag_idx+1))

                # Give a significance score
                score_off_diag = (abs(b) + abs(c)) + abs(b - (-c)) # higher for more significant off-diagonal values and values are (conjugate) similar
                score_diag = abs(a - d) # higher if a and d are closer
                block_scores.append(score_off_diag + score_diag)

    # Sort blocks by score
    sorted_scores_idx = np.argsort(block_scores)[::-1] # descending order
    sorted_block_scores = [block_scores[i] for i in sorted_scores_idx]
    sorted_block_indices = [block_indices[i] for i in sorted_scores_idx]

    # Remove overlapping blocks (keep only the highest scoring)
    final_blocks_idx = []
    final_idxs = []
    for idx, score in zip(sorted_block_indices, sorted_block_scores):
        if not any(i in final_idxs for i in idx): # if not already included, include this block
            final_blocks_idx.append(idx)
            final_idxs.extend(idx)

    return final_blocks_idx
    

def rotation_blocks_to_complex(Lambda, Phi, complex_pair_idx):
    """
    Converts detected 2x2 rotation blocks in Lambda and corresponding columns in Phi into complex conjugate pairs.
    """

    Lambda_complex = Lambda.copy().astype(np.complex128)
    Phi_complex = Phi.copy().astype(np.complex128)

    # Loop through complex blocks
    for idx in complex_pair_idx:
        i, j = idx

        # Get 2x2 block <a,b; c,d>
        a = Lambda[i, i]
        c = Lambda[i, j] 
        b = Lambda[j, i] 
        d = Lambda[j, j]

        # find complex value
        complex_val_1 = a + 1j*b
        complex_val_2 = d + 1j*c

        # In Lambda: replace block with complex values on diagonal
        Lambda_complex[i, i] = complex_val_1 
        Lambda_complex[j, j] = complex_val_2 # should be conjugate of complex_val_1, but allow assymetric imperfections
        Lambda_complex[i, j] = 0
        Lambda_complex[j, i] = 0

        # In Phi: v1 = re(mode), v2 = im(mode)
        real_part = Phi[:, i]
        imag_part = Phi[:, j]
        Phi_complex[:, i] = real_part + 1j*imag_part
        Phi_complex[:, j] = real_part - 1j*imag_part

    return Lambda_complex, Phi_complex


def plot_transition_matrices(matrices, title, model_expansion_names, analytic_expansion_names, threshold_include_val = 1e-3, save_path=None):
    fig = plt.figure(figsize=(18, 10))
    # Create a 2x3 grid
    num_rows = int(np.ceil(len(matrices) / 3))
    gs = gridspec.GridSpec(num_rows, 3, width_ratios=[1, 1, 1])
    axes = [fig.add_subplot(gs[i,j]) for i in range(num_rows) for j in range(3)]

    fig.suptitle(title, fontsize=22)
    
    for i, (M, subtitle) in enumerate(matrices):
        ax = axes[i]
        M_mag = np.abs(M)
        im = ax.imshow(M_mag, vmin=0) # color by magnitude, but show values as real/imaginary parts
        ax.set_title(subtitle, fontsize=14)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Model or analytic expansion names (for x/y ticks)
        expansion_names = model_expansion_names if "model" in subtitle.lower() else analytic_expansion_names

        # --- Corrected X Ticks ---
        if "K" in subtitle:
            # K maps basis functions to basis functions (Square Matrix)
            ax.set_xticks(range(len(expansion_names)))
            ax.set_xticklabels(expansion_names, rotation=60, fontsize=9)
        else:
            # Remove x-labels for Phi (modes) and Lambda (modes)
            ax.set_xticks([])

        # --- Corrected Y Ticks ---
        if "K" in subtitle or "Phi" in subtitle:
            # Both K and Phi rows index the basis functions (Observables)
            ax.set_yticks(range(len(expansion_names)))
            ax.set_yticklabels(expansion_names, fontsize=9)
        else:
            # Remove y-labels for Lambda (modes)
            ax.set_yticks([])

        # Dynamically scale font size based on matrix dimension
        n_cols = M.shape[1]
        f_size = 10 if n_cols <= 5 else (8 if n_cols <= 8 else 6)

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

    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.1, top=0.92, hspace=0.4, wspace=0.2)
        
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


def get_real_representation(V, eigvals, jordan_value=1.0, threshold_imag=1e-5, threshold_jordan=1):
    """
    Converts complex Koopman modes and eigenvalues into their 
    real-valued block-diagonal form.
    """
    
    # Initialize real-valued V and Lambda
    V_real = np.copy(np.real(V).astype(np.float64))
    Lambda_real = np.copy(np.real(eigvals).astype(np.float64))

    i = 0
    while i < len(eigvals):
        if abs(np.imag(eigvals[i,i])) < threshold_imag: # real eigenvalue
            V_real[:, i] = np.real(V[:, i])
            Lambda_real[i, i] = np.real(eigvals[i, i])
            
            if jordan_value != 0:
                # Check for Jordan block
                if (i + 2 <= len(eigvals) and (
                    abs(eigvals[i+1,i]) > threshold_jordan or
                    abs(eigvals[i,i+1]) > threshold_jordan
                )): 
                    Lambda_real[i, i+1] = jordan_value # Jordan block off-diagonal
        
            i += 1

        else: # complex conjugate pair
            if i + 1 < len(eigvals):
                V_real[:, i] = np.real(V[:, i]) # Re(v)
                V_real[:, i+1] = np.imag(V[:, i]) # Im(v)
                
                a = np.real(eigvals[i,i])
                b = np.imag(eigvals[i,i])
                c = np.imag(eigvals[i+1,i+1])
                d = np.real(eigvals[i+1,i+1])
                
                # <a, -b; b, a> since Lambda is transposed
                Lambda_real[i, i] = a
                Lambda_real[i, i+1] = c
                Lambda_real[i+1, i] = b
                Lambda_real[i+1, i+1] = d
                i += 2
            else:
                V_real[:, i] = np.real(V[:, i])
                Lambda_real[i, i] = np.real(eigvals[i, i])
                i += 1

    return V_real, Lambda_real


def plot_koopman_mode_rollout(model, Phi, Lambda, real_traj, save_path=None, model_type="regression_dmd"):
    """
    Plot data projected onto one mode at a time to compare real vs. model evolution.
    """
    # Measure dimensions
    n_steps = real_traj.shape[0]
    n_trajs = real_traj.shape[1]
    state_dim = real_traj.shape[2]
    n_modes = Phi.shape[1]
    lifted_dim = Phi.shape[0]
    
    # Lift real trajectory
    real_traj_flattened = real_traj.reshape(-1, state_dim) # (n_steps * n_trajs, state_dim)
    expanded_traj_flattened = safe_expand(model, torch.tensor(real_traj_flattened, dtype=torch.float32)).cpu().numpy() # (n_steps * n_trajs, lifted_dim)
    expanded_traj = expanded_traj_flattened.reshape(n_steps, n_trajs, lifted_dim)
    
    # Grab initial conditions
    init_conditions = torch.as_tensor(real_traj[0, :, :], dtype=torch.float32) # (n_trajs, state_dim)
    expanded_init_conditions = expanded_traj[0, :, :] 

    ### 1. Real trajectory projected onto modes ###
    real_proj = expanded_traj @ Phi
    real_proj = real_proj.real

    ### 2. Model rollout trajectory ###
    model_rollouts = model.rollout(init_conditions, steps=n_steps-1).detach().numpy() # (n_steps, n_trajs, state_dim)
    model_rollouts_flattened_real = model_rollouts.reshape(-1, state_dim).real if np.iscomplexobj(model_rollouts) else model_rollouts.reshape(-1, state_dim) # (n_steps * n_trajs, state_dim)
    model_rollouts_flattened_expanded = safe_expand(model, torch.as_tensor(model_rollouts_flattened_real, dtype=torch.float32)).detach().numpy()
    model_rollouts_expanded = model_rollouts_flattened_expanded.reshape(n_steps, n_trajs, lifted_dim) # (n_steps, n_trajs, lifted_dim)
    model_proj = model_rollouts_expanded @ Phi
    model_proj = model_proj.real

    ### 3. Mode evolution under Lambda ###
    # Initialize mode amplitudes from initial conditions
    z0 = expanded_init_conditions @ Phi # Initial mode amplitudes
    mode_evolution = np.zeros((n_steps, n_trajs, n_modes), dtype=complex)
    mode_evolution[0, :, :] = z0

    # Evolve with Lambda
    for t in range(1, n_steps):
        z = mode_evolution[t-1, :, :]
        if "regression" in model_type: # For regression models, we can directly apply Lambda
            z_next = z @ Lambda.T
        else:
            z_next = z @ Lambda.T
        mode_evolution[t, :, :] = z_next
    mode_evolution = mode_evolution.real

    ### 4. Plotting ###
    fig, axes = plt.subplots(n_trajs, n_modes, figsize=(2.5 * n_modes, 10), sharex=True)
    if n_modes == 1: axes = [axes]

    time = np.arange(n_steps)

    for i in range(n_modes): # columns
        for traj_idx in range(n_trajs): # rows
        
            # Label columns
            if i == 0:
                axes[traj_idx,i].set_ylabel(f"Trajectory {traj_idx + 1}")

            # Label rows
            if traj_idx == 0:
                axes[traj_idx,i].set_title(f"Mode {i + 1}", fontsize=10)

            # Plot real projection, model rollout, and mode evolution
            axes[traj_idx,i].plot(time, real_proj[:, traj_idx, i], 'k-', label='Real (Proj)', alpha=0.6)
            axes[traj_idx,i].plot(time, model_proj[:, traj_idx, i], 'r--', label='Model Rollout')
            axes[traj_idx,i].plot(time, mode_evolution[:, traj_idx, i], 'b:', label='$\Lambda$ Evolution')

            # y-lim : keep close to real projection
            y_min_real = real_proj[:, traj_idx, i].min()
            y_max_real = real_proj[:, traj_idx, i].max()
            range_real = y_max_real - y_min_real
            y_lower_bound_real = y_min_real - range_real * 0.1
            y_upper_bound_real = y_max_real + range_real * 0.1
            y_range_real = y_upper_bound_real - y_lower_bound_real

            y_min_other = min(model_proj[:, traj_idx, i].min(), mode_evolution[:, traj_idx, i].min())
            y_max_other = max(model_proj[:, traj_idx, i].max(), mode_evolution[:, traj_idx, i].max())
            range_other = y_max_other - y_min_other
            y_lower_bound_other = y_min_other - range_other * 0.1
            y_upper_bound_other = y_max_other + range_other * 0.1
            y_range_other = y_upper_bound_other - y_lower_bound_other
            
            if y_range_other > 10 * y_range_real:
                axes[traj_idx,i].set_ylim([y_lower_bound_real - y_range_real * 2, y_upper_bound_real + y_range_real * 2])
            else:
                y_lower_bound_final = min(y_lower_bound_real, y_lower_bound_other)
                y_upper_bound_final = max(y_upper_bound_real, y_upper_bound_other)
                axes[traj_idx,i].set_ylim([y_lower_bound_final, y_upper_bound_final])

            # Legend info
            if i == 0 and traj_idx == 0:
                # Grab handles and labels from this specific subplot
                handles, labels = axes[traj_idx, i].get_legend_handles_labels()
            
            # Final subplot gets x-label
            if traj_idx == n_trajs - 1:
                axes[traj_idx, i].set_xlabel("Time Steps")
        
    fig.suptitle("Koopman Mode Evolution (Real Part)", fontsize=22, y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # move plots down to avoid overlap
    fig.legend(handles, labels, loc='upper left', ncol=3, fontsize=14, frameon=True, bbox_to_anchor=(0.01, 0.99))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_eigenfunctions(
        grid_points, 
        grid_points_expanded, 
        Phi,
        scores,
        score_metric,
        complex_pair_idx,
        cmap="inferno", 
        save_path=None
    ):
    # Compute eigenfunction values on the grid for the top N modes
    eigenfunction_vals = grid_points_expanded @ Phi

    # Compute complex pairs for annotation
    complex_pairs = [sorted([int(i+1), int(j+1)]) for i, j in complex_pair_idx] # 1-based index and internal sort

    grid_n = int(np.sqrt(len(grid_points)))
    extent = [grid_points[:,0].min(), grid_points[:,0].max(), grid_points[:,1].min(), grid_points[:,1].max()]
    num_modes = eigenfunction_vals.shape[1]

    fig, axes = plt.subplots(4, num_modes, figsize=(3+ 2.5*num_modes, 10))
        
    for mode_idx in range(num_modes):
        data_map = {
            "Real": np.real(eigenfunction_vals[:, mode_idx]), 
            "Imag": np.imag(eigenfunction_vals[:, mode_idx]),
            "Mag": np.abs(eigenfunction_vals[:, mode_idx]), 
            "Phase": np.angle(eigenfunction_vals[:, mode_idx])
        }
    
        # Annotate complex pairs
        pair_string = ""
        for n_pair, pair in enumerate(complex_pairs):
            if mode_idx+1 in pair: # if this mode is part of a complex pair
                pair_string += f" {'*' * (n_pair+1)}\n"

        for i, (label, data) in enumerate(data_map.items()):
            if i==0:
                label = (
                    pair_string + 
                    rf"$\mathbf{{EF{mode_idx+1}}}$" + 
                    f"\n{score_metric}: {scores[mode_idx]:.3f}" +
                    f"\n\n{label}")
            ax = axes[i, mode_idx]
            im = ax.imshow(data.reshape(grid_n, grid_n), extent=extent, origin="lower", cmap=cmap, aspect='auto')
            ax.set_title(f"{label}")
            plt.colorbar(im, ax=ax)

    fig.suptitle(f"{num_modes} first eigenfunctions", fontsize=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def modes_by_quality(
    model, 
    W, 
    eigvals_analytic, 
    state_bounds, 
    n_modes_to_keep=8
    ):

    n_eval_traj = 14
    eval_steps = 90
    
    rng = np.random.default_rng(0)
    num_modes = W.shape[1]
    residual_accum = np.zeros(num_modes, dtype=np.float64)

    for _ in range(n_eval_traj):
        x0 = np.zeros((1, model.state_dim), dtype=np.float32)
        for dim in range(model.state_dim):
            x0[0, dim] = rng.uniform(state_bounds[dim][0], state_bounds[dim][1])
            
        xt = torch.tensor(x0, dtype=torch.float32)
        traj = [x0.flatten()]
        with torch.no_grad():
            for _ in range(eval_steps):
                xt = model(xt)
                traj.append(xt.cpu().numpy().flatten())

        traj = np.asarray(traj, dtype=np.float32)
        with torch.no_grad():
            z_roll = safe_expand(model, torch.tensor(traj)).cpu().numpy()
            phi_roll = z_roll @ W

        lhs = phi_roll[1:, :] 
        rhs = phi_roll[:-1, :] * eigvals_analytic[None, :] 

        num = np.linalg.norm(lhs - rhs, axis=0) 
        den = np.linalg.norm(phi_roll[:-1, :], axis=0) + 1e-12 
        residual_accum += num / den

    residual_mean = residual_accum / max(1, n_eval_traj)

    # Calculate Spatial Score (Taking a 2D slice if state_dim > 2)
    x_range = np.linspace(state_bounds[0][0], state_bounds[0][1], 50)
    y_range = np.linspace(state_bounds[1][0], state_bounds[1][1], 50)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    grid_cols = [X_grid.ravel(), Y_grid.ravel()]
    # Pad higher dimensions with their median values
    for dim in range(2, model.state_dim):
        dim_mean = (state_bounds[dim][0] + state_bounds[dim][1]) / 2.0
        grid_cols.append(np.full_like(X_grid.ravel(), dim_mean))
        
    pts = np.column_stack(grid_cols)
    
    with torch.no_grad():
        z_grid = safe_expand(model, torch.as_tensor(pts, dtype=torch.float32)).cpu().numpy()
        phi_grid = z_grid @ W
    
    spatial_std = np.std(np.real(phi_grid), axis=0)

    def to_unit(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-12)

    res_score = 1.0 - to_unit(residual_mean)
    spat_score = to_unit(spatial_std)
    
    mode_score = 0.8 * res_score + 0.2 * spat_score
    ranked_indices = np.argsort(mode_score)[::-1]

    best_ids = ranked_indices[:n_modes_to_keep]
    
    return best_ids, mode_score, residual_mean


def plot_eigenvalue_spectrum(eigvals, mode_scores, score_metric, save_path=None):
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
    ax.set_ylim(y_bounds)
    
    ax.axhline(0, color=circle_color, linewidth=0.8, alpha=0.5)
    ax.axvline(0, color=circle_color, linewidth=0.8, alpha=0.5)
    # ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Eigenvalue Spectrum (Colored by {score_metric})", fontsize=12)
    ax.set_xlabel("$\mathbb{R}(\lambda)$")
    ax.set_ylabel("$\mathbb{I}(\lambda)$")
    
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'Mode {score_metric}', rotation=270, labelpad=15)
    
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_freq_magnitude(eigvals, mode_scores, score_metric, save_path=None):
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

    ### Annotate overlapping points together
    # Threshold for how close points need to be to be considered overlapping (as a fraction of the total spread)
    overlap_th = 0.05

    # Calculate spreads for dynamic thresholding
    x_spread = (frequencies.max() - frequencies.min()) 
    y_spread = (magnitudes.max() - magnitudes.min()) 
    x_th = overlap_th * x_spread + 1e-4
    y_th = overlap_th * y_spread + 1e-4

    # Loop through points
    plot_checklist = np.zeros(len(eigvals), dtype=bool)  # keep track of annotated points
    for i, (freq, mag) in enumerate(zip(frequencies, magnitudes)):

        # skip if already annotated
        if plot_checklist[i]:
            continue
        plot_checklist[i] = True
        
        # loop through nearby points
        nearby_indices = np.where((np.abs(frequencies - freq) < x_th) & (np.abs(magnitudes - mag) < y_th))[0]
        label_string = f"{i}"
        for near_idx in nearby_indices:

            # skip self or if already annotated
            if (near_idx == i) or (plot_checklist[near_idx]):
                continue

            # add to label and mark as annotated
            label_string += f", {near_idx}"
            plot_checklist[near_idx] = True

        # label cluster
        x_coord = freq
        y_coord = mag - 0.5 * overlap_th * y_spread  # offset label slightly below the cluster
        ax.text(x_coord, y_coord, label_string, fontsize=8, ha='center', va='center')
    
    ax.axhline(1.0, color=circle_color, linestyle='--', alpha=0.6, label="Unit Circle (Stable)")
    
    ax.set_title("Eigenvalue Distribution: Frequency vs. Magnitude", fontsize=14)
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


def plot_mode_contributions_vs_quality(V, phi_traj, best_ids, scores, state_dim, n_top=10, save_path=None):
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
    ax.set_title("Physical Mode Energy (Weighted by Average Activation)")
    
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


def plot_mode_energy_vs_quality(V, scores, state_dim, n_top=20, save_path=None):
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
    ax.set_title("Mode Selection: Quality vs. Physical Energy", fontsize=14)
    
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()