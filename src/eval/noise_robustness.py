import os
import numpy as np
import torch
import warnings
import textwrap
import matplotlib.pyplot as plt

from src.eval.model_io import predict_rollout_from_x0


# ============================================================
# Basic utilities
# ============================================================

def _real_array(x):
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        arr = np.real(arr)
    return np.asarray(arr, dtype=float)

def ensure_3d(X):
    if X.ndim == 2:
        return X[:, None, :]
    if X.ndim == 3:
        return X
    raise ValueError(f"Expected X to be 2D or 3D, got {X.shape}")


def load_split_X(data_path, split="test"):
    path = os.path.join(data_path, f"{split}.npz")
    data = np.load(path, allow_pickle=True)
    return ensure_3d(data["X"]), data


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def clean_mode_indices(mode_indices):
    """
    Convert mode index selections to a contiguous int64 numpy array.

    Needed because np.argsort(...)[::-1] creates a negative-stride view,
    which torch.as_tensor cannot handle.
    """
    if mode_indices is None:
        return None

    idx = np.array(mode_indices, dtype=np.int64, copy=True).reshape(-1)

    if idx.size == 0:
        return None

    if np.any(idx < 0):
        raise ValueError("mode_indices must be non-negative.")

    return idx


def safe_nanmean(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.nanmean(values))

def state_rms_scale(X):
    flat = X.reshape(-1, X.shape[-1])
    scale = np.sqrt(np.mean(flat**2, axis=0))
    scale[scale == 0.0] = 1.0
    return scale


def rmse(a, b):
    """
    Robust RMSE for arrays that may have different rollout lengths.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if np.iscomplexobj(a):
        a = np.real(a)
    if np.iscomplexobj(b):
        b = np.real(b)

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return np.nan

    a = a[:n]
    b = b[:n]

    if a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])
    if b.ndim > 2:
        b = b.reshape(-1, b.shape[-1])

    n = min(a.shape[0], b.shape[0])
    a = a[:n]
    b = b[:n]

    mask = np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
    if not np.any(mask):
        return np.nan

    diff = a[mask] - b[mask]
    return float(np.sqrt(np.mean(diff**2)))

def perturbation_rmse(pred, feedback):
    """
    RMS size of the injected perturbation in Option C.

    feedback[k] = pred[k] + injected_noise for k >= 1,
    so this measures how large the actual injected noise is.
    """
    pred = np.asarray(pred)
    feedback = np.asarray(feedback)

    if np.iscomplexobj(pred):
        pred = np.real(pred)
    if np.iscomplexobj(feedback):
        feedback = np.real(feedback)

    pred = np.asarray(pred, dtype=float)
    feedback = np.asarray(feedback, dtype=float)

    n = min(len(pred), len(feedback))
    if n <= 1:
        return np.nan

    return rmse(feedback[1:n], pred[1:n])


def finite_prefix_length(X):
    X = np.asarray(X)
    if X.ndim == 1:
        return int(np.isfinite(X).all())

    good = np.isfinite(X).all(axis=-1)
    bad = np.where(~good)[0]
    if len(bad) == 0:
        return len(X)
    return int(bad[0])


def parse_plot_traj_indices(plot_traj_indices, n_traj):
    if plot_traj_indices is None:
        return list(range(min(4, n_traj)))

    out = []
    for idx in plot_traj_indices:
        idx = int(idx)
        if 0 <= idx < n_traj:
            out.append(idx)

    if len(out) == 0:
        out = list(range(min(4, n_traj)))

    return out[:4]

def _get_delay_depth(model):
    return int(getattr(getattr(model, "expander", None), "delay_depth", 1))

def _get_x0_hist(X, t0, traj_id, delay_depth):
    """Returns properly formatted delay history [x(t0), x(t0-1), ...]"""
    if delay_depth <= 1:
        return X[t0, traj_id, :]
    return np.concatenate([X[t0 - lag, traj_id, :] for lag in range(delay_depth)])

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
            # BROAD PROTECTION: Detect any significant coupling anywhere in the matrix
            L_mag = np.abs(L)
            connected = L_mag > 1e-3
            np.fill_diagonal(connected, False) 
            connected = connected | connected.T # Symmetrize
            
            active = np.zeros(L.shape[0], dtype=bool)
            active[list(expanded_idx)] = True
            
            # Loop until no new connected modes are found (Transitive Closure)
            while True:
                new_active = active | (active @ connected)
                if np.array_equal(active, new_active):
                    break # We found the whole isolated subsystem
                active = new_active
                
            expanded_idx.update(np.where(active)[0])
            
    return sorted(list(expanded_idx))


# ============================================================
# Model wrappers
# ============================================================

def rollout_model(model, x0, steps, *, model_name=None, extras=None, rollout_mode="DMD", mode_indices=None):
    if model is None:
        if model_name is None or extras is None:
            raise ValueError("Baseline rollouts require model_name and extras.")
        return np.asarray(
            predict_rollout_from_x0(
                x0=x0,
                steps=steps,
                model_name=model_name,
                model=model,
                extras=extras,
                mode_indices=mode_indices,
            )
        )

    with torch.no_grad():
        # Regression_DMD-style rollout API
        if hasattr(model, "Phi_lift_fitted") and hasattr(model, "rollout"):
            kwargs = {"mode": rollout_mode}
            if rollout_mode in {"DMD", "projected_DMD"}:
                idx_np = clean_mode_indices(mode_indices)
                
                # --- FIX: UNIVERSAL MODE COUPLING PROTECTION (REGRESSION DMD) ---
                if idx_np is not None:
                    idx_np = np.array(_get_expanded_indices(idx_np, model), dtype=np.int64)
                # ----------------------------------------------------------------
                
                kwargs["mode_indices"] = idx_np
            out = model.rollout(x0, steps=steps, **kwargs)
            return to_numpy(out)

        # ML_DMD native mode-subset rollout in modal space
        if mode_indices is not None and hasattr(model, "_get_modal_coords") and hasattr(model, "_step_modal"):
            dev = next(model.parameters()).device
            m_dtype = next(model.parameters()).dtype
            x0_t = torch.as_tensor(x0, dtype=m_dtype, device=dev)

            is_1d = x0_t.ndim == 1
            if is_1d:
                x0_t = x0_t.unsqueeze(0)

            delay_depth = int(getattr(model.expander, "delay_depth", 1))

            expected_width = model.state_dim * delay_depth
            if x0_t.shape[1] != expected_width:
                 raise ValueError(
                    f"{model.__class__.__name__}.rollout expected delay-state width "
                    f"{expected_width}, got {x0_t.shape[1]}."
                )

            x = x0_t
            z = model.expander.expand(x)
            z_norm = model._normalize(z)
            b = model._get_modal_coords(z_norm)

            idx_np = clean_mode_indices(mode_indices)
            if idx_np is None:
                return to_numpy(model.rollout(x0, steps=steps))

            expanded_idx = _get_expanded_indices(idx_np, model)
            idx = torch.as_tensor(np.array(expanded_idx, dtype=np.int64), dtype=torch.long, device=dev)
            
            mask = torch.zeros_like(b)
            mask[:, idx] = 1.0
            b = b * mask

            if delay_depth > 1:
                x_curr0 = x[:, : model.state_dim]
            else:
                x_curr0 = x

            traj = [x_curr0.squeeze(0).detach().cpu().numpy()]
            for _ in range(steps):
                b = model._step_modal(b)
                b = b * mask
                z_norm_next = model._modal_to_latent(b)
                z_next = model._unnormalize(z_norm_next)
                x_next_head = model.expander.de_expand(z_next)

                traj.append(x_next_head.squeeze(0).detach().cpu().numpy())

                if delay_depth > 1:
                    x = torch.cat([x_next_head, x[:, :-model.state_dim]], dim=1)
                    z = model.expander.expand(x)
                    z_norm = model._normalize(z)
                    b = model._get_modal_coords(z_norm) * mask

            return np.asarray(traj)

        # Generic learned-model rollout API (ML_DMD default and others)
        out = model.rollout(x0, steps=steps)
        return to_numpy(out)


def predict_one_step(model, x0, *, model_name=None, extras=None, rollout_mode="DMD", mode_indices=None):
    out = rollout_model(
        model,
        x0,
        steps=1,
        model_name=model_name,
        extras=extras,
        rollout_mode=rollout_mode,
        mode_indices=mode_indices,
    )
    if len(out) < 2:
        return np.full_like(np.asarray(out[0] if len(out) > 0 else x0, dtype=float), np.nan)
    return out[1]


# ============================================================
# Option B: modal projection denoising
# ============================================================

def modal_project_denoise(model, x, mode_indices=None):
    """
    noisy x -> lift -> project onto selected DMD modes -> decode.
    If mode_indices is None, uses all modes in the checkpoint.
    """
    # If no model is provided (e.g., baseline numpy models), return input unchanged
    if model is None:
        arr = np.asarray(x)
        if arr.ndim == 1:
            return arr
        return arr

    # Only call .eval() for objects that have it (e.g., torch.nn.Module)
    if hasattr(model, "eval") and callable(getattr(model, "eval")):
        try:
            model.eval()
        except Exception:
            pass

    with torch.no_grad():
        # Try to infer device and dtype from model tensors
        dev = torch.device("cpu")
        m_dtype = torch.float64
        
        # Check explicit Regression DMD and ML DMD tensors first
        for attr in ["Phi_lift_fitted", "Phi", "scale", "lift_scale"]:
            val = getattr(model, attr, None)
            if isinstance(val, torch.Tensor):
                dev = val.device
                m_dtype = val.dtype
                break
        else:
            # Fallback to parameters/buffers safely
            try:
                p = next(model.parameters())
                dev = p.device
                m_dtype = p.dtype
            except Exception:
                try:
                    b = next(model.buffers())
                    dev = b.device
                    m_dtype = b.dtype
                except Exception:
                    pass

        x_t = torch.as_tensor(x, dtype=m_dtype, device=dev)
        is_1d = x_t.ndim == 1
        if is_1d:
            x_t = x_t.unsqueeze(0)

        # Regression_DMD path
        if hasattr(model, "_normalize_x") and hasattr(model, "psi_scale") and hasattr(model, "Phi_lift_fitted"):
            x_n = model._normalize_x(x_t)
            z = (model.expand(x_n) / model.psi_scale).to(torch.complex128)

            Phi = model.Phi_lift_fitted.to(torch.complex128)
            # Use the pre-calculated pseudo-inverse to match ML W-matrix projection
            Phi_pinv = model.Phi_pinv_fitted.to(torch.complex128) 
            C = model.C_fitted.to(torch.complex128)

            # 1. Project onto ALL modes to get true independent coordinates
            b_modal = (Phi_pinv @ z.T).T 

            idx_np = clean_mode_indices(mode_indices)
            if idx_np is not None:
                idx_np = np.array(_get_expanded_indices(idx_np, model), dtype=np.int64)
                idx = torch.as_tensor(idx_np, dtype=torch.long, device=dev)
                
                # 2. Mask the dropped modes (Exact Truncation, no re-solving)
                mask = torch.zeros_like(b_modal)
                mask[:, idx] = 1.0
                b_modal = b_modal * mask

            # 3. Project back to latent space
            z_proj = (Phi @ b_modal.T).T

            x_proj_n = (C @ z_proj.T).T.real.to(m_dtype)
            x_proj = model._denormalize_x(x_proj_n)

        # ML_DMD path
        elif hasattr(model, "_normalize") and hasattr(model, "_get_modal_coords") and hasattr(model, "expander"):
            z_raw = model.expander.expand(x_t)
            z_norm = model._normalize(z_raw)

            b_modal = model._get_modal_coords(z_norm)

            idx_np = clean_mode_indices(mode_indices)
            if idx_np is not None:
                expanded_idx = _get_expanded_indices(idx_np, model)
                idx = torch.as_tensor(np.array(expanded_idx, dtype=np.int64), dtype=torch.long, device=dev)
                
                mask = torch.zeros_like(b_modal)
                mask[:, idx] = 1.0
                b_modal = b_modal * mask

            z_norm_proj = model._modal_to_latent(b_modal)
            z_proj = model._unnormalize(z_norm_proj)
            x_proj = model.expander.de_expand(z_proj)

        else:
            warnings.warn(
                "modal_project_denoise: model does not implement a supported modal API; returning input unchanged",
                RuntimeWarning,
            )
            arr = np.asarray(x)
            return arr[0] if (arr.ndim == 1) else arr

        out = x_proj.detach().cpu().numpy()
        return out[0] if is_1d else out


# ============================================================
# Option A: repeated one-step prediction from noisy states
# ============================================================

def repeated_one_step_predictions(
    model,
    X_noisy,
    X_clean,
    *,
    model_name=None,
    extras=None,
    max_pairs=None,
    rollout_mode="DMD",
    mode_indices=None,
):
    preds, targets = [], []
    delay_depth = _get_delay_depth(model)
    t0 = max(0, delay_depth - 1)
    
    T, N, _ = X_noisy.shape
    count = 0

    for j in range(N):
        for t in range(t0, T - 1):
            # FIX: Use real history
            x0 = _get_x0_hist(X_noisy, t, j, delay_depth)

            try:
                pred = predict_one_step(
                    model,
                    x0,
                    model_name=model_name,
                    extras=extras,
                    rollout_mode=rollout_mode,
                    mode_indices=mode_indices,
                )
            except Exception:
                pred = np.full(X_clean.shape[-1], np.nan)

            preds.append(pred)
            targets.append(X_clean[t + 1, j])

            count += 1
            if max_pairs is not None and count >= max_pairs:
                return np.asarray(preds), np.asarray(targets)

    return _real_array(preds), _real_array(targets)


# ============================================================
# Option C: noisy-feedback DMD rollout
# ============================================================

def noisy_feedback_rollout(
    model,
    x0_clean_hist, # <--- Expect full history here
    X_clean_ref,
    *,
    model_name=None,
    extras=None,
    noise_std,
    noise_scale,
    seed=0,
    rollout_mode="DMD",
    mode_indices=None,
    blowup_factor=1e6,
):
    rng = np.random.default_rng(seed)
    steps = X_clean_ref.shape[0] - 1
    delay_depth = _get_delay_depth(model)
    state_dim = X_clean_ref.shape[-1]
    
    x_feedback_hist = _real_array(x0_clean_hist).copy()
    
    preds = [x_feedback_hist[:state_dim].copy()]
    feedback_states = [x_feedback_hist[:state_dim].copy()]

    clean_scale = max(1.0, float(np.nanmax(np.abs(X_clean_ref))))
    blowup_threshold = blowup_factor * clean_scale

    for _ in range(steps):
        try:
            pred = predict_one_step(
                model, x_feedback_hist, model_name=model_name, extras=extras,
                rollout_mode=rollout_mode, mode_indices=mode_indices,
            )
        except Exception as exc:
            print(f"[warning] noisy_feedback_rollout failed: {exc}")
            break

        pred = _real_array(pred)
        if pred.ndim > 1:
            pred = pred[0]

        if (not np.all(np.isfinite(pred))) or np.max(np.abs(pred)) > blowup_threshold:
            print("[warning] noisy_feedback_rollout blew up; stopping early.")
            break

        preds.append(pred.copy())

        noise = rng.normal(0.0, noise_std * noise_scale, size=pred.shape)
        noisy_curr = _real_array(pred + noise)

        if (not np.all(np.isfinite(noisy_curr))) or np.max(np.abs(noisy_curr)) > blowup_threshold:
            break

        feedback_states.append(noisy_curr.copy())
        
        # FIX: Update the rolling history buffer for the next step
        if delay_depth > 1:
            x_feedback_hist = np.concatenate([noisy_curr, x_feedback_hist[:-state_dim]])
        else:
            x_feedback_hist = noisy_curr

    return _real_array(preds), _real_array(feedback_states)


# ============================================================
# Option D: noisy-initial-condition free rollout
# ============================================================

def noisy_initial_free_rollout(
    model,
    x0_noisy,
    X_clean_ref,
    *,
    model_name=None,
    extras=None,
    rollout_mode="DMD",
    mode_indices=None,
):
    """
    D: noisy x0 -> autonomous DMD rollout -> compare to clean trajectory.
    """
    steps = X_clean_ref.shape[0] - 1

    try:
        pred = rollout_model(
            model,
            x0_noisy,
            steps=steps,
            model_name=model_name,
            extras=extras,
            rollout_mode=rollout_mode,
            mode_indices=mode_indices,
        )
    except Exception as exc:
        print(f"[warning] noisy_initial_free_rollout failed: {exc}")
        pred = np.full_like(_real_array(X_clean_ref), np.nan)

    n = min(len(pred), len(X_clean_ref))
    return _real_array(pred[:n])


# ============================================================
# Mode diagnostics for full model
# ============================================================

def modal_coefficients(model, X_states, mode_indices=None, max_samples=20000):
    X_states = np.asarray(X_states)
    delay_depth = _get_delay_depth(model)
    
    # FIX: Build genuine delay histories from the trajectory
    if delay_depth > 1 and X_states.ndim == 3:
        T, N, D = X_states.shape
        hist_list = []
        for lag in range(delay_depth):
            hist_list.append(X_states[delay_depth - 1 - lag : T - lag, :, :])
        X_hist = np.concatenate(hist_list, axis=-1)
        X_flat = X_hist.reshape(-1, X_hist.shape[-1])
    else:
        X_flat = X_states.reshape(-1, X_states.shape[-1])

    if max_samples is not None and X_flat.shape[0] > max_samples:
        idx = np.linspace(0, X_flat.shape[0] - 1, max_samples).astype(int)
        X_flat = X_flat[idx]

    with torch.no_grad():
        # Try to infer device and dtype from model tensors
        dev = torch.device("cpu")
        m_dtype = torch.float64
        
        # Check explicit Regression DMD and ML DMD tensors first
        for attr in ["Phi_lift_fitted", "Phi", "scale", "lift_scale"]:
            val = getattr(model, attr, None)
            if isinstance(val, torch.Tensor):
                dev = val.device
                m_dtype = val.dtype
                break
        else:
            # Fallback to parameters/buffers safely
            try:
                p = next(model.parameters())
                dev = p.device
                m_dtype = p.dtype
            except Exception:
                try:
                    b = next(model.buffers())
                    dev = b.device
                    m_dtype = b.dtype
                except Exception:
                    pass

        x_t = torch.as_tensor(X_flat, dtype=m_dtype, device=dev)

        # Branch for Regression_DMD-like API
        if hasattr(model, "_normalize_x") and hasattr(model, "psi_scale") and hasattr(model, "Phi_lift_fitted"):
            x_n = model._normalize_x(x_t)
            z = (model.expand(x_n) / model.psi_scale).to(torch.complex128)

            # 1. Project onto ALL modes first to get true independent coordinates
            Phi_pinv = getattr(model, "Phi_pinv_fitted", torch.linalg.pinv(model.Phi_lift_fitted)).to(torch.complex128)
            b = (Phi_pinv @ z.T).T

            # 2. Slice the columns AFTER projection if subset is requested
            idx_np = clean_mode_indices(mode_indices)
            if idx_np is not None:
                idx = torch.as_tensor(idx_np, dtype=torch.long, device=dev)
                b = b[:, idx]

        # Branch for ML_DMD-like API
        elif hasattr(model, "_normalize") and hasattr(model, "_get_modal_coords"):
            # ML_DMD normalizes lifted features, so expand first then normalize
            # Ensure expander is on the same device
            expander = getattr(model, "expander", None)
            if expander is None:
                raise RuntimeError("Model appears to be ML_DMD but has no expander")

            z_raw = expander.expand(x_t)
            # ML_DMD stores lift_scale buffer
            if hasattr(model, "_normalize"):
                z_norm = model._normalize(z_raw)
            else:
                # fallback to manual scaling
                lift_scale = getattr(model, "lift_scale", None)
                if lift_scale is None:
                    z_norm = z_raw
                else:
                    z_norm = z_raw / lift_scale.to(dev)

            # Use model helper to get modal coefficients
            b = model._get_modal_coords(z_norm)
            if mode_indices is not None:
                expanded_idx = _get_expanded_indices(mode_indices, model)
                idx_t = torch.as_tensor(np.array(expanded_idx, dtype=np.int64), dtype=torch.long, device=b.device)
                
                mask = torch.zeros_like(b)
                mask[:, idx_t] = 1.0
                b = b * mask

            # If mode subset selected, subset columns
            idx_np = clean_mode_indices(mode_indices)
            if idx_np is not None:
                idx = torch.as_tensor(idx_np, dtype=torch.long, device=dev)
                b = b[:, idx]

        else:
            raise RuntimeError("Unsupported model API for modal coefficient extraction")

    return b.cpu().numpy()


def compute_mode_diagnostics(model, X_states, dt=None, max_samples=20000):
    b = modal_coefficients(model, X_states, max_samples=max_samples)

    coeff_rms = np.sqrt(np.mean(np.abs(b) ** 2, axis=0))

    # Default placeholders
    mode_state_norm = None
    lambdas = None

    # Regression_DMD-style saved spectral objects
    if hasattr(model, "Phi_state_fitted") and hasattr(model, "Lambda_fitted"):
        Phi_state = model.Phi_state_fitted.detach().cpu().numpy()
        
        # --- FIX: Restore physical units before calculating contribution norms ---
        if hasattr(model, "x_scale"):
            x_scale = model.x_scale[:model.state_dim].detach().cpu().numpy()
            # Multiply each row (state dimension) by its corresponding physical scale
            Phi_state = Phi_state * x_scale[:, None]
            
        mode_state_norm = np.linalg.norm(Phi_state, axis=0)
        lambdas = model.Lambda_fitted.detach().cpu().numpy()

    else:
        # ML_DMD-style: try to reconstruct state contribution per mode
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            try:
                dev = next(model.buffers()).device
            except StopIteration:
                dev = torch.device("cpu")

        if hasattr(model, "Phi") and hasattr(model, "expander"):
            Phi_param = model.Phi.detach().to(device=dev)
            n_modes = Phi_param.shape[1]
            Phi_state_cols = []
            for j in range(n_modes):
                col = Phi_param[:, j].unsqueeze(0)  # (1, latent_dim)
                if hasattr(model, "_unnormalize"):
                    col_phys = model._unnormalize(col)
                else:
                    lift_scale = getattr(model, "lift_scale", None)
                    if lift_scale is not None:
                        col_phys = col * lift_scale.to(dev)
                    else:
                        col_phys = col

                try:
                    state_vec = model.expander.de_expand(col_phys).squeeze(0)
                    Phi_state_cols.append(state_vec.detach().cpu().numpy())
                except Exception:
                    Phi_state_cols.append(np.zeros(getattr(model, "state_dim", 0)))

            if len(Phi_state_cols) > 0:
                Phi_state = np.stack(Phi_state_cols, axis=1)
                mode_state_norm = np.linalg.norm(Phi_state, axis=0)
            else:
                mode_state_norm = np.zeros_like(coeff_rms)

        else:
            mode_state_norm = np.zeros_like(coeff_rms)

        # Eigenvalues: prefer get_eigenvalues(), else try Lambda or form K
        if hasattr(model, "get_eigenvalues"):
            lambdas = model.get_eigenvalues().detach().cpu().numpy()
        elif hasattr(model, "Lambda") and hasattr(model, "Phi"):
            try:
                K = (model.Phi @ model.Lambda @ torch.linalg.pinv(model.Phi)).to(device=dev)
                eigvals = torch.linalg.eigvals(K.to(torch.complex128))
                lambdas = eigvals.detach().cpu().numpy()
            except Exception:
                lambdas = np.full(coeff_rms.shape, np.nan)
        else:
            lambdas = np.full(coeff_rms.shape, np.nan)

    # Ensure arrays have compatible shapes
    coeff_rms = np.asarray(coeff_rms)
    mode_state_norm = np.asarray(mode_state_norm)
    if mode_state_norm.shape[0] != coeff_rms.shape[0]:
        # Broadcast or trim as needed
        if mode_state_norm.size == 0:
            mode_state_norm = np.zeros_like(coeff_rms)
        else:
            mode_state_norm = np.resize(mode_state_norm, coeff_rms.shape)

    state_contribution = coeff_rms * mode_state_norm

    lambdas = np.asarray(lambdas)
    if lambdas.shape[0] != coeff_rms.shape[0]:
        # Pad or truncate
        if lambdas.size == 0:
            lambdas = np.full(coeff_rms.shape, np.nan)
        else:
            lambdas = np.resize(lambdas, coeff_rms.shape)

    eig_abs = np.abs(lambdas)

    if dt is not None and np.all(np.isfinite(lambdas)):
        mu = np.log(lambdas) / dt
        growth_rate = np.real(mu)
        frequency = np.imag(mu) / (2 * np.pi)
    else:
        growth_rate = np.full_like(eig_abs, np.nan, dtype=float)
        frequency = np.full_like(eig_abs, np.nan, dtype=float)

    order_amp = np.argsort(coeff_rms)[::-1]
    order_contrib = np.argsort(state_contribution)[::-1]
    order_amp = np.array(order_amp, dtype=np.int64, copy=True)
    order_contrib = np.array(order_contrib, dtype=np.int64, copy=True)

    cum_amp_score = cumulative_score_fractions(coeff_rms, order_amp)
    cum_contrib_score = cumulative_score_fractions(state_contribution, order_contrib)
    print("Correlation between coeff_rms and lift_scale:", np.corrcoef(coeff_rms, lift_scale))
    return {
        "coeff_rms": coeff_rms,
        "state_contribution": state_contribution,
        "eig_abs": eig_abs,
        "growth_rate": growth_rate,
        "frequency": frequency,
        "order_amp": order_amp,
        "order_contrib": order_contrib,
        "cum_amp_score": cum_amp_score,
        "cum_contrib_score": cum_contrib_score,
        "lambdas": lambdas,
    }


def write_mode_diagnostics(path, diag, top_n=20):
    with open(path, "w", encoding="utf-8") as f:
        thresholds = [0.90, 0.95, 0.99, 0.999]

        f.write("Mode score threshold ranks\n")
        f.write("Coefficient amplitude score mass:\n")
        for th in thresholds:
            r = rank_for_score_threshold(diag["cum_amp_score"], th)
            f.write(f"  {100*th:5.1f}% -> {r} modes\n")

        f.write("State contribution score mass:\n")
        for th in thresholds:
            r = rank_for_score_threshold(diag["cum_contrib_score"], th)
            f.write(f"  {100*th:5.1f}% -> {r} modes\n")

        f.write("\nSelected-k score fractions\n")
        for k in [1, 2, 3, 5, 10, 15, 20]:
            if k <= len(diag["cum_amp_score"]):
                f.write(
                    f"k={k:2d}: "
                    f"amp_score={diag['cum_amp_score'][k-1]:.6f}, "
                    f"contrib_score={diag['cum_contrib_score'][k-1]:.6f}\n"
                )

        f.write("\n")

        f.write("Top modes by RMS modal coefficient amplitude\n")
        f.write("idx, coeff_rms, state_contribution, |lambda|, growth_rate, frequency\n")

        for idx in diag["order_amp"][:top_n]:
            f.write(
                f"{idx}, "
                f"{diag['coeff_rms'][idx]:.8e}, "
                f"{diag['state_contribution'][idx]:.8e}, "
                f"{diag['eig_abs'][idx]:.8e}, "
                f"{diag['growth_rate'][idx]:.8e}, "
                f"{diag['frequency'][idx]:.8e}\n"
            )

        f.write("\nTop modes by RMS state contribution\n")
        f.write("idx, coeff_rms, state_contribution, |lambda|, growth_rate, frequency\n")

        for idx in diag["order_contrib"][:top_n]:
            f.write(
                f"{idx}, "
                f"{diag['coeff_rms'][idx]:.8e}, "
                f"{diag['state_contribution'][idx]:.8e}, "
                f"{diag['eig_abs'][idx]:.8e}, "
                f"{diag['growth_rate'][idx]:.8e}, "
                f"{diag['frequency'][idx]:.8e}\n"
            )

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
    
    # --- FIX: Expand the raw slice to include missing conjugate pairs ---
    raw_idx = np.asarray(diag["order_contrib"][:n_modes], dtype=int)
    expanded_idx = _get_expanded_indices(raw_idx, model)
    
    # Return the expanded list and its ACTUAL length
    return np.asarray(expanded_idx, dtype=int), pct_label, len(expanded_idx)


def _mode_subset_specs_for_fractions(diag, fractions, total_modes, model):
    specs = []
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

        specs.append({
            "pct_labels": [pct_label],
            "mode_indices": contrib_idx,
            "n_modes": n_modes,
            "actual_score": actual_score,
        })

    return specs
        

def cumulative_score_fractions(scores, order):
    scores = np.asarray(scores, dtype=float)
    order = np.array(order, dtype=np.int64, copy=True)

    power = scores**2
    total = np.sum(power)

    if total <= 0 or not np.isfinite(total):
        return np.full(len(order), np.nan)

    return np.cumsum(power[order]) / total


def rank_for_score_threshold(cumfrac, threshold):
    idx = np.where(cumfrac >= threshold)[0]
    if len(idx) == 0:
        return None
    return int(idx[0] + 1)

# ============================================================
# Plotting
# ============================================================

def make_2x2_axes(title, subtitle=None):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), squeeze=False)
    if subtitle:
        # Wrap each line individually to preserve your explicit \n characters
        subtitle = "\n".join(textwrap.fill(line, width=68) for line in str(subtitle).splitlines())
        fig.suptitle(f"{title}\n{subtitle}", fontsize=14)
    else:
        fig.suptitle(title, fontsize=14)
    return fig, axes.ravel()


def finish_2x2(fig, axes, outpath): # <-- Add kwargs here
    for ax in axes:
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True, linestyle="--", alpha=0.5) 
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=250)
    plt.close(fig)


def plot_option_A_grid(
    model,
    X_clean,
    X_noisy,
    traj_indices,
    outdir,
    *,
    steps,
    model_name=None,
    extras=None,
    rollout_mode,
    mode_indices=None,
    subtitle=None,
):
    fig, axes = make_2x2_axes("Option A: repeated one-step prediction from noisy states", subtitle=subtitle)
    # 1. CALCULATE BOUNDS ONCE

    delay_depth = _get_delay_depth(model)
    t0 = max(0, delay_depth - 1)
    T_plot = min(steps + t0, X_clean.shape[0] - 1)

    for ax, traj_id in zip(axes, traj_indices):
        clean_next = X_clean[t0 + 1 : T_plot + 1, traj_id, :]
        preds = []

        for t in range(t0, T_plot):
            x0_hist = _get_x0_hist(X_noisy, t, traj_id, delay_depth) # <--- FIXED
            try:
                pred = predict_one_step(
                    model, x0_hist, model_name=model_name, extras=extras,
                    rollout_mode=rollout_mode, mode_indices=mode_indices,
                )
            except Exception:
                pred = np.full(X_clean.shape[-1], np.nan)

            preds.append(pred)

        preds = np.asarray(preds)

        ax.plot(clean_next[:, 0], clean_next[:, 1], "-", linewidth=2.5, label="Clean next")
        ax.plot(preds[:, 0], preds[:, 1], "--", linewidth=2.0, label="Pred from noisy")
        ax.set_title(f"Trajectory {traj_id}")

    # 2. PASS BOUNDS TO FINISH
    finish_2x2(
        fig, axes, 
        os.path.join(outdir, "option_A_one_step_grid.png"))


def plot_option_B_grid(
    model,
    X_clean,
    X_noisy,
    traj_indices,
    outdir,
    *,
    steps,
    mode_indices=None,
    subtitle=None,
):
    fig, axes = make_2x2_axes("Option B: modal projection denoising", subtitle=subtitle)

    delay_depth = _get_delay_depth(model)
    t0 = max(0, delay_depth - 1)
    T_plot = min(steps + t0 + 1, X_clean.shape[0])

    for ax, traj_id in zip(axes, traj_indices):
        # --- FIX 4: Build sliding window for plot ---
        if delay_depth > 1:
            hist_noisy = []
            for lag in range(delay_depth):
                hist_noisy.append(X_noisy[t0 - lag : T_plot - lag, traj_id, :])
            noisy_input = np.concatenate(hist_noisy, axis=-1)
        else:
            noisy_input = X_noisy[:T_plot, traj_id, :]

        clean_target = X_clean[t0:T_plot, traj_id, :]
        noisy_target = X_noisy[t0:T_plot, traj_id, :]

        proj = modal_project_denoise(model, noisy_input, mode_indices=mode_indices)

        ax.plot(clean_target[:, 0], clean_target[:, 1], "-", linewidth=2.5, label="Clean")
        ax.plot(noisy_target[:, 0], noisy_target[:, 1], ":", linewidth=2.0, label="Noisy observed")
        ax.plot(proj[:, 0], proj[:, 1], "--", linewidth=2.0, label="Modal projected")
        ax.set_title(f"Trajectory {traj_id}")

    finish_2x2(fig, axes, os.path.join(outdir, "option_B_modal_projection_grid.png"))


def plot_option_C_grid(
    model,
    X_clean,
    traj_indices,
    outdir,
    *,
    steps,
    noise_std,
    noise_scale,
    seed,
    model_name=None,
    extras=None,
    rollout_mode,
    mode_indices=None,
    subtitle=None,
):
    fig, axes = make_2x2_axes("Option C: noisy-feedback DMD rollout", subtitle=subtitle)
    delay_depth = _get_delay_depth(model)
    t0 = max(0, delay_depth - 1)

    for k, (ax, traj_id) in enumerate(zip(axes, traj_indices)):
        # FIX: Start from t0 and pass history
        X_ref = X_clean[t0 : t0 + steps + 1, traj_id, :]
        x0_hist = _get_x0_hist(X_clean, t0, traj_id, delay_depth)

        pred, feedback = noisy_feedback_rollout(
            model,
            x0_hist, # <--- PASS TRUE HISTORY
            X_ref,
            model_name=model_name,
            extras=extras,
            noise_std=noise_std,
            noise_scale=noise_scale,
            seed=seed + k,
            rollout_mode=rollout_mode,
            mode_indices=mode_indices,
        )

        n = min(len(pred), len(feedback), len(X_ref))
        X_ref = X_ref[:n]
        pred = pred[:n]
        feedback = feedback[:n]

        ax.plot(X_ref[:, 0], X_ref[:, 1], "-", linewidth=2.5, label="Clean")
        ax.plot(pred[:, 0], pred[:, 1], "--", linewidth=2.0, label="Prediction")
        ax.plot(feedback[:, 0], feedback[:, 1], ":", linewidth=2.0, label="Perturbed input")
        ax.set_title(f"Trajectory {traj_id}")

    finish_2x2(fig, axes, os.path.join(outdir, "option_C_noisy_feedback_grid.png"))


def plot_option_D_grid(
    model,
    X_clean,
    X_noisy,
    traj_indices,
    outdir,
    *,
    steps,
    model_name=None,
    extras=None,
    rollout_mode,
    mode_indices=None,
    subtitle=None,
):
    fig, axes = make_2x2_axes("Option D: noisy-initial-condition free rollout", subtitle=subtitle)
    delay_depth = _get_delay_depth(model)
    t0 = max(0, delay_depth - 1)

    for ax, traj_id in zip(axes, traj_indices):
        # FIX: Start from t0 and pass history
        X_ref = X_clean[t0 : t0 + steps + 1, traj_id, :]
        x0_noisy_hist = _get_x0_hist(X_noisy, t0, traj_id, delay_depth)

        pred = noisy_initial_free_rollout(
            model,
            x0_noisy_hist, # <--- PASS TRUE HISTORY
            X_ref,
            model_name=model_name,
            extras=extras,
            rollout_mode=rollout_mode,
            mode_indices=mode_indices,
        )

        n = min(len(pred), len(X_ref))
        X_ref = X_ref[:n]
        pred = pred[:n]

        ax.plot(X_ref[:, 0], X_ref[:, 1], "-", linewidth=2.5, label="Clean")
        ax.plot(pred[:, 0], pred[:, 1], "--", linewidth=2.0, label="Free rollout from noisy x0")
        ax.scatter([x0_noisy_hist[0]], [x0_noisy_hist[1]], s=45, label="Noisy x0", zorder=5)

        ax.set_title(f"Trajectory {traj_id}")

    finish_2x2(fig, axes, os.path.join(outdir, "option_D_noisy_initial_free_rollout_grid.png"))


# ============================================================
# Metric evaluation
# ============================================================

def evaluate_variant(
    *,
    model,
    model_name,
    extras,
    X_clean,
    X_noisy,
    traj_indices,
    steps,
    noise_std_for_feedback,
    noise_scale,
    max_pairs,
    seed,
    rollout_mode,
    mode_indices,
    variant_name,
):
    delay_depth = _get_delay_depth(model)
    
    # --- FIX 1: Build sliding window history for Option B ---
    if delay_depth > 1:
        T, N, D = X_noisy.shape
        hist_n = []
        for lag in range(delay_depth):
            hist_n.append(X_noisy[delay_depth - 1 - lag : T - lag, :, :])
        Xn_input_packed = np.concatenate(hist_n, axis=-1)
        Xn_input_flat = Xn_input_packed.reshape(-1, Xn_input_packed.shape[-1])
        
        # The targets for comparison are just the CURRENT states (lag=0)
        t0 = delay_depth - 1
        Xc_target_flat = X_clean[t0:, :, :].reshape(-1, D)
        Xn_target_flat = X_noisy[t0:, :, :].reshape(-1, D)
    else:
        Xn_input_flat = X_noisy.reshape(-1, X_noisy.shape[-1])
        Xc_target_flat = X_clean.reshape(-1, X_clean.shape[-1])
        Xn_target_flat = X_noisy.reshape(-1, X_noisy.shape[-1])

    X_proj = modal_project_denoise(model, Xn_input_flat, mode_indices=mode_indices)

    modal_input_rmse = rmse(Xn_target_flat, Xc_target_flat)
    modal_output_rmse = rmse(X_proj, Xc_target_flat)
    # ---------------------------------------------------------

    # A
    one_preds, one_targets = repeated_one_step_predictions(
        model,
        X_noisy,
        X_clean,
        model_name=model_name,
        extras=extras,
        max_pairs=max_pairs,
        rollout_mode=rollout_mode,
        mode_indices=mode_indices,
    )
    one_step_rmse = rmse(one_preds, one_targets)

    # C + D over selected plot trajectories
    feedback_rmses = []
    feedback_valid = []
    feedback_fracs = []
    noisy_init_rmses = []
    feedback_perturb_rmses = []

    # --- FIX 2: Use t0 and histories for metrics ---
    t0 = max(0, delay_depth - 1)

    for k, traj_id in enumerate(traj_indices):
        X_ref = X_clean[t0 : t0 + steps + 1, traj_id, :]
        
        x0_clean_hist = _get_x0_hist(X_clean, t0, traj_id, delay_depth)
        x0_noisy_hist = _get_x0_hist(X_noisy, t0, traj_id, delay_depth)

        fb_pred, fb_feedback = noisy_feedback_rollout(
            model,
            x0_clean_hist,  # <--- FIXED
            X_ref,
            model_name=model_name,
            extras=extras,
            noise_std=noise_std_for_feedback,
            noise_scale=noise_scale,
            seed=seed + k,
            rollout_mode=rollout_mode,
            mode_indices=mode_indices,
        )

        valid = min(len(fb_pred), len(X_ref))
        feedback_valid.append(valid)
        feedback_fracs.append(valid / len(X_ref))
        feedback_rmses.append(rmse(fb_pred, X_ref))
        feedback_perturb_rmses.append(perturbation_rmse(fb_pred, fb_feedback))

        d_pred = noisy_initial_free_rollout(
            model,
            x0_noisy_hist,  # <--- FIXED
            X_ref,
            model_name=model_name,
            extras=extras,
            rollout_mode=rollout_mode,
            mode_indices=mode_indices,
        )
        noisy_init_rmses.append(rmse(d_pred, X_ref))

    row = {
        "variant": variant_name,
        "n_modes_used": -1 if mode_indices is None else int(len(mode_indices)),
        "mode_indices": "all" if mode_indices is None else ",".join(map(str, mode_indices)),
        "modal_input_rmse_noisy_vs_clean": float(modal_input_rmse),
        "modal_output_rmse_projected_vs_clean": float(modal_output_rmse),
        "one_step_rmse_pred_vs_clean_next": float(one_step_rmse),
        "feedback_rollout_rmse_pred_vs_clean": safe_nanmean(feedback_rmses),
        "feedback_valid_steps_mean": safe_nanmean(feedback_valid),
        "feedback_completed_fraction_mean": safe_nanmean(feedback_fracs),
        "noisy_initial_free_rollout_rmse": safe_nanmean(noisy_init_rmses),
        "noise_std_for_feedback": float(noise_std_for_feedback),
        "feedback_perturbation_rmse_mean": safe_nanmean(feedback_perturb_rmses),
    }

    return row


# ============================================================
# Main suite
# ============================================================
def plot_all_options_for_variant(
    *,
    model,
    model_name,
    extras,
    X_clean,
    X_noisy,
    traj_indices,
    outdir,
    steps,
    noise_std_for_feedback,
    noise_scale,
    seed,
    rollout_mode,
    mode_indices,
    modal_supported=True,
    subtitle=None,
):
    os.makedirs(outdir, exist_ok=True)

    plot_option_A_grid(
        model,
        X_clean,
        X_noisy,
        traj_indices,
        outdir,
        steps=steps,
        model_name=model_name,
        extras=extras,
        rollout_mode=rollout_mode,
        mode_indices=mode_indices,
        subtitle=subtitle,
    )

    if modal_supported:
        plot_option_B_grid(
            model,
            X_clean,
            X_noisy,
            traj_indices,
            outdir,
            steps=steps,
            mode_indices=mode_indices,
            subtitle=subtitle,
        )

    plot_option_C_grid(
        model,
        X_clean,
        traj_indices,
        outdir,
        steps=steps,
        noise_std=noise_std_for_feedback,
        noise_scale=noise_scale,
        seed=seed,
        model_name=model_name,
        extras=extras,
        rollout_mode=rollout_mode,
        mode_indices=mode_indices,
        subtitle=subtitle,
    )

    plot_option_D_grid(
        model,
        X_clean,
        X_noisy,
        traj_indices,
        outdir,
        steps=steps,
        model_name=model_name,
        extras=extras,
        rollout_mode=rollout_mode,
        mode_indices=mode_indices,
        subtitle=subtitle,
    )

def run_noise_robustness_suite(
    *,
    model,
    model_name,
    extras,
    clean_data_path,
    noisy_data_path,
    outdir,
    split="test",
    traj_index=0,
    steps=200,
    noise_std_for_feedback=0.001,
    max_pairs=5000,
    seed=0,
    plot_traj_indices=None,
    mode_subset_thresholds=None,
    feedback_rollout_mode="DMD",
    plot_mode_subsets=False,
):
    os.makedirs(outdir, exist_ok=True)

    X_clean, clean_data = load_split_X(clean_data_path, split)
    X_noisy, noisy_data = load_split_X(noisy_data_path, split)

    system = str(clean_data["system"]) if "system" in clean_data else ""
    subtitle = None
    if system:
        from src.eval.diagnostics import format_model_label

        subtitle = format_model_label(model_name, model, extras, system=system)

    if X_clean.shape != X_noisy.shape:
        raise ValueError(f"Clean/noisy shape mismatch: {X_clean.shape} vs {X_noisy.shape}")

    dt = None
    if "dt" in clean_data:
        dt = float(np.asarray(clean_data["dt"]).item())

    traj_indices = parse_plot_traj_indices(plot_traj_indices, X_clean.shape[1])
    scale = state_rms_scale(X_clean)

    # Normalize subset specification so percentage-style lists like
    # [1, 5, 10, 25, 50, 100] map to fractions [0.01, ..., 1.0].
    # Fraction-style inputs in [0, 1] remain unchanged.
    if mode_subset_thresholds is not None:
        normalized_thresholds = [float(t) for t in mode_subset_thresholds]
        if any(t > 1.0 for t in normalized_thresholds):
            normalized_thresholds = [t / 100.0 for t in normalized_thresholds]
        mode_subset_thresholds = [
            float(np.clip(t, 0.0, 1.0)) for t in normalized_thresholds if t > 0.0
        ]

    rows = []

    # ------------------------------------------------------------
    # Mode diagnostics from this checkpoint (optional)
    # If the model does not support modal extraction, fall back to an
    # empty diagnostics record and continue with the base evaluation.
    # ------------------------------------------------------------
    modal_supported = False
    try:
        if model is None:
            raise RuntimeError("No torch model object available; skip modal diagnostics.")
        diag = compute_mode_diagnostics(model, X_clean, dt=dt)
        write_mode_diagnostics(os.path.join(outdir, "mode_diagnostics.txt"), diag)

        np.savez(
            os.path.join(outdir, "mode_diagnostics.npz"),
            coeff_rms=diag["coeff_rms"],
            state_contribution=diag["state_contribution"],
            eig_abs=diag["eig_abs"],
            growth_rate=diag["growth_rate"],
            frequency=diag["frequency"],
            order_amp=diag["order_amp"],
            order_contrib=diag["order_contrib"],
            cum_amp_score=diag["cum_amp_score"],
            cum_contrib_score=diag["cum_contrib_score"],
        )

        print("\n--- Mode diagnostics ---")
        print("Top by RMS coefficient amplitude:", diag["order_amp"][:10].tolist())
        print("Top by RMS state contribution    :", diag["order_contrib"][:10].tolist())

        for th in [0.90, 0.95, 0.99, 0.999]:
            r_amp = rank_for_score_threshold(diag["cum_amp_score"], th)
            r_con = rank_for_score_threshold(diag["cum_contrib_score"], th)
            print(f"Mode score {100*th:5.1f}%: amp -> {r_amp} modes, contribution -> {r_con} modes")

        for k in [5, 10, 15]:
            if k <= len(diag["cum_amp_score"]):
                print(
                    f"k={k}: "
                    f"amp_score={diag['cum_amp_score'][k-1]:.4f}, "
                    f"contrib_score={diag['cum_contrib_score'][k-1]:.4f}"
                )

        modal_supported = bool(len(diag["coeff_rms"]))
    except Exception as exc:
        print(f"[noise_robustness] Modal diagnostics unavailable: {exc}")
        diag = {
            "coeff_rms": np.array([], dtype=float),
            "state_contribution": np.array([], dtype=float),
            "eig_abs": np.array([], dtype=float),
            "growth_rate": np.array([], dtype=float),
            "frequency": np.array([], dtype=float),
            "order_amp": np.array([], dtype=int),
            "order_contrib": np.array([], dtype=int),
            "cum_amp_score": np.array([], dtype=float),
            "cum_contrib_score": np.array([], dtype=float),
        }
        # write an empty diagnostics file so downstream tooling knows this ran
        try:
            write_mode_diagnostics(os.path.join(outdir, "mode_diagnostics.txt"), diag)
        except Exception:
            pass

    # ------------------------------------------------------------
    # Base checkpoint evaluation
    # ------------------------------------------------------------
    base_row = evaluate_variant(
        model=model,
        model_name=model_name,
        extras=extras,
        X_clean=X_clean,
        X_noisy=X_noisy,
        traj_indices=traj_indices,
        steps=steps,
        noise_std_for_feedback=noise_std_for_feedback,
        noise_scale=scale,
        max_pairs=max_pairs,
        seed=seed,
        rollout_mode=feedback_rollout_mode,
        mode_indices=None,
        variant_name="checkpoint_default",
    )
    rows.append(base_row)

    # If subset plots include a 100% contribution run, that folder already contains
    # the full-model plots, so skip duplicate root-level plot exports.
    has_pct100_subset_request = False
    if plot_mode_subsets and mode_subset_thresholds is not None:
        for th in mode_subset_thresholds:
            raw_th = float(th)
            pct_value = raw_th * 100.0 if raw_th <= 1.0 else raw_th
            pct_value = float(np.clip(pct_value, 0.0, 100.0))
            if np.isclose(pct_value, 100.0):
                has_pct100_subset_request = True
                break

    # Base plots
    if not (has_pct100_subset_request and modal_supported):
        plot_all_options_for_variant(
            model=model,
            model_name=model_name,
            extras=extras,
            X_clean=X_clean,
            X_noisy=X_noisy,
            traj_indices=traj_indices,
            outdir=outdir,
            steps=steps,
            noise_std_for_feedback=noise_std_for_feedback,
            noise_scale=scale,
            seed=seed,
            rollout_mode=feedback_rollout_mode,
            mode_indices=None,
            modal_supported=modal_supported,
            subtitle=subtitle,
        )

    # ------------------------------------------------------------
    # Full-model mode subset evaluations
    # ------------------------------------------------------------
    if mode_subset_thresholds is not None:
        n_modes = len(diag["coeff_rms"])

        for spec in _mode_subset_specs_for_fractions(diag, mode_subset_thresholds, n_modes, model):
            contrib_idx = spec["mode_indices"]
            n_used = spec["n_modes"]
            actual_score = spec.get("actual_score")
            pct_tag = "_".join(spec["pct_labels"])
            pct_text = ", ".join(f"{label}%" for label in spec["pct_labels"])
            contrib_name = f"top_contrib_modes_pct{pct_tag}"

            contrib_row = evaluate_variant(
                model=model,
                model_name=model_name,
                extras=extras,
                X_clean=X_clean,
                X_noisy=X_noisy,
                traj_indices=traj_indices,
                steps=steps,
                noise_std_for_feedback=noise_std_for_feedback,
                noise_scale=scale,
                max_pairs=max_pairs,
                seed=seed,
                rollout_mode=feedback_rollout_mode,
                mode_indices=contrib_idx,
                variant_name=contrib_name,
            )
            rows.append(contrib_row)

            if plot_mode_subsets:
                subset_subtitle = subtitle
                if actual_score is not None:
                    subset_text = f"{n_used} Modes ({pct_text}) | Contribution = {actual_score:.3f}"
                else:
                    subset_text = f"{n_used} Modes ({pct_text})"
                if subset_subtitle:
                    subset_subtitle = f"{subset_subtitle}\n{subset_text}"
                else:
                    subset_subtitle = subset_text

                plot_all_options_for_variant(
                    model=model,
                    model_name=model_name,
                    extras=extras,
                    X_clean=X_clean,
                    X_noisy=X_noisy,
                    traj_indices=traj_indices,
                    outdir=os.path.join(outdir, contrib_name),
                    steps=steps,
                    noise_std_for_feedback=noise_std_for_feedback,
                    noise_scale=scale,
                    seed=seed,
                    rollout_mode=feedback_rollout_mode,
                    mode_indices=contrib_idx,
                    modal_supported=modal_supported,
                    subtitle=subset_subtitle,
                )

    # ------------------------------------------------------------
    # Save local summary
    # ------------------------------------------------------------
    np.savez(
        os.path.join(outdir, "noise_robustness_rows.npz"),
        rows=np.asarray(rows, dtype=object),
    )

    with open(os.path.join(outdir, "noise_robustness_rows.txt"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write("\n")
            for key, value in row.items():
                f.write(f"{key}: {value}\n")

    return base_row, rows