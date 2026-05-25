import os
import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# Basic utilities
# ============================================================

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

def state_rms_scale(X):
    flat = X.reshape(-1, X.shape[-1])
    scale = np.sqrt(np.mean(flat**2, axis=0))
    scale[scale == 0.0] = 1.0
    return scale


def rmse(a, b):
    """
    Robust RMSE for arrays that may have different rollout lengths.
    """
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


# ============================================================
# Model wrappers
# ============================================================

def rollout_model(model, x0, steps, *, rollout_mode="DMD", mode_indices=None):
    kwargs = {"mode": rollout_mode}

    if rollout_mode in {"DMD", "projected_DMD"}:
        kwargs["mode_indices"] = clean_mode_indices(mode_indices)

    with torch.no_grad():
        out = model.rollout(x0, steps=steps, **kwargs)

    return to_numpy(out)


def predict_one_step(model, x0, *, rollout_mode="DMD", mode_indices=None):
    out = rollout_model(
        model,
        x0,
        steps=1,
        rollout_mode=rollout_mode,
        mode_indices=mode_indices,
    )
    return out[1]


# ============================================================
# Option B: modal projection denoising
# ============================================================

def modal_project_denoise(model, x, mode_indices=None):
    """
    noisy x -> lift -> project onto selected DMD modes -> decode.
    If mode_indices is None, uses all modes in the checkpoint.
    """
    model.eval()

    with torch.no_grad():
        x_t = torch.as_tensor(x, dtype=torch.float64)
        is_1d = x_t.ndim == 1
        if is_1d:
            x_t = x_t.unsqueeze(0)

        x_n = model._normalize_x(x_t)
        z = (model.expand(x_n) / model.psi_scale).to(torch.complex128)

        Phi = model.Phi_lift_fitted.to(torch.complex128)
        C = model.C_fitted.to(torch.complex128)

        idx_np = clean_mode_indices(mode_indices)
        if idx_np is not None:
            idx = torch.as_tensor(idx_np, dtype=torch.long)
            Phi = Phi[:, idx]

        b = (torch.linalg.pinv(Phi) @ z.T).T
        z_proj = (Phi @ b.T).T

        x_proj_n = (C @ z_proj.T).T.real.to(torch.float64)
        x_proj = model._denormalize_x(x_proj_n)

        out = x_proj.cpu().numpy()
        return out[0] if is_1d else out


# ============================================================
# Option A: repeated one-step prediction from noisy states
# ============================================================

def repeated_one_step_predictions(
    model,
    X_noisy,
    X_clean,
    *,
    max_pairs=None,
    rollout_mode="DMD",
    mode_indices=None,
):
    preds = []
    targets = []

    T, N, _ = X_noisy.shape
    count = 0

    for j in range(N):
        for t in range(T - 1):
            x0 = X_noisy[t, j]

            try:
                pred = predict_one_step(
                    model,
                    x0,
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

    return np.asarray(preds), np.asarray(targets)


# ============================================================
# Option C: noisy-feedback DMD rollout
# ============================================================

def noisy_feedback_rollout(
    model,
    x0_clean,
    X_clean_ref,
    *,
    noise_std,
    noise_scale,
    seed=0,
    rollout_mode="DMD",
    mode_indices=None,
    blowup_factor=1e6,
):
    """
    Closed-loop noisy-feedback rollout.

    Each step:
        pred_next = DMD_rollout(x_feedback, 1)[1]
        x_feedback_next = pred_next + Gaussian noise

    This now uses the requested rollout mode, e.g. DMD.
    """
    rng = np.random.default_rng(seed)

    steps = X_clean_ref.shape[0] - 1
    x_feedback = np.asarray(x0_clean, dtype=float).copy()

    preds = [x_feedback.copy()]
    feedback_states = [x_feedback.copy()]

    clean_scale = max(1.0, float(np.nanmax(np.abs(X_clean_ref))))
    blowup_threshold = blowup_factor * clean_scale

    for _ in range(steps):
        try:
            pred = predict_one_step(
                model,
                x_feedback,
                rollout_mode=rollout_mode,
                mode_indices=mode_indices,
            )
        except Exception as exc:
            print(f"[warning] noisy_feedback_rollout failed: {exc}")
            break

        pred = np.asarray(pred, dtype=float)

        if pred.ndim > 1:
            pred = pred[0]

        if (not np.all(np.isfinite(pred))) or np.max(np.abs(pred)) > blowup_threshold:
            print("[warning] noisy_feedback_rollout became non-finite or blew up; stopping early.")
            break

        preds.append(pred.copy())

        noise = rng.normal(0.0, noise_std * noise_scale, size=pred.shape)
        x_feedback = pred + noise

        if (not np.all(np.isfinite(x_feedback))) or np.max(np.abs(x_feedback)) > blowup_threshold:
            print("[warning] noisy feedback state became non-finite or blew up; stopping early.")
            break

        feedback_states.append(x_feedback.copy())

    return np.asarray(preds), np.asarray(feedback_states)


# ============================================================
# Option D: noisy-initial-condition free rollout
# ============================================================

def noisy_initial_free_rollout(
    model,
    x0_noisy,
    X_clean_ref,
    *,
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
            rollout_mode=rollout_mode,
            mode_indices=mode_indices,
        )
    except Exception as exc:
        print(f"[warning] noisy_initial_free_rollout failed: {exc}")
        pred = np.full_like(X_clean_ref, np.nan)

    n = min(len(pred), len(X_clean_ref))
    return np.asarray(pred[:n])


# ============================================================
# Mode diagnostics for full model
# ============================================================

def modal_coefficients(model, X_states, mode_indices=None, max_samples=20000):
    X_states = np.asarray(X_states)
    X_flat = X_states.reshape(-1, X_states.shape[-1])

    if max_samples is not None and X_flat.shape[0] > max_samples:
        idx = np.linspace(0, X_flat.shape[0] - 1, max_samples).astype(int)
        X_flat = X_flat[idx]

    with torch.no_grad():
        x_t = torch.as_tensor(X_flat, dtype=torch.float64)
        x_n = model._normalize_x(x_t)
        z = (model.expand(x_n) / model.psi_scale).to(torch.complex128)

        Phi = model.Phi_lift_fitted.to(torch.complex128)

        idx_np = clean_mode_indices(mode_indices)
        if idx_np is not None:
            idx = torch.as_tensor(idx_np, dtype=torch.long)
            Phi = Phi[:, idx]

        b = (torch.linalg.pinv(Phi) @ z.T).T

    return b.cpu().numpy()


def compute_mode_diagnostics(model, X_states, dt=None, max_samples=20000):
    b = modal_coefficients(model, X_states, max_samples=max_samples)

    coeff_rms = np.sqrt(np.mean(np.abs(b) ** 2, axis=0))

    Phi_state = model.Phi_state_fitted.detach().cpu().numpy()
    mode_state_norm = np.linalg.norm(Phi_state, axis=0)
    state_contribution = coeff_rms * mode_state_norm

    lambdas = model.Lambda_fitted.detach().cpu().numpy()
    eig_abs = np.abs(lambdas)

    if dt is not None and dt > 0:
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

def make_2x2_axes(title):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), squeeze=False)
    fig.suptitle(title, fontsize=16)
    return fig, axes.ravel()


def finish_2x2(fig, axes, outpath):
    for ax in axes:
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
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
    rollout_mode,
    mode_indices=None,
):
    fig, axes = make_2x2_axes("Option A: repeated one-step prediction from noisy states")

    T_plot = min(steps, X_clean.shape[0] - 1)

    for ax, traj_id in zip(axes, traj_indices):
        clean_next = X_clean[1 : T_plot + 1, traj_id, :]
        preds = []

        for t in range(T_plot):
            try:
                pred = predict_one_step(
                    model,
                    X_noisy[t, traj_id, :],
                    rollout_mode=rollout_mode,
                    mode_indices=mode_indices,
                )
            except Exception:
                pred = np.full(X_clean.shape[-1], np.nan)

            preds.append(pred)

        preds = np.asarray(preds)

        ax.plot(clean_next[:, 0], clean_next[:, 1], "-", linewidth=2.5, label="Clean next")
        ax.plot(preds[:, 0], preds[:, 1], "--", linewidth=2.0, label="Pred from noisy")
        ax.set_title(f"Trajectory {traj_id}")

    finish_2x2(fig, axes, os.path.join(outdir, "option_A_one_step_grid.png"))


def plot_option_B_grid(
    model,
    X_clean,
    X_noisy,
    traj_indices,
    outdir,
    *,
    steps,
    mode_indices=None,
):
    fig, axes = make_2x2_axes("Option B: modal projection denoising")

    T_plot = min(steps, X_clean.shape[0])

    for ax, traj_id in zip(axes, traj_indices):
        clean = X_clean[:T_plot, traj_id, :]
        noisy = X_noisy[:T_plot, traj_id, :]
        proj = modal_project_denoise(model, noisy, mode_indices=mode_indices)

        ax.plot(clean[:, 0], clean[:, 1], "-", linewidth=2.5, label="Clean")
        ax.plot(noisy[:, 0], noisy[:, 1], ":", linewidth=2.0, label="Noisy observed")
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
    rollout_mode,
    mode_indices=None,
):
    fig, axes = make_2x2_axes("Option C: noisy-feedback DMD rollout")

    for k, (ax, traj_id) in enumerate(zip(axes, traj_indices)):
        X_ref = X_clean[: steps + 1, traj_id, :]
        pred, feedback = noisy_feedback_rollout(
            model,
            X_ref[0],
            X_ref,
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
        ax.set_title(f"Trajectory {traj_id}, valid {n}/{steps+1}")

    finish_2x2(fig, axes, os.path.join(outdir, "option_C_noisy_feedback_grid.png"))


def plot_option_D_grid(
    model,
    X_clean,
    X_noisy,
    traj_indices,
    outdir,
    *,
    steps,
    rollout_mode,
    mode_indices=None,
):
    fig, axes = make_2x2_axes("Option D: noisy-initial-condition free rollout")

    for ax, traj_id in zip(axes, traj_indices):
        X_ref = X_clean[: steps + 1, traj_id, :]
        x0_noisy = X_noisy[0, traj_id, :]

        pred = noisy_initial_free_rollout(
            model,
            x0_noisy,
            X_ref,
            rollout_mode=rollout_mode,
            mode_indices=mode_indices,
        )

        n = min(len(pred), len(X_ref))
        X_ref = X_ref[:n]
        pred = pred[:n]

        ax.plot(X_ref[:, 0], X_ref[:, 1], "-", linewidth=2.5, label="Clean")
        ax.plot(pred[:, 0], pred[:, 1], "--", linewidth=2.0, label="Free rollout from noisy x0")
        ax.scatter([x0_noisy[0]], [x0_noisy[1]], s=45, label="Noisy x0", zorder=5)

        ax.set_title(f"Trajectory {traj_id}")

    finish_2x2(fig, axes, os.path.join(outdir, "option_D_noisy_initial_free_rollout_grid.png"))


# ============================================================
# Metric evaluation
# ============================================================

def evaluate_variant(
    *,
    model,
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
    # B
    Xn_flat = X_noisy.reshape(-1, X_noisy.shape[-1])
    Xc_flat = X_clean.reshape(-1, X_clean.shape[-1])

    X_proj = modal_project_denoise(model, Xn_flat, mode_indices=mode_indices)

    modal_input_rmse = rmse(Xn_flat, Xc_flat)
    modal_output_rmse = rmse(X_proj, Xc_flat)

    # A
    one_preds, one_targets = repeated_one_step_predictions(
        model,
        X_noisy,
        X_clean,
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

    for k, traj_id in enumerate(traj_indices):
        X_ref = X_clean[: steps + 1, traj_id, :]
        X_obs = X_noisy[: steps + 1, traj_id, :]

        fb_pred, fb_feedback = noisy_feedback_rollout(
            model,
            X_ref[0],
            X_ref,
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
            X_obs[0],
            X_ref,
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
        "feedback_rollout_rmse_pred_vs_clean": float(np.nanmean(feedback_rmses)),
        "feedback_valid_steps_mean": float(np.nanmean(feedback_valid)),
        "feedback_completed_fraction_mean": float(np.nanmean(feedback_fracs)),
        "noisy_initial_free_rollout_rmse": float(np.nanmean(noisy_init_rmses)),
        "noise_std_for_feedback": float(noise_std_for_feedback),
        "feedback_perturbation_rmse_mean": float(np.nanmean(feedback_perturb_rmses)),
    }

    return row


# ============================================================
# Main suite
# ============================================================
def plot_all_options_for_variant(
    *,
    model,
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
):
    os.makedirs(outdir, exist_ok=True)

    plot_option_A_grid(
        model,
        X_clean,
        X_noisy,
        traj_indices,
        outdir,
        steps=steps,
        rollout_mode=rollout_mode,
        mode_indices=mode_indices,
    )

    plot_option_B_grid(
        model,
        X_clean,
        X_noisy,
        traj_indices,
        outdir,
        steps=steps,
        mode_indices=mode_indices,
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
        rollout_mode=rollout_mode,
        mode_indices=mode_indices,
    )

    plot_option_D_grid(
        model,
        X_clean,
        X_noisy,
        traj_indices,
        outdir,
        steps=steps,
        rollout_mode=rollout_mode,
        mode_indices=mode_indices,
    )

def run_noise_robustness_suite(
    *,
    model,
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
    mode_subset_ks=None,
    feedback_rollout_mode="DMD",
    plot_mode_subsets=False,
):
    os.makedirs(outdir, exist_ok=True)

    X_clean, clean_data = load_split_X(clean_data_path, split)
    X_noisy, noisy_data = load_split_X(noisy_data_path, split)

    if X_clean.shape != X_noisy.shape:
        raise ValueError(f"Clean/noisy shape mismatch: {X_clean.shape} vs {X_noisy.shape}")

    dt = None
    if "dt" in clean_data:
        dt = float(np.asarray(clean_data["dt"]).item())

    traj_indices = parse_plot_traj_indices(plot_traj_indices, X_clean.shape[1])
    scale = state_rms_scale(X_clean)

    rows = []

    # ------------------------------------------------------------
    # Mode diagnostics from this checkpoint
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Base checkpoint evaluation
    # ------------------------------------------------------------
    base_row = evaluate_variant(
        model=model,
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

    # Base plots
    plot_all_options_for_variant(
        model=model,
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
    )

    # ------------------------------------------------------------
    # Full-model mode subset evaluations
    # ------------------------------------------------------------
    if mode_subset_ks is not None:
        n_modes = len(diag["coeff_rms"])

        for k in mode_subset_ks:
            k = int(k)
            if k <= 0:
                continue
            k = min(k, n_modes)

            amp_idx = diag["order_amp"][:k]
            contrib_idx = diag["order_contrib"][:k]

            amp_name = f"top_amp_modes_k{k}"
            contrib_name = f"top_contrib_modes_k{k}"

            amp_row = evaluate_variant(
                model=model,
                X_clean=X_clean,
                X_noisy=X_noisy,
                traj_indices=traj_indices,
                steps=steps,
                noise_std_for_feedback=noise_std_for_feedback,
                noise_scale=scale,
                max_pairs=max_pairs,
                seed=seed,
                rollout_mode=feedback_rollout_mode,
                mode_indices=amp_idx,
                variant_name=amp_name,
            )
            rows.append(amp_row)

            contrib_row = evaluate_variant(
                model=model,
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
                plot_all_options_for_variant(
                    model=model,
                    X_clean=X_clean,
                    X_noisy=X_noisy,
                    traj_indices=traj_indices,
                    outdir=os.path.join(outdir, amp_name),
                    steps=steps,
                    noise_std_for_feedback=noise_std_for_feedback,
                    noise_scale=scale,
                    seed=seed,
                    rollout_mode=feedback_rollout_mode,
                    mode_indices=amp_idx,
                )

                plot_all_options_for_variant(
                    model=model,
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