import matplotlib.pyplot as plt
import os
from itertools import combinations
import numpy as np

from src.eval.diagnostics import format_model_label

def darken_color(color, factor=0.5):
    """Darkens a matplotlib color by a factor (0.0 to 1.0)."""
    import matplotlib.colors as mc
    import colorsys
    rgb = mc.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, max(0, l * factor), s)

def plot_combined_rollout(X_true, X_hat, figdir, traj_index, *, model_label=None, system=None, xlim=None, ylim=None):
    """Plot a compact 2-panel rollout summary: Combined time series and phase space."""
    state_dim = X_true.shape[1]
    if state_dim < 2:
        raise ValueError("plot_combined_rollout requires at least 2 state dimensions")

    # --- BOUNDS CALCULATION (as requested previously) ---
    if xlim is None or ylim is None:
        x_min, x_max = float(np.nanmin(X_true[:, 0])), float(np.nanmax(X_true[:, 0]))
        y_min, y_max = float(np.nanmin(X_true[:, 1])), float(np.nanmax(X_true[:, 1]))
        x_pad, y_pad = max((x_max - x_min) * 0.05, 1e-3), max((y_max - y_min) * 0.05, 1e-3)
        if xlim is None: xlim = (x_min - x_pad, x_max + x_pad)
        if ylim is None: ylim = (y_min - y_pad, y_max + y_pad)
    # ----------------------------------------------------

    # Create 2 subplots instead of 3
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # --- AXIS 0: COMBINED TIME SERIES ---
    # Plot x1
    line1 = axes[0].plot(X_true[:, 0], label="True x1")
    color1 = line1[0].get_color()
    # Apply darken_color here
    axes[0].plot(X_hat[:, 0], "--", color=darken_color(color1), label="Pred x1")
    
    # Plot x2
    line2 = axes[0].plot(X_true[:, 1], label="True x2")
    color2 = line2[0].get_color()
    # Apply darken_color here
    axes[0].plot(X_hat[:, 1], "--", color=darken_color(color2), label="Pred x2")
    
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("State values")
    axes[0].set_title("States over time")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # --- AXIS 1: PHASE SPACE ---
    axes[1].plot(X_true[:, 0], X_true[:, 1], label="True phase")
    axes[1].plot(X_hat[:, 0], X_hat[:, 1], "--", label="Pred phase")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    axes[1].set_title("Phase space")
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    if model_label:
        # Since model_label is already a string, just use it directly!
        fig_title = f"Rollout Comparison: {model_label}"
        fig.suptitle(fig_title, fontsize=14)
        plt.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        plt.tight_layout()
        
    plt.savefig(f"{figdir}/rollout_idx{traj_index}.png", dpi=200)
    plt.close()


def plot_combined_rollout_with_reference(
    X_true,
    X_hat,
    X_ref,
    figdir,
    traj_index,
    true_label="True / observed",
    ref_label="Clean reference",
    model_label=None,
    system=None,
):
    """Combined 3-panel rollout summary with optional reference overlay."""

    state_dim = X_true.shape[1]
    if state_dim < 2:
        raise ValueError("plot_combined_rollout_with_reference requires at least 2 state dimensions")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    if X_ref is not None:
        axes[0].plot(X_ref[:, 0], label=f"{ref_label} x1", linewidth=2)
    axes[0].plot(X_true[:, 0], label=f"{true_label} x1", alpha=0.6)
    axes[0].plot(X_hat[:, 0], "--", label="Pred x1", linewidth=2)
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("x1")
    axes[0].set_title("x1 over time")
    axes[0].legend()

    if X_ref is not None:
        axes[1].plot(X_ref[:, 1], label=f"{ref_label} x2", linewidth=2)
    axes[1].plot(X_true[:, 1], label=f"{true_label} x2", alpha=0.6)
    axes[1].plot(X_hat[:, 1], "--", label="Pred x2", linewidth=2)
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("x2")
    axes[1].set_title("x2 over time")
    axes[1].legend()

    if X_ref is not None:
        axes[2].plot(X_ref[:, 0], X_ref[:, 1], label=ref_label, linewidth=2)
    axes[2].plot(X_true[:, 0], X_true[:, 1], label=true_label, alpha=0.6)
    axes[2].plot(X_hat[:, 0], X_hat[:, 1], "--", label="Prediction", linewidth=2)
    axes[2].set_xlabel("x1")
    axes[2].set_ylabel("x2")
    axes[2].set_title("Phase space rollout")
    axes[2].legend()

    if model_label is None:
        model_label = format_model_label("unknown", None, {}, system=system)
    fig.suptitle(f"Rollout summary\n{model_label}", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{figdir}/rollout_idx{traj_index}.png", dpi=200)
    plt.close(fig)


def plot_time_series(X_true, X_hat, figdir, traj_index, *, model_label=None, system=None):
    """
    Plot time series of each state dimension.
    Works for any dimensional system.
    """

    state_dim = X_true.shape[1]
    
    plt.figure(figsize=(10, 2.5 * state_dim))
    
    for i in range(state_dim):
        plt.subplot(state_dim, 1, i + 1)
        plt.plot(X_true[:, i], label="True")
        plt.plot(X_hat[:, i], "--", label="Predicted")
        
        plt.grid(True, linestyle="--", alpha=0.5) # <-- ADD GRID HERE
        
        plt.xlabel("Time step")
        plt.ylabel(f"x{i+1}")
        plt.title(f"x{i+1} over time")

        plt.legend()

    if model_label:
        fig = plt.gcf()
        fig_title = f"Time series rollout\n{model_label}"
        fig.suptitle(fig_title, fontsize=14, y=0.94)
        plt.tight_layout(rect=(0, 0, 1, 0.90))
    else:
        plt.tight_layout()
    plt.savefig(f"{figdir}/time_series_idx{traj_index}.png")
    plt.close()


def plot_phase_space(X_true, X_hat, system, figdir, model_name, traj_index, *, model_label=None):
    """
    Plot phase space projections.

    Behavior:
    - 2D systems: one phase portrait (x1 vs x2)
    - 3D systems: all pairwise projections
        (x1-x2, x1-x3, x2-x3)
    """

    state_dim = X_true.shape[1]

    # --------------------------------------------------
    # 2D systems
    # --------------------------------------------------
    if state_dim == 2:

        plt.figure(figsize=(10.5, 6.1))

        plt.plot(X_true[:, 0], X_true[:, 1], label="True")
        plt.plot(X_hat[:, 0], X_hat[:, 1], "--", label="Prediction")

        plt.xlabel("x1")
        plt.ylabel("x2")

        title = "Phase space rollout"
        if model_label is None:
            model_label = format_model_label(model_name, None, {}, system=system)
        plt.title(f"{title}\n{model_label}", pad=4)

        plt.legend()
        plt.tight_layout(rect=(0, 0, 1, 0.985))

        plt.savefig(f"{figdir}/rollout_idx{traj_index}.png")
        plt.close()

    # --------------------------------------------------
    # 3D systems (e.g. Lorenz)
    # --------------------------------------------------
    elif state_dim == 3:

        pairs = list(combinations(range(3), 2))

        plt.figure(figsize=(6 * len(pairs), 5))

        for k, (i, j) in enumerate(pairs):

            plt.subplot(1, len(pairs), k + 1)

            plt.plot(X_true[:, i], X_true[:, j], label="True")
            plt.plot(X_hat[:, i], X_hat[:, j], "--", label="Prediction")

            plt.xlabel(f"x{i+1}")
            plt.ylabel(f"x{j+1}")

            plt.title(f"x{i+1} vs x{j+1}")

            plt.legend()

        title = "Phase space rollout"
        if model_label is None:
            model_label = format_model_label(model_name, None, {}, system=system)
        fig = plt.gcf()
        fig.suptitle(f"{title}\n{model_label}", fontsize=14, y=0.95)
        plt.tight_layout(rect=(0, 0, 1, 0.92))

        plt.savefig(f"{figdir}/rollout_idx{traj_index}.png")
        plt.close()

    # --------------------------------------------------
    # fallback for higher dimensions
    # --------------------------------------------------
    else:

        plt.figure(figsize=(10.5, 6.1))

        plt.plot(X_true[:, 0], X_true[:, 1], label="True")
        plt.plot(X_hat[:, 0], X_hat[:, 1], "--", label="Prediction")

        plt.xlabel("x1")
        plt.ylabel("x2")

        title = "Phase space projection"
        if model_label is None:
            model_label = format_model_label(model_name, None, {}, system=system)
        plt.title(f"{title}\n{model_label}", pad=4)

        plt.legend()
        plt.tight_layout(rect=(0, 0, 1, 0.985))

        plt.savefig(f"{figdir}/rollout_idx{traj_index}.png")
        plt.close()

def plot_time_series_with_reference(
    X_true,
    X_hat,
    X_ref,
    figdir,
    traj_index,
    true_label="True / observed",
    ref_label="Clean reference",
    model_label=None,
    system=None,
):
    """
    Plot time series with optional clean/reference trajectory overlay.
    """
    state_dim = X_true.shape[1]

    plt.figure(figsize=(6 * state_dim, 4))

    for i in range(state_dim):
        plt.subplot(1, state_dim, i + 1)

        if X_ref is not None:
            plt.plot(X_ref[:, i], label=f"{ref_label} x{i+1}", linewidth=2)

        plt.plot(X_true[:, i], label=f"{true_label} x{i+1}", alpha=0.6)
        plt.plot(X_hat[:, i], "--", label=f"Pred x{i+1}", linewidth=2)

        plt.xlabel("Time step")
        plt.ylabel(f"x{i+1}")
        plt.title(f"x{i+1} over time")
        plt.legend()

    if model_label:
        fig = plt.gcf()
        fig_title = f"Time series rollout\n{model_label}"
        fig.suptitle(fig_title, fontsize=14, y=0.94)
        plt.tight_layout(rect=(0, 0, 1, 0.90))
    else:
        plt.tight_layout()
    plt.savefig(f"{figdir}/time_series_overlay_idx{traj_index}.png", dpi=200)
    plt.close()


def plot_phase_space_with_reference(
    X_true,
    X_hat,
    X_ref,
    system,
    figdir,
    model_name,
    traj_index,
    true_label="True / observed",
    ref_label="Clean reference",
    model_label=None,
    xlim=None,
    ylim=None,
):
    """
    Plot phase space with optional clean/reference trajectory overlay.
    """
    state_dim = X_true.shape[1]

    if state_dim < 2:
        return
        
    # --- ADD BOUNDS CALCULATION ---
    ref_data = X_ref if X_ref is not None else X_true
    if xlim is None or ylim is None:
        x_min, x_max = float(np.nanmin(ref_data[:, 0])), float(np.nanmax(ref_data[:, 0]))
        y_min, y_max = float(np.nanmin(ref_data[:, 1])), float(np.nanmax(ref_data[:, 1]))
        x_pad, y_pad = max((x_max - x_min) * 0.05, 1e-3), max((y_max - y_min) * 0.05, 1e-3)
        if xlim is None: xlim = (x_min - x_pad, x_max + x_pad)
        if ylim is None: ylim = (y_min - y_pad, y_max + y_pad)
    # ------------------------------

    plt.figure(figsize=(6, 6))

    if X_ref is not None:
        plt.plot(X_ref[:, 0], X_ref[:, 1], label=ref_label, linewidth=2)

    plt.plot(X_true[:, 0], X_true[:, 1], label=true_label, alpha=0.8)
    plt.plot(X_hat[:, 0], X_hat[:, 1], "--", label="Predicted", linewidth=2)
    
    # --- APPLY BOUNDS AND GRID ---
    ax = plt.gca()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, linestyle="--", alpha=0.5)
    # -----------------------------

    plt.xlabel("x1")
    plt.ylabel("x2")
    title = "Phase space rollout"
    if model_label is None:
        model_label = format_model_label(model_name, None, {}, system=system)
    plt.title(f"{title}\n{model_label}", pad=8)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{figdir}/rollout_overlay_idx{traj_index}.png", dpi=200)
    plt.close()