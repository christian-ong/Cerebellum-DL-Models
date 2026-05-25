import matplotlib.pyplot as plt
import os
from itertools import combinations


def plot_time_series(X_true, X_hat, figdir, traj_index):
    """
    Plot time series of each state dimension.
    Works for any dimensional system.
    """

    state_dim = X_true.shape[1]

    plt.figure(figsize=(6 * state_dim, 4))

    for i in range(state_dim):
        plt.subplot(1, state_dim, i + 1)

        plt.plot(X_true[:, i], label=f"True x{i+1}")
        plt.plot(X_hat[:, i], "--", label=f"Pred x{i+1}")

        plt.xlabel("Time step")
        plt.ylabel(f"x{i+1}")
        plt.title(f"x{i+1} over time")

        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{figdir}/time_series_idx{traj_index}.png")
    plt.close()


def plot_phase_space(X_true, X_hat, system, figdir, model_name, traj_index):
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

        plt.figure(figsize=(6, 6))

        plt.plot(X_true[:, 0], X_true[:, 1], label="True")
        plt.plot(X_hat[:, 0], X_hat[:, 1], "--", label="Prediction")

        plt.xlabel("x1")
        plt.ylabel("x2")

        plt.title(f"Phase space rollout ({model_name}_{system})")

        plt.legend()
        plt.tight_layout()

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

        plt.suptitle(f"Phase space rollout ({model_name}_{system})")

        plt.tight_layout()

        plt.savefig(f"{figdir}/rollout_idx{traj_index}.png")
        plt.close()

    # --------------------------------------------------
    # fallback for higher dimensions
    # --------------------------------------------------
    else:

        plt.figure(figsize=(6, 6))

        plt.plot(X_true[:, 0], X_true[:, 1], label="True")
        plt.plot(X_hat[:, 0], X_hat[:, 1], "--", label="Prediction")

        plt.xlabel("x1")
        plt.ylabel("x2")

        plt.title(f"Phase space projection ({model_name}_{system})")

        plt.legend()
        plt.tight_layout()

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
):
    """
    Plot phase space with optional clean/reference trajectory overlay.
    Currently intended mainly for 2D systems.
    """
    state_dim = X_true.shape[1]

    if state_dim < 2:
        return

    plt.figure(figsize=(6, 6))

    if X_ref is not None:
        plt.plot(X_ref[:, 0], X_ref[:, 1], label=ref_label, linewidth=2)

    plt.plot(X_true[:, 0], X_true[:, 1], label=true_label, alpha=0.6)
    plt.plot(X_hat[:, 0], X_hat[:, 1], "--", label="Prediction", linewidth=2)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Phase space rollout ({model_name}_{system})")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{figdir}/rollout_overlay_idx{traj_index}.png", dpi=200)
    plt.close()