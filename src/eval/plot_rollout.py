import matplotlib.pyplot as plt
import os


def plot_time_series(X_true, X_hat, figdir, traj_index):
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

    state_dim = X_true.shape[1]

    if system == "lorenz" and state_dim >= 3:
        i, j = 0, 2
    else:
        i, j = 0, 1

    plt.figure(figsize=(6, 6))

    plt.plot(X_true[:, i], X_true[:, j], label="True")
    plt.plot(X_hat[:, i], X_hat[:, j], "--", label="Prediction")

    plt.xlabel(f"x{i+1}")
    plt.ylabel(f"x{j+1}")

    plt.title(f"Phase space rollout ({model_name}_{system})")

    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{figdir}/rollout_idx{traj_index}.png")

    plt.close()