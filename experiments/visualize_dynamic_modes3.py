import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.ml_dmd import ML_DMD
from src.models.ml_linear_dynamics import ML_LinearDynamics

"""
Plots
* Koopman Operator Matrices
* Eigenfunctions and modes (top N modes)
* Spectrum plot with quality coloring
* Scatter: Frequency vs Magnitude of each eigenvalue
* Trajectories: How each mode evolves over time with random initial conditions
* Bar or Pie: Each mode's contribution to state reconstruction.
"""

def build_model_from_checkpoint(ckpt):
    model_name = ckpt.get("model", "ml_dmd")
    train_args = ckpt["train_args"]
    
    kwargs = {
        "state_dim": ckpt["state_dim"],
        "expansion_degree": train_args["expansion_degree"],
        "bias": str(train_args.get("bias", "true")).lower() == "true",
        "sine_cosine_expansion": str(train_args.get("sine_cosine_expansion", "false")).lower() == "true",
        "expansion_type": train_args["expansion_type"],
        "system": ckpt["system"],
    }

    if model_name == "ml_dmd":
        model = ML_DMD(**kwargs)
    elif model_name == "ml_lineardynamics":
        model = ML_LinearDynamics(**kwargs)
    else:
        raise ValueError(f"Unsupported: {model_name}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_name


def get_koopman_eigensystem(model):
    """
    Extracts the Koopman modes and eigenfunctions STRICTLY from 
    the neural network's learned parameters (Phi and Lambda).
    """
    if hasattr(model, "Phi"):
        Phi_true = model.get_Phi_true().detach().numpy()
        Phi_scaled = model.Phi.detach().numpy()
        Lambda = model.get_Lambda().detach().numpy()
        K_true = model.get_K_true().detach().numpy()
        
        eigvals, V_inner = np.linalg.eig(Lambda)
        _, W_inner = np.linalg.eig(Lambda.T) 
        
        V = Phi_true @ V_inner
        
        v_norms = np.linalg.norm(V, axis=0)
        V = V / v_norms
        
        if hasattr(model, "Phi_inv"):
            Phi_scaled_inv = model.Phi_inv.detach().numpy()
        else:
            Phi_scaled_inv = np.linalg.pinv(Phi_scaled)
            
        Phi_scaled_inv_T = Phi_scaled_inv.T
        W = Phi_scaled_inv_T @ W_inner
        
        W = W * v_norms
        
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
    
    raise ValueError("Model format not recognized for eigensystem extraction.")


def plot_transition_matrices(matrices, title, expansion_names):
    num_rows = int(np.ceil(len(matrices) / 2))
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))
    fig.suptitle(title, fontsize=16)
    for ax, (M, subtitle) in zip(axes.flat, matrices):
        M_mag = np.abs(M) 
        im = ax.imshow(M_mag)
        ax.set_title(subtitle)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")
        ax.set_xticks(range(len(expansion_names)))
        ax.set_xticklabels(expansion_names, rotation=70, fontsize=6)
        ax.set_yticks(range(len(expansion_names)))
        ax.set_yticklabels(expansion_names, fontsize=6)

        for (i, j), v in np.ndenumerate(M):
            if abs(v) > 1e-3:
                ax.text(
                    j, i, f"{v:.3f}",
                    ha="center", va="center",
                    rotation=20, fontsize=7, color="red"
                )

    for i in range(len(matrices), axes.shape[0] * axes.shape[1]):
        axes.flat[i].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.13, top=0.9, hspace=0.43)
    plt.show()


def plot_complex_field(points, values, title, cmap="inferno"):
    grid_n = int(np.sqrt(len(points)))
    extent = [points[:,0].min(), points[:,0].max(), points[:,1].min(), points[:,1].max()]
    num_modes = values.shape[-1]

    fig, axes = plt.subplots(num_modes, 4, figsize=(10,10))
    if num_modes == 1: axes = np.expand_dims(axes, 0)
        
    for mode_idx in range(num_modes):
        data_map = {
            "Real": np.real(values[:, mode_idx]), "Imag": np.imag(values[:, mode_idx]),
            "Mag": np.abs(values[:, mode_idx]), "Phase": np.angle(values[:, mode_idx])
        }
    
        for i, (label, data) in enumerate(data_map.items()):
            ax = axes[mode_idx, i]
            im = ax.imshow(data.reshape(grid_n, grid_n), extent=extent, origin="lower", cmap=cmap, aspect='auto')
            ax.set_title(f"{label}")
            plt.colorbar(im, ax=ax)
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def calculate_mode_quality(model, W, eigvals, z_scale, state_bounds, n_modes_to_keep=8):
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
            z_roll = model.expand(torch.tensor(traj)).cpu().numpy() / z_scale
            phi_roll = z_roll @ W

        lhs = phi_roll[1:, :] 
        rhs = phi_roll[:-1, :] * eigvals[None, :] 

        num = np.linalg.norm(lhs - rhs, axis=0) 
        den = np.linalg.norm(phi_roll[:-1, :], axis=0) + 1e-12 
        residual_accum += num / den

    residual_mean = residual_accum / max(1, n_eval_traj)

    x_range = np.linspace(state_bounds[0][0], state_bounds[0][1], 50)
    y_range = np.linspace(state_bounds[1][0], state_bounds[1][1], 50)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    pts = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    with torch.no_grad():
        z_grid = model.expand(torch.as_tensor(pts, dtype=torch.float32)).cpu().numpy() / z_scale
        phi_grid = z_grid @ W
    
    spatial_std = np.std(np.real(phi_grid), axis=0)

    def to_unit(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-12)

    res_score = 1.0 - to_unit(residual_mean)
    spat_score = to_unit(spatial_std)
    
    mode_score = 0.8 * res_score + 0.2 * spat_score
    ranked_indices = np.argsort(mode_score)[::-1]
    
    return ranked_indices[:n_modes_to_keep], mode_score, residual_mean


def plot_quality_spectrum(eigvals, mode_scores, theme="dark"):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    circle_color = "#9ca3af" if theme == "dark" else "#374151"
    
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
    
    ax.axhline(0, color=circle_color, linewidth=0.8, alpha=0.5)
    ax.axvline(0, color=circle_color, linewidth=0.8, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Eigenvalue Spectrum (Colored by Quality)", fontsize=12)
    ax.set_xlabel("$\mathbb{R}(\lambda)$")
    ax.set_ylabel("$\mathbb{I}(\lambda)$")
    
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mode Quality Score', rotation=270, labelpad=15)
    
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_freq_magnitude(eigvals, mode_scores, theme="dark"):
    magnitudes = np.abs(eigvals)
    frequencies = np.abs(np.angle(eigvals)) / np.pi 
    
    fig, ax = plt.subplots(figsize=(8, 5))
    circle_color = "#9ca3af" if theme == "dark" else "#374151"
    
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

    x_spread = (frequencies.max() - frequencies.min())
    y_spread = (magnitudes.max() - magnitudes.min())
    plot_checklist = np.zeros(len(mode_scores), dtype=bool)
    for i, (freq, mag) in enumerate(zip(frequencies, magnitudes)):
        if plot_checklist[i]:
            continue
        overlap_th = 0.02
        nearby_indices = np.where((np.abs(frequencies - freq) < overlap_th * x_spread) & (np.abs(magnitudes - mag) < overlap_th * y_spread))[0]
        label_string = f"{i}"
        for near_idx in nearby_indices:
            if near_idx != i and not plot_checklist[near_idx]:
                label_string += f", {near_idx}"
                plot_checklist[near_idx] = True
        ax.text(freq, mag+overlap_th * y_spread, label_string, fontsize=8, ha='center', va='center')
        plot_checklist[i] = True
    
    ax.axhline(1.0, color=circle_color, linestyle='--', alpha=0.6, label="Unit Circle (Stable)")
    
    ax.set_title("Eigenvalue Distribution: Frequency vs. Magnitude", fontsize=14)
    ax.set_xlabel("Normalized Frequency ($\omega / \pi$)")
    ax.set_ylabel("Magnitude ($|\lambda|$)")
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mode Quality Score')
    
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_mode_trajectories(model, W, eigvals, z_scale, best_ids, real_traj):
    n_steps = real_traj.shape[0]
    t = np.arange(n_steps)
    
    with torch.no_grad():
        z_real = model.expand(torch.as_tensor(real_traj, dtype=torch.float32)).cpu().numpy() / z_scale
        phi_real_traj = z_real @ W
    
    xt = torch.as_tensor(real_traj[0:1, :], dtype=torch.float32)
    phi_nn_traj = []
    with torch.no_grad():
        for _ in range(n_steps):
            z_t = model.expand(xt).cpu().numpy() / z_scale
            phi_nn_traj.append(z_t @ W)
            xt = model(xt)
    phi_nn_traj = np.array(phi_nn_traj).squeeze()
    
    fig, axes = plt.subplots(len(best_ids), 1, figsize=(12, 2.5 * len(best_ids)), sharex=True)
    if len(best_ids) == 1: axes = [axes]

    for i, m_idx in enumerate(best_ids):
        ax = axes[i]
        lam = eigvals[m_idx]
        phi0 = phi_real_traj[0, m_idx]
        
        phi_theory = phi0 * (lam ** t)
        
        ax.plot(t, phi_real_traj[:, m_idx].real, 'k-', alpha=0.3, label='Real Data (Ground Truth)', linewidth=3)
        ax.plot(t, phi_nn_traj[:, m_idx].real, 'o', markersize=2, label='Model Prediction (NN Rollout)', alpha=0.7)
        ax.plot(t, phi_theory.real, '--', color='red', label='Theoretical Linear ($\lambda^t$)', linewidth=1.5)
        
        ax.set_ylabel(f"$\phi_{{{m_idx}}}(x)$")
        ax.set_title(f"Mode {m_idx} | $\lambda = {lam.real:.3f} + {lam.imag:.3f}j$", fontsize=10, loc='right')
        if i == 0:
            ax.legend(loc='upper right', ncol=3, fontsize='small', frameon=True)

    axes[-1].set_xlabel("Time Steps")
    fig.suptitle("Koopman Mode Evolution: Reality vs. Model vs. Theory", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_mode_contributions_vs_quality(V, phi_traj, best_ids, scores, state_dim, n_top=10):
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
    plt.show()


def plot_mode_energy_vs_quality(V, scores, state_dim, n_top=20):
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
    plt.show()


# --------------------------------------------------
# Execution
# --------------------------------------------------

if __name__ == "__main__":
    # Settings
    model_name = "ml_dmd"
    system_name = "closed_large"
    custom_name = "spec5_new"
    grid_res = 100
    n_top_modes = 5
    
    # 1. Load test trajectories FIRST to find true boundaries
    data_path = f"data/trajectories/linear/{system_name}/test.npz" if system_name in ["degenerate_node", "harmonic_oscillator", "inward_spiral", "saddle_point"] else f"data/trajectories/nonlinear/{system_name}/test.npz"
    data = np.load(data_path)
    trajectories = data['X'] 
    single_trajectory = trajectories[:, 0, :] 
    
    # Find min and max dynamically
    x_min, x_max = trajectories[:, :, 0].min(), trajectories[:, :, 0].max()
    y_min, y_max = trajectories[:, :, 1].min(), trajectories[:, :, 1].max()
    state_bounds = [(x_min, x_max), (y_min, y_max)]
    
    # 2. Load model and eigensystem
    model_path = f"data/models/{model_name}/{system_name}/{custom_name}/model_best.pt"
    ckpt = torch.load(model_path, map_location="cpu")
    model, model_type = build_model_from_checkpoint(ckpt)
    z_scale = model.z_scale.detach().cpu().numpy()
    Phi, Lambda, eigvals, V, W, K = get_koopman_eigensystem(model)
    
    # 3. Dynamic Grid setup based on true boundaries
    x_range = np.linspace(x_min, x_max, grid_res)
    y_range = np.linspace(y_min, y_max, grid_res)
    X, Y = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    with torch.no_grad():
        z = model.expand(torch.as_tensor(grid_points, dtype=torch.float32)).cpu().numpy() / z_scale
    
    print("Phi shape:", Phi.shape)
    print("Eigvals shape:", eigvals.shape)
    
    # 4. Calculate mode qualities using true boundaries
    best_ids, scores, residuals = calculate_mode_quality(
        model, W, eigvals, z_scale, state_bounds=state_bounds)
    
    num_modes = len(eigvals)
    print(f"Total modes: {num_modes}")
    print(f"Top {n_top_modes} modes indices: {best_ids[:n_top_modes]}")
    print(f"Top {n_top_modes} modes scores: {[f'{s:.3f}' for s in scores[best_ids[:n_top_modes]]]}")
    top_n_modes = best_ids[:n_top_modes]
    
    # --------------------------------------------------
    # Plot Koopman Operator Matrices
    # --------------------------------------------------
    matrices = [
        (V, "Extracted Koopman Modes (V)"),
        (np.diag(eigvals), "True Complex $\Lambda$"),
        (K, r"Operator K")]
    plot_transition_matrices(matrices, f"Koopman Operator Matrices, {system_name}", model.expand_names)
    
    # --------------------------------------------------
    # Plot eigenfunctions (top N modes)
    # --------------------------------------------------
    phi_vals = z @ W[:, top_n_modes] 
    plot_complex_field(grid_points, phi_vals, f"Eigenfunctions {top_n_modes} (Score: {[f'{e:.1f}' for e in scores[top_n_modes]]})")
    
    # --------------------------------------------------
    # Spectrum plot with quality coloring
    # --------------------------------------------------
    best_ids, scores, residuals = calculate_mode_quality(model, W, eigvals, z_scale, state_bounds=state_bounds)
    plot_quality_spectrum(eigvals, scores, theme="dark")
    
    # --------------------------------------------------
    # Visualize eigenvalue frequencies vs magnitudes
    # --------------------------------------------------
    plot_freq_magnitude(eigvals, scores, theme="dark")
    
    # --------------------------------------------------
    # Visualize mode trajectories
    # --------------------------------------------------
    plot_mode_trajectories(model, W, eigvals, z_scale, best_ids[:n_top_modes], single_trajectory)
    
    # --------------------------------------------------
    # Visualize mode contributions to state reconstruction
    # --------------------------------------------------
    with torch.no_grad():
        z_real = model.expand(torch.as_tensor(single_trajectory)).cpu().numpy() / z_scale
        phi_real_traj = z_real @ W
    plot_mode_contributions_vs_quality(V, phi_real_traj, best_ids, scores, model.state_dim, n_top=10) 
    
    # --------------------------------------------------
    # Visualize mode energy vs quality
    # --------------------------------------------------
    plot_mode_energy_vs_quality(V, scores, model.state_dim, n_top=10)