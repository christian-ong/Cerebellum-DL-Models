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
* Bar or Pie: Each mode's contribution to state reconstruction. Would this be its ability to reconstruct trajectories on its own?
"""

def build_model_from_checkpoint(ckpt):
    model_name = ckpt.get("model", "ml_dmd")
    train_args = ckpt["train_args"]
    
    # Common kwargs for both model types
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
    Extracts math: 
    Left W -> Eigenfunctions (Linearizing Coordinates)
    Right V -> Modes (Physical Patterns)
    """
    if hasattr(model, "Phi"):
        Phi = model.get_Phi_true().detach().numpy()
        Lambda = model.get_Lambda().detach().numpy()
        K = model.get_K_true().detach().numpy()
        eigvals, V_inner = np.linalg.eig(Lambda)
        _, W_inner = np.linalg.eig(Lambda.T)
        V = Phi @ V_inner
        W = W_inner  # Project onto lifted states directly
        return Phi, Lambda, eigvals, V, W, K
    
    raise ValueError("Model format not recognized for eigensystem extraction.")


def plot_transition_matrices(matrices, title, expansion_names):
    num_rows = int(np.ceil(len(matrices) / 2))
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))
    fig.suptitle(f"Koopman Operator Matrices, {system_name}", fontsize=16)
    for ax, (M, title) in zip(axes.flat, matrices):
        M_real = np.real(M)
        im = ax.imshow(M_real)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")
        ax.set_xticks(range(len(expansion_names)))
        ax.set_xticklabels(expansion_names, rotation=70, fontsize=6)
        ax.set_yticks(range(len(expansion_names)))
        ax.set_yticklabels(expansion_names, fontsize=6)
        

        # Annotate cell values
        for (i, j), v in np.ndenumerate(M):
            if abs(v) > 1e-3:  # Only annotate significant values
                ax.text(
                    j, i, f"{v:.3f}",
                    ha="center", va="center",
                    rotation=20, fontsize=7, color="red"
                )

    # hide unused subplot
    for i in range(len(matrices), axes.shape[0] * axes.shape[1]):
        axes.flat[i].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.13, top=0.9, hspace=0.43)
    plt.show()


def plot_complex_field(points, values, title, cmap="inferno"):
    """Plots 1x4 Real, Imag, Mag, Phase for a single mode."""
    grid_n = int(np.sqrt(len(points)))
    extent = [points[:,0].min(), points[:,0].max(), points[:,1].min(), points[:,1].max()]
    num_modes = values.shape[-1]

    fig, axes = plt.subplots(num_modes, 4, figsize=(10,10))
    for mode_idx in range(num_modes):
        data_map = {
            "Real": np.real(values[:, mode_idx]), "Imag": np.imag(values[:, mode_idx]),
            "Mag": np.abs(values[:, mode_idx]), "Phase": np.angle(values[:, mode_idx])
        }
    
        for i, (label, data) in enumerate(data_map.items()):
            ax = axes[mode_idx, i]
            # curr_cmap = "twilight" if label == "Phase" else cmap
            curr_cmap = cmap
            im = ax.imshow(data.reshape(grid_n, grid_n), extent=extent, origin="lower", cmap=curr_cmap)
            ax.set_title(f"{label}")
            plt.colorbar(im, ax=ax) #fraction=0.046, pad=0.04
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def calculate_mode_quality(model, W, eigvals, z_scale, grid_bounds, n_modes_to_keep=8):
    """
    Evaluates Koopman eigenfunctions (Left Eigenvectors) based on:
    1. Temporal Residual (how linear is the evolution?)
    2. Stability (is it near the unit circle?)
    3. Spatial Variance (is it non-trivial?)
    """
    n_eval_traj = 14
    eval_steps = 90
    grid_min, grid_max = grid_bounds
    
    rng = np.random.default_rng(0)
    num_modes = W.shape[1]
    residual_accum = np.zeros(num_modes, dtype=np.float64)

    # 1. Compute Temporal Residuals via Trajectory Rollouts
    for _ in range(n_eval_traj):
        x0 = rng.uniform(grid_min, grid_max, size=(1, model.state_dim)).astype(np.float32)
        xt = torch.tensor(x0, dtype=torch.float32)

        traj = [x0.flatten()]
        with torch.no_grad():
            for _ in range(eval_steps):
                xt = model(xt)
                traj.append(xt.cpu().numpy().flatten())

        traj = np.asarray(traj, dtype=np.float32)
        with torch.no_grad():
            # Lift and project onto Left Eigenvectors
            z_roll = model.expand(torch.tensor(traj)).cpu().numpy() / z_scale
            phi_roll = z_roll @ W

        # Linear evolution check: phi(t+1) - lambda * phi(t)
        lhs = phi_roll[1:, :] # phi at next time step
        rhs = phi_roll[:-1, :] * eigvals[None, :] # predict next phi

        num = np.linalg.norm(lhs - rhs, axis=0) # error magnitude for each mode
        den = np.linalg.norm(phi_roll[:-1, :], axis=0) + 1e-12 # normalize
        residual_accum += num / den

    residual_mean = residual_accum / max(1, n_eval_traj)

    # 2. Compute Spatial Score (The part I missed)
    # We need a grid of points to check spatial variation
    x_range = np.linspace(grid_min, grid_max, 50)
    X_grid, Y_grid = np.meshgrid(x_range, x_range)
    pts = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    with torch.no_grad():
        z_grid = model.expand(torch.as_tensor(pts, dtype=torch.float32)).cpu().numpy() / z_scale
        phi_grid = z_grid @ W
    
    spatial_std = np.std(np.real(phi_grid), axis=0)

    # 3. Compute Stability Score
    # stability = np.exp(-np.abs(np.abs(eigvals) - 1.0) / 0.15)
    
    def to_unit(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-12)

    # 4. Final Weighted Scoring
    res_score = 1.0 - to_unit(residual_mean)
    # stab_score = to_unit(stability)
    spat_score = to_unit(spatial_std)
    
    # Weights optimized for identifying governing equations
    mode_score = (
        0.8 * res_score + 
        0.2 * spat_score
        # 0.0 * stab_score
    )
    
    ranked_indices = np.argsort(mode_score)[::-1]
    return ranked_indices[:n_modes_to_keep], mode_score, residual_mean


def plot_quality_spectrum(eigvals, mode_scores, theme="dark"):
    """
    Plots the eigenvalue spectrum colored by the quality score.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 1. Aesthetics based on your preference
    point_color = "#f9fafb" if theme == "dark" else "#111827"
    circle_color = "#9ca3af" if theme == "dark" else "#374151"
    
    # 2. Color and Size based on Quality
    # We use a colormap (e.g., 'viridis' or 'plasma') to map score to color
    scatter = ax.scatter(
        eigvals.real, 
        eigvals.imag, 
        c=mode_scores, 
        cmap='viridis', 
        s=40 + (mode_scores * 60), # Better modes are slightly larger
        edgecolors='white',
        linewidths=0.5,
        alpha=0.9,
        zorder=3
    )
    
    # 3. Reference Unit Circle
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "--", color=circle_color, linewidth=1.2, zorder=1)
    
    # 4. Axes and Labels
    ax.axhline(0, color=circle_color, linewidth=0.8, alpha=0.5)
    ax.axvline(0, color=circle_color, linewidth=0.8, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Eigenvalue Spectrum (Colored by Quality)", fontsize=12)
    ax.set_xlabel("$\mathbb{R}(\lambda)$")
    ax.set_ylabel("$\mathbb{I}(\lambda)$")
    
    # Add a colorbar to explain the quality scale
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mode Quality Score', rotation=270, labelpad=15)
    
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_freq_magnitude(eigvals, mode_scores, theme="dark"):
    """
    Plots Frequency (Phase) vs. Magnitude of eigenvalues, 
    colored by their quality score.
    """
    # 1. Calculate Magnitude and Frequency
    # Magnitude: Growth/Decay rate
    magnitudes = np.abs(eigvals)
    
    # Frequency: Phase angle of the complex eigenvalue
    # We take the absolute angle and normalize it by pi for easy reading
    frequencies = np.abs(np.angle(eigvals)) / np.pi 
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 2. Aesthetics
    circle_color = "#9ca3af" if theme == "dark" else "#374151"
    
    # 3. Scatter Plot colored by Quality
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

    # Add point labels
    x_spread = (frequencies.max() - frequencies.min())
    y_spread = (magnitudes.max() - magnitudes.min())
    plot_checklist = np.zeros(len(mode_scores), dtype=bool)
    for i, (freq, mag) in enumerate(zip(frequencies, magnitudes)):
        if plot_checklist[i]:
            continue
        else:
            # if nearby point, label both in same text to avoid overlap
            overlap_th = 0.02
            nearby_indices = np.where((np.abs(frequencies - freq) < overlap_th * x_spread) & (np.abs(magnitudes - mag) < overlap_th * y_spread))[0]
            label_string = f"{i}"
            for near_idx in nearby_indices:
                if near_idx != i and not plot_checklist[near_idx]:
                    label_string += f", {near_idx}"
                    plot_checklist[near_idx] = True
            ax.text(freq, mag+overlap_th * y_spread, label_string, fontsize=8, ha='center', va='center')
            plot_checklist[i] = True
    
    # 4. Reference lines
    ax.axhline(1.0, color=circle_color, linestyle='--', alpha=0.6, label="Unit Circle (Stable)")
    
    ax.set_title("Eigenvalue Distribution: Frequency vs. Magnitude", fontsize=14)
    ax.set_xlabel("Normalized Frequency ($\omega / \pi$)")
    ax.set_ylabel("Magnitude ($|\lambda|$)")
    
    # Colorbar for context
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mode Quality Score')
    
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_mode_trajectories(model, W, eigvals, z_scale, best_ids, real_traj):
    """
    Plots the temporal evolution of the top eigenfunctions.
    Compares:
    1. Real Data (projected onto eigenfunctions)
    2. Model Prediction (iterative NN rollout)
    3. Theoretical Linear Evolution (lambda^t)
    """
    n_steps = real_traj.shape[0]
    t = np.arange(n_steps)
    
    # 1. Project Real Data into Koopman Space
    with torch.no_grad():
        z_real = model.expand(torch.as_tensor(real_traj, dtype=torch.float32)).cpu().numpy() / z_scale
        phi_real_traj = z_real @ W
    
    # 2. Generate NN Rollout (starting from the same x0)
    xt = torch.as_tensor(real_traj[0:1, :], dtype=torch.float32)
    phi_nn_traj = []
    with torch.no_grad():
        for _ in range(n_steps):
            z_t = model.expand(xt).cpu().numpy() / z_scale
            phi_nn_traj.append(z_t @ W)
            xt = model(xt)
    phi_nn_traj = np.array(phi_nn_traj).squeeze()
    
    # 3. Rank modes to select which to plot
    # best_ids, _, _ = calculate_mode_quality(model, W, eigvals, z_scale, grid_bounds, n_modes_to_keep)
    
    fig, axes = plt.subplots(len(best_ids), 1, figsize=(12, 2.5 * len(best_ids)), sharex=True)
    if len(best_ids) == 1: axes = [axes]

    for i, m_idx in enumerate(best_ids):
        ax = axes[i]
        lam = eigvals[m_idx]
        phi0 = phi_real_traj[0, m_idx]
        
        # 4. Theoretical linear evolution: phi0 * lambda^t
        phi_theory = phi0 * (lam ** t)
        
        # Plotting the three comparisons
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


def plot_mode_contributions_vs_quality(V, phi_traj, best_ids, scores, n_top=10):
    """
    Plots the physical energy of each mode.
    Energy = Mean Activation (|phi|) * Mode Norm (||v||).
    """
    # 1. Calculate the mean activation (Amplitude) over the trajectory
    # phi_traj shape is (time_steps, num_modes)
    amplitudes = np.mean(np.abs(phi_traj), axis=0)
    
    # 2. Calculate the norm of the Koopman modes
    # (If V was unit-normalized, this part will be 1.0)
    v_norms = np.linalg.norm(V, axis=0)
    
    # 3. Combined Physical Energy
    mode_energies = amplitudes * v_norms
    
    # 4. Normalize to show relative percentage
    total_energy = np.sum(mode_energies)
    relative_contribution = (mode_energies / total_energy) * 100
    
    # 5. Sort and Plot
    energy_sort_idx = np.argsort(relative_contribution)[::-1]
    top_energy_idx = energy_sort_idx[:n_top]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(scores[top_energy_idx])
    
    bars = ax.bar(range(n_top), relative_contribution[top_energy_idx], color=colors)
    
    ax.set_xticks(range(n_top))
    ax.set_xticklabels([f"Mode {i}" for i in top_energy_idx], rotation=45)
    ax.set_ylabel("Contribution to Reconstruction (%)")
    ax.set_title("Physical Mode Energy (Weighted by Average Activation)")
    
    # Add quality labels
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.2, 
                f"Q:{scores[top_energy_idx[i]]:.2f}", ha='center', va='bottom', fontsize=8)

    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label="Quality Score")
    plt.tight_layout()
    plt.show()


def plot_mode_energy_vs_quality(V, scores, n_top=20):
    """
    Plots Quality (Mode Score) vs. Energy (Contribution).
    Used to identify governing modes vs. numerical noise.
    """
    # 1. Calculate Energy (Norm of Right Eigenvectors)
    mode_energies = np.linalg.norm(V, axis=0)
    
    # Normalize energy for better scale (0 to 1 range for the plot)
    energy_norm = (mode_energies - mode_energies.min()) / (mode_energies.max() - mode_energies.min() + 1e-12)

    fig, ax = plt.subplots(figsize=(10, 7))

    # 2. Scatter Plot
    # x-axis: Quality (Scores)
    # y-axis: Energy (Normalized Contribution)
    scatter = ax.scatter(
        scores, 
        energy_norm, 
        # c=scores, 
        # cmap='viridis', 
        s=100, 
        alpha=0.7, 
        edgecolors='black',
        zorder=3
    )

    # 3. Label the top N modes for easy identification
    top_indices = np.argsort(mode_energies)[::-1][:n_top]
    plot_checklist = np.zeros(len(scores), dtype=bool)
    for idx in top_indices:
        if plot_checklist[idx]:
            continue
        else:
            # if nearby point, label both in same text to avoid overlap
            nearby_indices = np.where(np.abs(scores - scores[idx]) < 0.001)[0]
            label_string = f"Mode {idx}"
            for near_idx in nearby_indices:
                if near_idx != idx and not plot_checklist[near_idx]:
                    label_string += f", {near_idx}"
                    plot_checklist[near_idx] = True
                
            ax.text(scores[idx], energy_norm[idx]+0.02, label_string, fontsize=8, ha='center', va='center', zorder=4)
            plot_checklist[idx] = True

    # 4. Quadrant Lines (at medians or 0.5)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.3)

    # # 5. Colorbar for Quality Scores
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.set_label('Mode Quality Score', rotation=270, labelpad=15)

    # Annotate Quadrants for Interpretation
    ax.text(0.75, 0.9, "Governing Modes", fontsize=12, fontweight='bold', alpha=0.5, ha='center')
    ax.text(0.75, 0.1, "Math Harmonics", fontsize=12, alpha=0.5, ha='center')
    ax.text(0.25, 0.9, "Overfitted Noise", fontsize=12, alpha=0.5, ha='center')
    ax.text(0.25, 0.1, "Junk Modes", fontsize=12, alpha=0.5, ha='center')

    ax.set_xlabel("Quality Score (Linearity & Stability)", fontsize=12)
    ax.set_ylabel("Normalized Energy (Contribution to $x$)", fontsize=12)
    ax.set_title("Mode Selection: Quality vs. Physical Energy", fontsize=14)
    
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Execution
# --------------------------------------------------

# Settings
model_name = "ml_dmd"
system_name = "vanderpol"
custom_name = "vanderpol_gen5"
grid_res = 100
n_top_modes = 8

# Load model and eigensystem
model_path = f"data/models/{model_name}/{system_name}/{custom_name}/model.pt"
ckpt = torch.load(model_path, map_location="cpu")
model, model_type = build_model_from_checkpoint(ckpt)
z_scale = model.z_scale.detach().cpu().numpy()
Phi, Lambda, eigvals, V, W, K = get_koopman_eigensystem(model)

# Grid setup
x_range = np.linspace(-2, 2, grid_res)
X, Y = np.meshgrid(x_range, x_range)
grid_points = np.column_stack([X.ravel(), Y.ravel()])
with torch.no_grad():
    z = model.expand(torch.as_tensor(grid_points, dtype=torch.float32)).cpu().numpy() / z_scale

print(Phi.shape)
print(eigvals.shape)

# Calculate eigenvector qualities using the Left Eigenvectors
best_ids, scores, residuals = calculate_mode_quality(
    model, W, eigvals, z_scale, grid_bounds=(-2.0, 2.0))

# print(f"Top mode found: Index {best_ids[0]} with score {scores[best_ids[0]]:.3f}")
num_modes = len(eigvals)
print(f"Total modes: {num_modes}")
print(f"Top {n_top_modes} modes indices: {best_ids[:n_top_modes]}")
print(f"Top {n_top_modes} modes scores: {[f'{s:.3f}' for s in scores[best_ids[:n_top_modes]]]}")
top_n_modes = best_ids[:n_top_modes]


# --------------------------------------------------
# Plot Koopman Operator Matrices
# --------------------------------------------------
matrices = [
    (Phi, "$\Phi$"),
    (Lambda, "$\Lambda$"),
    (K, r"K = $\Phi \Lambda \Phi^{-1}$")]
plot_transition_matrices(matrices, f"Koopman Operator Matrices, {system_name}", model.expand_names)

# --------------------------------------------------
# Plot eigenfunctions and modes (top N modes)
# --------------------------------------------------
phi_vals = z @ W[:, top_n_modes] # Left view (Eigenfunction)
v_vals = z @ V[:, top_n_modes] # Right view (Mode)
plot_complex_field(grid_points, phi_vals, f"Eigenfunctions {top_n_modes} (Score: {[f'{e:.1f}' for e in scores[top_n_modes]]})")
plot_complex_field(grid_points, v_vals, f"Physical Mode {top_n_modes} (Score: {[f'{e:.3f}' for e in scores[top_n_modes]]})")

# --------------------------------------------------
# Spectrum plot with quality coloring
# --------------------------------------------------
best_ids, scores, residuals = calculate_mode_quality(model, W, eigvals, z_scale, (-2.0, 2.0))
plot_quality_spectrum(eigvals, scores, theme="dark")

# --------------------------------------------------
# Visualize eigenvalue frequencies vs magnitudes
# --------------------------------------------------
plot_freq_magnitude(eigvals, scores, theme="dark")

# --------------------------------------------------
# Visualize mode trajectories (how each modes evolves over time, random initial condition)
# --------------------------------------------------
# Load test trajectories
data_path = f"data/trajectories/linear/{system_name}/test.npz" if system_name in ["degenerate_node", "harmonic_oscillator", "inward_spiral", "saddle_point"] else f"data/trajectories/nonlinear/{system_name}/test.npz"
data = np.load(data_path)
trajectories = data['X'] # Shape: (trajectory_length, num_trajectories, state_dim)
single_trajectory = trajectories[:, 0, :] # Shape: (trajectory_length, state_dim)
plot_mode_trajectories(model, W, eigvals, z_scale, best_ids[:n_top_modes], single_trajectory)

# --------------------------------------------------
# Visualize mode contributions to state reconstruction
# --------------------------------------------------
# which modes are actually contributing to the dynamics? Weight the mode norms by actual activations (phi) from the real trajectory
with torch.no_grad(): # Project the real trajectory to get activations
    z_real = model.expand(torch.as_tensor(single_trajectory)).cpu().numpy() / z_scale
    phi_real_traj = z_real @ W
plot_mode_contributions_vs_quality(V, phi_real_traj, best_ids, scores, n_top=10) # contributions based on actual signal strength

# --------------------------------------------------
# Visualize mode energy vs quality
# --------------------------------------------------
plot_mode_energy_vs_quality(V, scores, n_top=10)