import os
import argparse
import torch
import numpy as np
from src.data_generation.load_data import resolve_split_npz_path
from src.eval.visualize_modes import *

"""
This script visualizes the dynamic modes and eigensystem of a trained Koopman model.


python -m experiments.visualize_dynamic_modes --model_path data\sweeped_models\local2\ml_dmd\closed_trig_large\free_long_spec10\model_best.pt --data_path data/trajectories/nonlinear/closed_trig_large/long/test.npz

"""


# --------------------------------------------------
# Execution
# --------------------------------------------------

parser = argparse.ArgumentParser(description="Visualize Koopman dynamic modes and eigensystems.")
parser.add_argument("--model_path", type=str, help="Path to the trained model checkpoint (overrides model_name and custom_name if provided)")
parser.add_argument("--custom_name", type=str, default="default", help="Custom name of the trained model")
parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
args = parser.parse_args()

# 1. Load test trajectories to find true boundaries and system name dynamically
test_data_path = resolve_split_npz_path(args.data_path, "test")
data = np.load(test_data_path)
trajectories = data['X'] 
system = str(data["system"])
state_dim = trajectories.shape[-1]

print(f"Visualizing for System: {system}")

# Find min and max dynamically for ALL dimensions
state_bounds = []
for dim in range(state_dim):
    dim_min, dim_max = trajectories[:, :, dim].min(), trajectories[:, :, dim].max()
    state_bounds.append((dim_min, dim_max))

# 2. Load model and eigensystem
model, model_type = build_model_from_checkpoint(args.model_path)
z_scale = model.z_scale.detach().cpu().numpy()
Phi_true, Lambda, eigvals, V, W, K = get_koopman_eigensystem(model)

# 3. Dynamic Grid setup based on true boundaries (taking 2D slice for 3D+ systems)
grid_res = 100
n_top_modes = 5
x_range = np.linspace(state_bounds[0][0], state_bounds[0][1], grid_res)
y_range = np.linspace(state_bounds[1][0], state_bounds[1][1], grid_res)
X, Y = np.meshgrid(x_range, y_range)
grid_cols = [X.ravel(), Y.ravel()]

# Pad higher dimensions with their mean trajectory values
for dim in range(2, state_dim):
    dim_mean = trajectories[:, :, dim].mean()
    grid_cols.append(np.full_like(X.ravel(), dim_mean))
grid_points = np.column_stack(grid_cols)

with torch.no_grad():
    z = model.expand(torch.as_tensor(grid_points, dtype=torch.float32)).cpu().numpy() / z_scale

print("Phi shape:", Phi_true.shape)
print("Eigvals shape:", eigvals.shape)

# 4. Calculate mode qualities using true boundaries
best_ids, scores, residuals = calculate_mode_quality(
    model, W, eigvals, z_scale, state_bounds=state_bounds)

num_modes = len(eigvals)
print(f"Total modes: {num_modes}")
print(f"Top {n_top_modes} modes indices: {best_ids[:n_top_modes]}")
print(f"Top {n_top_modes} modes scores: {[f'{s:.3f}' for s in scores[best_ids[:n_top_modes]]]}")
top_n_modes = best_ids[:n_top_modes]

# Create the output directory for figures
specific_name = '_'.join([system, args.custom_name])
save_dir = f"experiments/figures/{specific_name}"
os.makedirs(save_dir, exist_ok=True)

# --------------------------------------------------
# Plot Koopman Operator Matrices
# --------------------------------------------------
matrices = [
    (Phi_true, "Raw $\Phi_{true}$ (Model Output)"),
    (Lambda, "Raw $\Lambda$ (Model Inner Evolution)"),
    (V, "Extracted Koopman Modes (Complex V)"),
    (np.diag(eigvals), "True Complex $\Lambda$ (Diagonalized)"),
    (K, r"Operator K")
]

plot_transition_matrices(
    matrices, 
    f"Koopman Operator Matrices, {system}", 
    model.expand_names, 
    save_path=os.path.join(save_dir, "transition_matrices.png")
)

# --------------------------------------------------
# Plot eigenfunctions (top N modes)
# --------------------------------------------------
phi_vals = z @ Phi_true[:, top_n_modes] 
plot_complex_field(grid_points, phi_vals, f"Eigenfunctions {top_n_modes} (Score: {[f'{e:.1f}' for e in scores[top_n_modes]]})", 
                    save_path=os.path.join(save_dir, "eigenfunctions.png"))

# --------------------------------------------------
# Spectrum plot with quality coloring
# --------------------------------------------------
best_ids, scores, residuals = calculate_mode_quality(model, W, eigvals, z_scale, state_bounds=state_bounds)
plot_quality_spectrum(eigvals, scores, theme="dark", 
                        save_path=os.path.join(save_dir, "quality_spectrum.png"))

# --------------------------------------------------
# Visualize eigenvalue frequencies vs magnitudes
# --------------------------------------------------
plot_freq_magnitude(eigvals, scores, theme="dark", 
                    save_path=os.path.join(save_dir, "freq_magnitude.png"))

# --------------------------------------------------
# Visualize mode trajectories
# --------------------------------------------------
single_trajectory = trajectories[:, 0, :] # Take the first trajectory for visualization
plot_mode_trajectories(model, W, eigvals, z_scale, best_ids[:n_top_modes], single_trajectory, 
                        save_path=os.path.join(save_dir, "mode_trajectories.png"))

# --------------------------------------------------
# Visualize mode contributions to state reconstruction
# --------------------------------------------------
with torch.no_grad():
    z_real = model.expand(torch.as_tensor(single_trajectory)).cpu().numpy() / z_scale
    phi_real_traj = z_real @ W
plot_mode_contributions_vs_quality(V, phi_real_traj, best_ids, scores, model.state_dim, n_top=10, 
                                    save_path=os.path.join(save_dir, "mode_contributions.png")) 

# --------------------------------------------------
# Visualize mode energy vs quality
# --------------------------------------------------
plot_mode_energy_vs_quality(V, scores, model.state_dim, n_top=10, 
                            save_path=os.path.join(save_dir, "mode_energy.png"))

print(f"All visualizations successfully saved to: {save_dir}")