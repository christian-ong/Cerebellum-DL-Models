import os
import argparse
import torch
import numpy as np
from src.data_generation.load_data import resolve_split_npz_path
from src.eval.visualize_modes import *

"""
This script visualizes the dynamic modes and eigensystem of a trained Koopman model.

python -m experiments.visualize_dynamic_modes --model_name ml_dmd_band --custom_name band_long_spec10 --data_path data\trajectories\nonlinear\closed_trig_large\long\test.npz

python -m experiments.visualize_dynamic_modes --model_name ml_dmd_band --custom_name band_long_gen3_fix3 --data_path data/trajectories/linear/saddle_point/long

python -m experiments.visualize_dynamic_modes --model_name ml_dmd_band --custom_name band_long_spec10_fix3 --data_path data/trajectories/nonlinear/closed_trig_large/long

python -m experiments.visualize_dynamic_modes --model_name ml_dmd_band --custom_name band_short_spec10_fix3 --data_path data/trajectories/nonlinear/duffing/long


python -m experiments.visualize_dynamic_modes --model_name hardcoded_dmd --custom_name deg3 --data_path data/trajectories/nonlinear/closed_trig_large/long
"""

# --------------------------------------------------
# Execution
# --------------------------------------------------

# Parser
parser = argparse.ArgumentParser(description="Visualize Koopman dynamic modes and eigensystems.")
parser.add_argument("--model_name", type=str, help="Name of the trained model (overrides model_path if provided)")
parser.add_argument("--custom_name", type=str, default="default", help="Custom name of the trained model")
parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
args = parser.parse_args()

# Settings
n_top_modes = 12
grid_res = 100
order_modes_by = "magnitude" # "magnitude" (abs lambda), "quality", "power", "energy"
detect_complex_modes = True
complex_mode_threshold = 1e-3

debug_printing = False

# Load test trajectories to find true boundaries and system name dynamically
test_data_path = resolve_split_npz_path(args.data_path, "test")
data = np.load(test_data_path)
trajectories = data['X'] 
system = str(data["system"])
state_dim = trajectories.shape[-1]
print(f"Visualizing for System: {system}")

# Load model and eigensystem
if "ml" in args.model_name:
    model_path = f"data/models/{args.model_name}/{system}/{args.custom_name}/model_best.pt"
elif "hardcoded_dmd" in args.model_name:
    model_path = f"data/models/{args.model_name}/{system}/{args.custom_name}/model.pt"
else:
    model_path = f"data/models/{args.model_name}/{system}/{args.custom_name}/model.npz"
    raise NotImplementedError("Sorry Kavus, I didn't implement regression models yet...")

model, model_type = build_model_from_checkpoint(model_path)

Phi_model, Lambda_model, Lambda_model_eig, V, W, K = get_koopman_eigensystem(model)
num_modes = Lambda_model.shape[0]
n_top_modes = min(n_top_modes, num_modes)

# Create the output directory for figures
save_dir = f"experiments/figures/{args.model_name}/{system}/{args.custom_name}"
os.makedirs(save_dir, exist_ok=True)
# --------------------------------------------------
# Plot Koopman Operator Matrices
# --------------------------------------------------
# Get analytic matrices for the system
K_c_analytic, K_d_analytic, eigvals_analytic, eigvecs_analytic, analytic_expansion_names = get_system_matrices(system)

# NEW: Build the exact analytic Jordan form from the discrete operator.
Lambda_jordan, V_theory = get_sorted_jordan_form(K_d_analytic)

# Re-format: Lambda_model, Phi_model --> complex matrices instead of rotation blocks
if detect_complex_modes:
    complex_pair_idx = find_complex_pairs(
        Lambda_model, 
        threshold_off_diag=complex_mode_threshold, 
        threshold_diag=complex_mode_threshold,
        print_info=debug_printing
    )
    Lambda_model_complex, Phi_model_complex = rotation_blocks_to_complex(
        Lambda_model, 
        Phi_model, 
        complex_pair_idx
    )
    Lambda_model = Lambda_model_complex
    Phi_model = Phi_model_complex

matrices_to_plot = [
    # Model matrices
    (K, "Model Operator K"),
    (Lambda_model, "Model Raw $\Lambda$"),
    (Phi_model, "Model Raw $\Phi_{true}$"),

    # Analytic matrices
    (K_d_analytic, "Analytic $K$"),
    (Lambda_jordan, "Analytic $\Lambda$ (Jordan Form)"), 
    (V_theory, "Analytic $\Phi_{true}$"),
]

plot_transition_matrices(
    matrices_to_plot, 
    f"Koopman Operator Matrices, {system}", 
    model.expand_names, 
    analytic_expansion_names,
    threshold_include_val=1e-3,
    save_path=os.path.join(save_dir, "transition_matrices.png")
)

# --------------------------------------------------
# Sort modes by chosen criterion
# --------------------------------------------------
# Create a grid covering the state space and lift to latent space
state_bounds, grid_points = get_data_bounds_and_grid_points(trajectories, grid_res=grid_res, state_dim=state_dim)
with torch.no_grad(): grid_points_expanded = safe_expand(model, torch.as_tensor(grid_points, dtype=torch.float32)).cpu().numpy()

# Order modes by chosen criterion
sorting_info = {} # scores are unsorted, indices are the order to sort by
if order_modes_by == "magnitude":
    # sort model modes learnt by the model
    if detect_complex_modes:
        # For complex modes, we can use the magnitude of the eigenvalue (which is the same for both modes in a pair)
        scores_model = np.abs(Lambda_model.diagonal()) # Use diagonal since Lambda is now complex and diagonalized
    else:
        scores_model = np.linalg.norm(Lambda_model, axis=1) # by row, since row i in Lambda determines row i in b_next
    sorted_idx_model = np.argsort(scores_model)[::-1] # Descending order
    # sort analytic modes
    scores_analytic = np.abs(eigvals_analytic)
    sorted_idx_analytic = np.argsort(scores_analytic)[::-1] # Descending order
    
    sorting_info["magnitude"] = {
        "scores_model": scores_model,
        "scores_analytic": scores_analytic,
        "indices_model": sorted_idx_model,
        "indices_analytic": sorted_idx_analytic
    }

elif order_modes_by == "quality":
    print("Warning: This one is cheating!")
    sorted_idx_model, scores_model, _ = modes_by_quality(
        model=model,
        W=W,
        eigvals_analytic=eigvals_analytic,
        state_bounds=state_bounds
    )

    sorting_info["quality"] = {
        "scores_model": scores_model,
        "indices_model": sorted_idx_model,

        # in this case, model = analytic
        "scores_analytic": scores_model, 
        "indices_analytic": sorted_idx_model
    }

else:
    raise ValueError(f"Invalid mode ordering method: {order_modes_by}")

# Apply sorting
sorting = sorting_info[order_modes_by]

# update complex pair indices to reflect sorting
if detect_complex_modes:
    sorted_complex_pair_idx = []
    for i, j in complex_pair_idx:
        new_i = np.where(sorting["indices_model"] == i)[0][0]
        new_j = np.where(sorting["indices_model"] == j)[0][0]
        sorted_complex_pair_idx.append((new_i, new_j))
    complex_pair_idx = sorted_complex_pair_idx

sorted_data = {
    "model": {
        "Lambda": Lambda_model[sorting["indices_model"]][:, sorting["indices_model"]], # matrix
        "Phi": Phi_model[:, sorting["indices_model"]],
        "K": K[sorting["indices_model"]][:, sorting["indices_model"]],
        "scores": sorting["scores_model"][sorting["indices_model"]], # sorted
        "indeces": sorting["indices_model"],
        "complex_pairs": complex_pair_idx if detect_complex_modes else None,
    },

    "analytic": {
        "Lambda": eigvals_analytic[sorting["indices_analytic"]], # vector
        "Phi": eigvecs_analytic[:, sorting["indices_analytic"]],
        "K_d": K_d_analytic[sorting["indices_analytic"]][:, sorting["indices_analytic"]],
        "scores": sorting["scores_analytic"][sorting["indices_analytic"]], # sorted
        "indeces": sorting["indices_analytic"],
    }
}

if debug_printing:
    print(f"Total modes: {num_modes}")
    print(f"Top {n_top_modes} modes:")
    print(f"  Model:")
    print(f"    Indices: {sorted_data['model']['indeces'][:n_top_modes]}")
    print(f"    Scores: {[f'{s:.4f}' for s in sorted_data['model']['scores'][:n_top_modes]]}")
    print(f"  Analytic:")
    print(f"    Indices: {sorted_data['analytic']['indeces'][:n_top_modes]}")
    print(f"    Scores: {[f'{s:.4f}' for s in sorted_data['analytic']['scores'][:n_top_modes]]}")

# --------------------------------------------------
# Plot eigenfunctions (top N modes)
# --------------------------------------------------
# Plot
plot_eigenfunctions(
    grid_points=grid_points, 
    grid_points_expanded=grid_points_expanded, 
    Phi=sorted_data["model"]["Phi"][:, :n_top_modes], 
    scores=sorted_data["model"]["Lambda"][:n_top_modes].diagonal(), 
    score_metric=order_modes_by,
    complex_pair_idx=sorted_data["model"]["complex_pairs"],
    save_path=os.path.join(save_dir, "eigenfunctions.png")
)

# --------------------------------------------------
# Spectrum plot with quality coloring
# --------------------------------------------------
plot_eigenvalue_spectrum(
    eigvals=sorted_data["model"]["Lambda"][:n_top_modes].diagonal(), 
    mode_scores=sorted_data["model"]["scores"][:n_top_modes], 
    score_metric=order_modes_by,
    save_path=os.path.join(save_dir, "eigenvalue_spectrum.png")
)

# --------------------------------------------------
# Visualize eigenvalue frequencies vs magnitudes
# --------------------------------------------------
plot_freq_magnitude(
    eigvals=sorted_data["model"]["Lambda"].diagonal(), 
    mode_scores=sorted_data["model"]["scores"], 
    score_metric=order_modes_by,
    save_path=os.path.join(save_dir, "freq_magnitude.png")
)

# --------------------------------------------------
# Visualize mode trajectories
# --------------------------------------------------
# Parameters
n_modes = 10
n_trajectories = 3
n_steps = 200

plot_trajectories = trajectories[:n_steps, :n_trajectories, :] # (steps, id, state_dim)
plot_Phi = sorted_data["model"]["Phi"][:, :n_modes] # (latent_dim, n_modes)
plot_Lambda = sorted_data["model"]["Lambda"][:n_modes, :n_modes]
plot_koopman_mode_rollout(
    model=model,
    Phi=plot_Phi,
    Lambda=plot_Lambda,
    real_traj=plot_trajectories,
    save_path=os.path.join(save_dir, "mode_trajectories.png")
)

os._exit(0)

# --------------------------------------------------
# Visualize mode contributions to state reconstruction
# --------------------------------------------------
with torch.no_grad():
    single_trajectory = torch.as_tensor(trajectories[:,0,:]).cpu().numpy()
    z_real = safe_expand(model, single_trajectory)
    phi_real_traj = z_real @ W
plot_mode_contributions_vs_quality(
    V, 
    phi_real_traj, 
    sorted_idx_analytic, 
    scores_analytic, 
    model.state_dim, 
    n_top=10, 
    save_path=os.path.join(save_dir, "mode_contributions.png")
)

# --------------------------------------------------
# Visualize mode energy vs quality
# --------------------------------------------------
plot_mode_energy_vs_quality(
    V, 
    scores_analytic, 
    model.state_dim, 
    n_top=10, 
    save_path=os.path.join(save_dir, "mode_energy.png")
)

print(f"All visualizations successfully saved to: {save_dir}")