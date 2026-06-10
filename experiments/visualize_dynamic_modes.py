import os
import argparse
import torch
import numpy as np
import glob
from src.data_generation.load_data import resolve_split_npz_path
from src.eval.diagnostics import format_model_label
from src.eval.visualize_modes import *

"""
This script visualizes the dynamic modes and eigensystem of a trained Koopman model.

python -m experiments.visualize_dynamic_modes --model_name ml_dmd_band --custom_name band_long_spec10 --data_path data\trajectories\nonlinear\closed_trig_large\long\test.npz
"""

# --------------------------------------------------
# Execution
# --------------------------------------------------

# Parser
parser = argparse.ArgumentParser(description="Visualize Koopman dynamic modes and eigensystems.")
parser.add_argument("--model_name", type=str, help="Name of the trained model (overrides model_path if provided)")
parser.add_argument("--custom_name", type=str, default="default", help="Custom name of the trained model")
parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")

parser.add_argument("--decomp_method", type=str, default="schur", choices=["numpy","jordan", "schur"], help="Method to use for decomposition (Jordan or Schur)")
parser.add_argument("--mode_order", type=str, default="contribution", choices=["original", "contribution", "mse", "time_int_energy", "magnitude"], help="Criterion to order modes by for visualization")
parser.add_argument("--outdir", type=str, default=None, help="Force a custom output directory for all mode-visualization plots.")
parser.add_argument("--num_steps", type=int, default=None, help="Number of steps for truncated rollouts")
args = parser.parse_args()

if args.model_name not in {"ml_dmd", "ml_dmd_drop", "regression_dmd"}:
    raise ValueError("Mode visualization is only supported for 'ml_dmd' and 'regression_dmd'.")

# Settings
n_top_modes = 10
grid_res = 100
order_modes_by = args.mode_order # "original", "magnitude", "phase", "mse", "init_energy", "time_int_energy" || TODO: "quality", "power"

# Load test trajectories to find true boundaries and system name dynamically
test_data_path = resolve_split_npz_path(args.data_path, "test")
data = np.load(test_data_path)
trajectories = data['X'] 
system = str(data["system"])
state_dim = trajectories.shape[-1]
print(f"Visualizing for System: {system}")

# Try to locate a longer test split (e.g. created with name suffix _T10). If present,
# use it only for the Koopman mode rollout plots so they show longer horizons.
def _find_long_test_npz(base_data_path: str):
    # If base_data_path is a directory, try several candidate folder names
    if os.path.isdir(base_data_path):
        parent = os.path.dirname(base_data_path)
        base = os.path.basename(base_data_path)
        candidates = [
            os.path.join(parent, base + "_T10"),
            base_data_path + "_T10",
        ]
        for c in candidates:
            test_npz = os.path.join(c, "test.npz")
            if os.path.exists(test_npz):
                return test_npz
        # Fallback: look for any sibling folder that contains both system and '_T10' in name
        siblings = glob.glob(os.path.join(parent, f"*{base.split('_')[0]}*_T10*"))
        for s in siblings:
            test_npz = os.path.join(s, "test.npz")
            if os.path.exists(test_npz):
                return test_npz
        return None

    # If base_data_path is a file (npz), try inserting _T10 before extension
    if base_data_path.endswith('.npz'):
        base, ext = os.path.splitext(base_data_path)
        candidate = base + "_T10" + ext
        if os.path.exists(candidate):
            return candidate
        return None

    return None

# Attempt to load a longer test set for mode-rollout visualizations
long_test_path = _find_long_test_npz(args.data_path)
long_trajectories = None
if long_test_path is not None:
    try:
        long_data = np.load(long_test_path)
        long_trajectories = long_data.get('X', None)
        if long_trajectories is None:
            long_trajectories = long_data[long_data.files[0]] if len(long_data.files) > 0 else None
        if long_trajectories is not None:
            print(f"Using long test set for koopman rollouts: {long_test_path}")
    except Exception:
        long_trajectories = None

# Load model and eigensystem
if "ml" in args.model_name:
    model_path = f"data/models/{args.model_name}/{system}/{args.custom_name}/model_best.pt"
else:
    model_path = f"data/models/{args.model_name}/{system}/{args.custom_name}/model.npz"

model, model_type, train_args = build_model_from_checkpoint(model_path, device="cpu")
expansion_type = getattr(model, "expansion_type", None)
plot_subtitle = format_model_label(args.model_name, model, {"ckpt": {"train_args": train_args}}, system=system)

model_param_type = "complex" if "regression" in args.model_name else "real"

if args.model_name in {"regression_dmd"}:
    Phi_model, Lambda_model, K_model, V, W,  = get_koopman_eigensystem(model)
    # Explicitly define the raw W_model for Regression DMD using the exact pseudo-inverse
    W_model = np.linalg.pinv(Phi_model).T
else:
    Phi_model, Lambda_model, W_model, K_model, V, W,  = get_koopman_eigensystem(model)
    
    # ---> CRITICAL FIX: The neural network stores W with modes as rows. 
    # Transpose it so modes are columns for NumPy operations! <---
    W_model = W_model.T

num_modes = Lambda_model.shape[0]
n_top_modes = min(n_top_modes, num_modes)

# Create the output directory for figures. Respect EVAL_BASE_DIR so callers can centralize outputs.
save_root = os.environ.get("EVAL_BASE_DIR", "experiments/figures")
save_dir = os.path.join(save_root, args.model_name, system)
if args.outdir:
    save_dir = args.outdir
else:
    expansion_folder = str(expansion_type) if expansion_type is not None else "none"
    if expansion_type == "rbf":
        bandwidth = None
        try:
            bandwidth = getattr(model, "rbf_bandwidth_mode", None)
        except Exception:
            bandwidth = None
        if bandwidth is None and isinstance(train_args, dict):
            bandwidth = train_args.get("rbf_bandwidth_mode", None)
        bw = str(bandwidth).strip().lower() if bandwidth is not None and not (isinstance(bandwidth, float) and np.isnan(bandwidth)) else "global"
        expansion_folder = os.path.join("rbf", "global" if bw == "global" else "knn")
    if expansion_type in {"hankel", "hankel_svd"}:
        expansion_folder = "hankel_svd"

    save_dir = os.path.join(save_dir, expansion_folder)
    if args.model_name in {"ml_dmd", "ml_dmd_drop"}:
        l1_weight = None
        try:
            l1_weight = getattr(model, "l1_weight", None)
        except Exception:
            l1_weight = None
        if l1_weight is None and isinstance(train_args, dict):
            l1_weight = train_args.get("l1_weight")
        if l1_weight is not None:
            try:
                l1_value = float(l1_weight)
                if l1_value == 0.0:
                    save_dir = os.path.join(save_dir, "l1_0.0")
                else:
                    save_dir = os.path.join(save_dir, f"l1_{l1_value:.0e}")
            except Exception:
                save_dir = os.path.join(save_dir, str(l1_weight))
    save_dir = os.path.join(save_dir, args.custom_name)
os.makedirs(save_dir, exist_ok=True)

# --------------------------------------------------
# Plot Koopman Operator Matrices
# --------------------------------------------------
# Get analytic matrices for the system so the downstream mode plots can use them.
K_c_analytic, K_d_analytic, Lambda_analytic, Phi_analytic, analytic_expansion_names = get_system_matrices(
    system,
    decomp_type=args.decomp_method,
)

# Find both complex and rotation block formats.
complex_pair_idx = find_complex_pairs(
    Lambda_model
)
if model_param_type == "real":
    Lambda_model_complex, Phi_model_complex, W_model_complex = rotation_blocks_to_complex(
        Lambda_model,
        Phi_model,
        complex_pair_idx,
        W=W_model,
    )
else:
    Lambda_model_complex, Phi_model_complex, W_model_complex = Lambda_model, Phi_model, W_model
    Phi_model, Lambda_model, W_model = get_real_representation(
        Phi_model,
        Lambda_model,
        jordan_value=1,
        threshold_jordan=1e-1,
        W=W_model,
    )

K_model_complex = Phi_model_complex @ Lambda_model_complex @ np.linalg.pinv(Phi_model_complex)

if args.custom_name != "no_expansion":
    skip_transition_matrices = expansion_type in {"general", "rbf", "hankel_svd"}
    if skip_transition_matrices:
        print(
            f"Skipping transition-matrix plot for expansion_type='{expansion_type}' "
            f"because the basis is too large / not human-readable."
        )
else:
    matrices_to_plot = [
        (K_model, "Model Operator K, real"),
        (Lambda_model, "Model Raw $\Lambda$, real"),
        (Phi_model, "Model Raw $\Phi_{true}$, real"),
        (K_model_complex, "Model Operator K, complex"),
        (Lambda_model_complex, "Model Raw $\Lambda$, complex"),
        (Phi_model_complex, "Model Raw $\Phi_{true}$, complex"),
        (K_d_analytic, "Analytic $K$"),
        (Lambda_analytic, "Analytic $\Lambda$"),
        (Phi_analytic, "Analytic $\Phi_{true}$"),
    ]

    plot_transition_matrices(
        matrices_to_plot,
        f"Koopman Operator Matrices, {system}",
        model.expand_names,
        analytic_expansion_names,
        threshold_include_val=1e-4,
        save_path=os.path.join(save_dir, "transition_matrices.png"),
        subtitle=plot_subtitle,
    )

# --------------------------------------------------
# Sort modes by chosen criterion
# --------------------------------------------------
# Create a grid covering the state space and lift to latent space
state_bounds, grid_points = get_data_bounds_and_grid_points(trajectories, grid_res=grid_res, state_dim=state_dim)
with torch.no_grad():
    dummy_t = torch.as_tensor(grid_points, dtype=torch.float32)
    
    if hasattr(model.expander, "delay_depth") and model.expander.delay_depth > 1:
        q = model.expander.delay_depth
        dummy_history = dummy_t.repeat_interleave(q, dim=1)
    else:
        dummy_history = dummy_t
        
    # FIX: Regression DMD requires external state normalization before expansion
    if hasattr(model, "_normalize_x"):
        dummy_history = model._normalize_x(dummy_history)
        
    grid_points_expanded = safe_expand(model, dummy_history)
        
    # --- FIX: Must scale the latent grid before passing to W! ---
    if hasattr(model, "_normalize"):
        grid_points_expanded = model._normalize(grid_points_expanded)
    elif hasattr(model, "psi_scale"):
        grid_points_expanded = grid_points_expanded / model.psi_scale
        
    grid_points_expanded = grid_points_expanded.cpu().numpy()

# Order modes by chosen criterion
sorting_info = {} # scores are unsorted, indices are the order to sort by
if order_modes_by == "original":
    # keep original order (no sorting)
    sorting_info["original"] = {
        "scores_model": np.zeros(num_modes), # dummy scores for plotting
        "scores_analytic": np.zeros(len(Lambda_analytic)), # dummy scores for plotting
        "indices_model": np.arange(num_modes),
        "indices_analytic": np.arange(len(Lambda_analytic))
    }
elif order_modes_by == "contribution":
    # 1. Get complex modal coefficients over time (b)
    with torch.no_grad():
        n_trajectories = min(3, trajectories.shape[1])
        
        # ---> FIX: Properly format history for delay models BEFORE expanding! <---
        if hasattr(model, "expander") and hasattr(model.expander, "delay_depth") and model.expander.delay_depth > 1:
            q = model.expander.delay_depth
            hist_list = []
            for lag in range(q):
                hist_list.append(trajectories[q - 1 - lag : trajectories.shape[0] - lag, :n_trajectories, :])
            x_packed = np.concatenate(hist_list, axis=-1)
            x = torch.as_tensor(x_packed.reshape(-1, x_packed.shape[-1]), dtype=torch.float32)
        else:
            x = torch.as_tensor(trajectories[:, :n_trajectories, :].reshape(-1, state_dim), dtype=torch.float32)
            
        # ---> FIX: Regression DMD requires external state normalization <---
        if hasattr(model, "_normalize_x"):
            x = model._normalize_x(x)
            
        expanded_traj = safe_expand(model, x)
        
        # Normalize the latent space
        if hasattr(model, "_normalize"):
            expanded_traj = model._normalize(expanded_traj)
        elif hasattr(model, "psi_scale"):
            expanded_traj = expanded_traj / model.psi_scale.to(expanded_traj.device)
            
    # Project onto Complex Left Eigenvectors
    b_t = expanded_traj.cpu().numpy() @ W_model_complex 
    
    # RMS amplitude of each mode (b_t is 2D: [Samples, Modes], so axis=0)
    coeff_rms = np.sqrt(np.mean(np.abs(b_t) ** 2, axis=0))

    # 2. Calculate Physical Norm of Complex Eigenvectors (Phi)
    latent_dim = Phi_model_complex.shape[1]
    mode_state_norms = np.zeros(latent_dim)
    
    for j in range(latent_dim):
        phi_j = Phi_model_complex[:, j]
        
        if hasattr(model, "C_fitted"):
            C_mat = model.C_fitted.detach().cpu().numpy()
            # ---> FIX: Slice the delay history away before scaling <---
            state_vec = (C_mat @ phi_j)[:model.state_dim] 
            if hasattr(model, "x_scale"):
                state_vec = state_vec * model.x_scale[:model.state_dim].detach().cpu().numpy()
        else:
            # ML DMD: Scale the direction and de-expand
            if hasattr(model, "lift_scale"):
                scale = model.lift_scale.detach().cpu().numpy()
                phi_j_unscaled = phi_j * scale
            elif hasattr(model, "psi_scale"):
                scale = model.psi_scale.detach().cpu().numpy()
                phi_j_unscaled = phi_j * scale
            else:
                phi_j_unscaled = phi_j
                
            device = "cpu"
            if hasattr(model, 'parameters') and list(model.parameters()): device = next(model.parameters()).device
            elif hasattr(model, 'buffers') and list(model.buffers()): device = next(model.buffers()).device
            
            phi_real = torch.as_tensor(phi_j_unscaled.real, dtype=torch.float32).unsqueeze(0).to(device)
            phi_imag = torch.as_tensor(phi_j_unscaled.imag, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                s_real = safe_de_expand(model, phi_real).squeeze(0).cpu().numpy()
                s_imag = safe_de_expand(model, phi_imag).squeeze(0).cpu().numpy()
            state_vec = s_real + 1j * s_imag
            
        mode_state_norms[j] = np.linalg.norm(state_vec[:model.state_dim])
        
    # 3. Compute final contribution and sort
    scores_model = coeff_rms * mode_state_norms
    sorted_idx_model = np.argsort(scores_model)[::-1]

    sorting_info["contribution"] = {
        "scores_model": scores_model,
        "scores_analytic": np.zeros(len(Lambda_analytic)),
        "indices_model": sorted_idx_model,
        "indices_analytic": np.arange(len(Lambda_analytic)),
    }

elif order_modes_by == "magnitude":
    # Always use the complex diagonal, regardless of real/complex model type!
    scores_model = np.abs(Lambda_model_complex.diagonal())
    scores_analytic = np.abs(Lambda_analytic)
    sorted_idx_model = np.argsort(scores_model)[::-1] 
    sorted_idx_analytic = np.argsort(scores_analytic)[::-1]

    sorting_info[order_modes_by] = {
        "scores_model": scores_model,
        "scores_analytic": scores_analytic,
        "indices_model": sorted_idx_model,
        "indices_analytic": sorted_idx_analytic
    }

elif order_modes_by in ["mse", "init_energy", "time_int_energy"]: # data-driven sorting criterias
    if order_modes_by == "mse":
        n_trajectories = 3
        n_steps = trajectories.shape[0] 
        calculate_trajectories = trajectories[:n_steps, :n_trajectories, :] 

        sorted_idx_model, mode_mses = modes_by_mse(
            model=model,
            Phi=Phi_model_complex,       # <--- FIX: Use Complex
            Lambda=Lambda_model_complex, # <--- FIX: Use Complex
            real_traj=calculate_trajectories,
            W=W_model_complex            # <--- FIX: Use Complex
        )
        scores_model = mode_mses

    elif order_modes_by == "init_energy":
        with torch.no_grad():
            if hasattr(model.expander, "delay_depth") and model.expander.delay_depth > 1:
                q = model.expander.delay_depth
                hist_list = [trajectories[q - 1 - lag, :, :] for lag in range(q)]
                x = torch.as_tensor(np.concatenate(hist_list, axis=-1), dtype=torch.float32)
            else:
                x = torch.as_tensor(trajectories[0,:,:], dtype=torch.float32)
            
            # ---> FIX: External normalization for Regression DMD <---
            if hasattr(model, "_normalize_x"):
                x = model._normalize_x(x)
                
            expanded_init_conditions = safe_expand(model, x)
            if hasattr(model, "_normalize"):
                expanded_init_conditions = model._normalize(expanded_init_conditions)
            elif hasattr(model, "psi_scale"):
                device = model.psi_scale.device if isinstance(model.psi_scale, torch.Tensor) else "cpu"
                expanded_init_conditions = expanded_init_conditions / model.psi_scale.to(device)
                
        init_mode_amplitudes = expanded_init_conditions.cpu().numpy() @ W_model_complex 
        mode_energies = np.linalg.norm(init_mode_amplitudes, axis=0) 
        scores_model = mode_energies
        sorted_idx_model = np.argsort(mode_energies)[::-1] 

    elif order_modes_by == "time_int_energy":
        with torch.no_grad():
            if hasattr(model.expander, "delay_depth") and model.expander.delay_depth > 1:
                q = model.expander.delay_depth
                hist_list = []
                for lag in range(q):
                    hist_list.append(trajectories[q - 1 - lag : trajectories.shape[0] - lag, :, :])
                x_packed = np.concatenate(hist_list, axis=-1)
                x = torch.as_tensor(x_packed.reshape(-1, x_packed.shape[-1]), dtype=torch.float32)
            else:
                x = torch.as_tensor(trajectories.reshape(-1, state_dim), dtype=torch.float32)
            
            # ---> FIX: External normalization for Regression DMD <---
            if hasattr(model, "_normalize_x"):
                x = model._normalize_x(x)
            
            expanded_traj = safe_expand(model, x)
            if hasattr(model, "_normalize"):
                expanded_traj = model._normalize(expanded_traj)
            elif hasattr(model, "psi_scale"):
                device = model.psi_scale.device if isinstance(model.psi_scale, torch.Tensor) else "cpu"
                expanded_traj = expanded_traj / model.psi_scale.to(device)

        mode_amplitudes = expanded_traj.cpu().numpy() @ W_model_complex 
        mode_energies_temp = np.linalg.norm(mode_amplitudes, axis=0) 
        scores_model = mode_energies_temp 
        sorted_idx_model = np.argsort(scores_model)[::-1]

    # Save to the sorting_info dictionary
    sorting_info[order_modes_by] = {
        "scores_model": scores_model,
        "indices_model": sorted_idx_model,
        "scores_analytic": np.zeros(len(Lambda_analytic)), 
        "indices_analytic": np.arange(len(Lambda_analytic)) 
    }
else:
    raise ValueError(f"Invalid mode ordering method: {order_modes_by}")

# Apply sorting
sorting = sorting_info[order_modes_by]
orig_complex_pair_idx = complex_pair_idx.copy()

# ---> FIX: Sanitize sort indices to keep 2x2 rotation blocks together and in order <---
new_indices_model = []
added = set()

for idx in sorting["indices_model"]:
    if idx in added:
        continue
        
    # Check if this index belongs to a complex pair
    pair_found = False
    for pair in orig_complex_pair_idx:
        if idx in pair:
            # Append BOTH indices in their original mathematical order
            new_indices_model.append(pair[0])
            new_indices_model.append(pair[1])
            added.add(pair[0])
            added.add(pair[1])
            pair_found = True
            break
            
    if not pair_found:
        new_indices_model.append(idx)
        added.add(idx)
        
sorting["indices_model"] = np.array(new_indices_model, dtype=int)
# --------------------------------------------------------------------------------

# update complex pair indices to reflect sorting
sorted_complex_pair_idx = []
for i, j in complex_pair_idx:
    new_i = np.where(sorting["indices_model"] == i)[0][0]
    new_j = np.where(sorting["indices_model"] == j)[0][0]
    sorted_complex_pair_idx.append((new_i, new_j))
complex_pair_idx = sorted_complex_pair_idx

sorted_data = {
    "model": {
        "complex": {
            # Use np.ix_ for safer, slightly faster 2D NumPy permutation
            "Lambda": Lambda_model_complex[np.ix_(sorting["indices_model"], sorting["indices_model"])],
            "Phi": Phi_model_complex[:, sorting["indices_model"]],
            "W": W_model_complex[:, sorting["indices_model"]],
            "K": K_model, # <--- FIX: Do not sort the transition operator!
            "scores": sorting["scores_model"][sorting["indices_model"]],
            "indeces": sorting["indices_model"],
            "complex_pairs": complex_pair_idx,
        },
        "real": {
            "Lambda": Lambda_model[np.ix_(sorting["indices_model"], sorting["indices_model"])],
            "Phi": Phi_model[:, sorting["indices_model"]],
            "W": W_model[:, sorting["indices_model"]],
            "K": K_model, # <--- FIX: Do not sort the transition operator!
            "scores": sorting["scores_model"][sorting["indices_model"]],
            "indeces": sorting["indices_model"],
            "complex_pairs": complex_pair_idx,
        }
    },
    "analytic": {
        "Lambda": Lambda_analytic[sorting["indices_analytic"]],
        "Phi": Phi_analytic[:, sorting["indices_analytic"]],
        "K_d": K_d_analytic, # <--- FIX: Do not sort the transition operator!
        "scores": sorting["scores_analytic"][sorting["indices_analytic"]],
        "indeces": sorting["indices_analytic"],
    }
}

# --------------------------------------------------
# Visualize koopman mode rollouts
# --------------------------------------------------
# Parameters
n_modes = 10
n_trajectories = 3

# 1. Base trajectories (target_time / short)
n_steps = trajectories.shape[0] 
delay_depth = int(getattr(model.expander, "delay_depth", 1)) if hasattr(model, "expander") else 1

if args.num_steps is not None:
    # We need args.num_steps PLUS the delay history to roll out properly
    trunc_len = min(args.num_steps + delay_depth, n_steps)
    plot_trajectories = trajectories[:trunc_len, :n_trajectories, :]
else:
    plot_trajectories = trajectories[:n_steps, :n_trajectories, :]

# 2. Long trajectories
if long_trajectories is not None:
    try:
        long_n_steps = long_trajectories.shape[0]
        long_plot_trajectories = long_trajectories[:long_n_steps, :n_trajectories, :]
    except Exception:
        long_plot_trajectories = None
else:
    long_plot_trajectories = None

# Build the list of configurations we want to plot
rollout_configs = [("target_time", plot_trajectories)]
if long_plot_trajectories is not None:
    rollout_configs.append(("long", long_plot_trajectories))

# Extract the mode matrices once
plot_Phi_real = sorted_data["model"]["real"]["Phi"]
plot_Lambda_real = sorted_data["model"]["real"]["Lambda"]
plot_W_real = sorted_data["model"]["real"]["W"]

plot_Phi_complex = sorted_data["model"]["complex"]["Phi"]
plot_Lambda_complex = sorted_data["model"]["complex"]["Lambda"]
plot_W_complex = sorted_data["model"]["complex"]["W"]

# Loop over the configs and plot both Real and Complex versions
for config_name, traj_data in rollout_configs:
    current_subtitle = f"{plot_subtitle}"

    # Real Plot
    plot_koopman_mode_rollout(
        model=model,
        Phi=plot_Phi_real,
        Lambda=plot_Lambda_real,
        real_traj=traj_data,
        save_path=os.path.join(save_dir, f"mode_trajectories_real_{config_name}.png"),
        subtitle=current_subtitle,
        main_title="Koopman Eigenfunction Rollout",
        W=plot_W_real
    )

    # Complex Plot
    plot_koopman_mode_rollout(
        model=model,
        Phi=plot_Phi_complex,
        Lambda=plot_Lambda_complex,
        real_traj=traj_data,
        save_path=os.path.join(save_dir, f"mode_trajectories_complex_{config_name}.png"),
        subtitle=current_subtitle,
        main_title="Koopman Eigenfunction Rollout",
        W=plot_W_complex
    )

# --------------------------------------------------
# Plot eigenfunctions (top N modes)
# --------------------------------------------------
plot_eigenfunctions(
    grid_points=grid_points, 
    grid_points_expanded=grid_points_expanded, 
    W=sorted_data["model"]["complex"]["W"][:, :n_top_modes],
    scores=sorted_data["model"]["complex"]["scores"][:n_top_modes], 
    eigvals=sorted_data["model"]["complex"]["Lambda"][:n_top_modes].diagonal(),
    score_metric=order_modes_by,
    complex_pair_idx=sorted_data["model"]["complex"]["complex_pairs"],
    save_path=os.path.join(save_dir, "eigenfunctions.png"),
    subtitle=plot_subtitle,
)

# --------------------------------------------------
# Plot truncated rollouts (using top N modes)
# --------------------------------------------------
# Parameters
n_trajectories = 4
n_modes = range(1, n_top_modes+1) 

# --- FIX: Slice to match target_time + delay_depth ---
if args.num_steps is not None:
    trunc_len = min(args.num_steps + delay_depth, trajectories.shape[0])
    trunc_trajectories = trajectories[:trunc_len, :n_trajectories, :]
else:
    trunc_trajectories = trajectories[:, :n_trajectories, :]

# Use complex representation for truncation so complex conjugate pairs stay together
Phi = Phi_model_complex
Lambda = Lambda_model_complex
W = W_model_complex
supports_truncated_rollout = (
    all(hasattr(model, attr) for attr in ("Phi_lift_fitted", "Lambda_fitted", "C_fitted", "psi_scale", "x_scale"))
    or (
        hasattr(model, "get_Phi")
        and hasattr(model, "get_Lambda")
        and hasattr(model, "expander")
        and hasattr(model.expander, "expand")
        and hasattr(model.expander, "de_expand")
    )
)
summary_stats = []
if supports_truncated_rollout:
    sorted_indices_full = sorted_data["model"]["real"]["indeces"]
    truncation_dir = os.path.join(save_dir, "truncation")
    os.makedirs(truncation_dir, exist_ok=True)

    # 1. Define target mode counts
    # Base: 1-10
    # Percentages: 10%, 25%, 50%, 75%, 100% of num_modes
    fixed_targets = list(range(1, 11))
    percent_targets = [max(1, min(num_modes, int(np.ceil(p * num_modes)))) for p in [0.1, 0.25, 0.5, 0.75, 1.0]]
    targets = sorted(list(set(fixed_targets + percent_targets)))
    targets = [n for n in targets if n <= num_modes]

    # Helper for contribution scores
    try:
        from src.eval.noise_robustness import compute_mode_diagnostics
        diag = compute_mode_diagnostics(model, trajectories)
        print("Successfully computed mode truncation scores")
    except Exception:
        diag = {}

    def _subset_contribution_score(mode_indices):
        contrib = np.asarray(diag.get("state_contribution", []), dtype=float)
        if contrib.size == 0:
            try:
                # FIX: Use the natively UNSORTED scores array so original matrix indices map correctly!
                contrib = np.asarray(sorting["scores_model"], dtype=float)
            except (NameError, KeyError, TypeError):
                return None
                
        if contrib.size == 0 or not np.isfinite(np.sum(contrib)):
            return None
            
        idx = np.asarray(mode_indices, dtype=int)
        valid_idx = idx[(idx >= 0) & (idx < contrib.shape[0])]
        if valid_idx.size == 0:
            return None
            
        total_contrib = float(np.sum(contrib))
        if total_contrib <= 1e-12: return 0.0
        
        # Now valid_idx correctly extracts the massive scores of the top modes
        return float(np.sum(contrib[valid_idx]) / total_contrib)

    # 2. Run loop over targets
    for n in targets:
        # Preserve complex pairs
        current_subset = list(sorted_indices_full[:n])
        modes_to_add = []
        for mode in current_subset:
            for pair in orig_complex_pair_idx:
                if mode in pair:
                    partner_idx = pair[1] if pair[0] == mode else pair[0]
                    if partner_idx not in current_subset and partner_idx not in modes_to_add:
                        modes_to_add.append(partner_idx)
        current_subset.extend(modes_to_add)
        current_subset = sorted(current_subset, key=lambda x: list(sorted_indices_full).index(x))

        mode_idx = np.asarray(current_subset, dtype=int)
        actual_n = len(mode_idx)
        
        # Check if we already evaluated this count (e.g., if 10% happens to be 5 modes, which is already in 1-10)
        if any(s['n_modes'] == actual_n for s in summary_stats):
            continue
            
        contrib_score = _subset_contribution_score(mode_idx)
        
        # Generate plot for all requested targets
        save_name = f"truncated_rollout_n{actual_n}_modes.png"
        subtitle_text = f"{plot_subtitle}\n{actual_n} Modes"
        if contrib_score is not None:
            subtitle_text += f" | Contribution = {contrib_score:.3f}"
            
        reconstructed_traj = truncated_rollout(
            model=model, real_traj=trunc_trajectories, n_modes=actual_n, 
            Phi=Phi, Lambda=Lambda, W=W,
            mode_indices=mode_idx, save_path=truncation_dir, 
            save_name=save_name, subtitle=subtitle_text, plot=True
        )
        
        mse = np.mean((reconstructed_traj - trunc_trajectories) ** 2)
        summary_stats.append({
            "n_modes": actual_n, 
            "rmse": np.sqrt(mse), 
            "contribution": contrib_score if contrib_score is not None else 0.0
        })

else:
    print("Skipping truncated rollout plot: model does not expose a supported modal/decoder path.")

if len(summary_stats) > 0:
    summary_stats = sorted(summary_stats, key=lambda x: x['n_modes'])
    counts = [s['n_modes'] for s in summary_stats]
    rmses = [s['rmse'] for s in summary_stats]
    contribs = [s['contribution'] * 100 for s in summary_stats]

    plot_rmse_contribution(
        mode_counts=counts, rmses=rmses, contributions=contribs,
        save_path=os.path.join(truncation_dir, "summary_performance.png"),
        subtitle=plot_subtitle
    )

# --------------------------------------------------
# Spectrum plot with quality coloring
# --------------------------------------------------
plot_eigenvalue_spectrum(
    eigvals=sorted_data["model"]["complex"]["Lambda"][:n_top_modes].diagonal(), 
    mode_scores=sorted_data["model"]["complex"]["scores"][:n_top_modes], 
    score_metric=order_modes_by,
    save_path=os.path.join(save_dir, "eigenvalue_spectrum.png"),
    subtitle=plot_subtitle,
)

# --------------------------------------------------
# Visualize eigenvalue frequencies vs magnitudes
# --------------------------------------------------
plot_freq_magnitude(
    eigvals=sorted_data["model"]["complex"]["Lambda"].diagonal(), 
    mode_scores=sorted_data["model"]["complex"]["scores"], 
    score_metric=order_modes_by,
    save_path=os.path.join(save_dir, "freq_magnitude.png"),
    subtitle=plot_subtitle,
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