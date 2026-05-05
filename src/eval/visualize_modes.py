import os
import torch
import numpy as np
from scipy.linalg import expm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from src.models.ml_dmd_free import ML_DMD
from src.models.ml_dmd_band import ML_DMD_BAND
from src.models.ml_linear_dynamics import ML_LinearDynamics



def build_model_from_checkpoint(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
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
    elif model_name == "ml_dmd_band":
        model = ML_DMD_BAND(**kwargs)
    elif model_name == "ml_lineardynamics":
        model = ML_LinearDynamics(**kwargs)
    else:
        raise ValueError(f"Unsupported: {model_name}")

    if "model_state_dict" in ckpt:
        if "lift_weights" in ckpt["model_state_dict"]:
            # Handle old checkpoints with 'lift_weights' key
            state_dict = ckpt["model_state_dict"]
            state_dict.pop("lift_weights")
            model.load_state_dict(state_dict)
    # model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_name


def get_koopman_eigensystem(model):
    """
    Extracts the Koopman modes and eigenfunctions STRICTLY from 
    the neural network's learned parameters (Phi and Lambda).
    """
    # ------------------------------------------------------------------
    # Case 1b: Models that expose getters `get_Phi` / `get_Phi_inv` / `get_Lambda`
    # (handles the updated ML_DMD_BAND implementation)
    # ------------------------------------------------------------------
    if hasattr(model, "get_Phi") and hasattr(model, "get_Phi_inv") and hasattr(model, "get_Lambda"):
        Phi_obj = model.get_Phi()
        Lambda_obj = model.get_Lambda()
        K_obj = model.get_K() if hasattr(model, "get_K") else None

        Phi_true = (
            Phi_obj.detach().cpu().numpy()
            if hasattr(Phi_obj, "detach")
            else np.array(Phi_obj)
        )
        Lambda = (
            Lambda_obj.detach().cpu().numpy()
            if hasattr(Lambda_obj, "detach")
            else np.array(Lambda_obj)
        )
        K_true = (
            K_obj.detach().cpu().numpy()
            if (K_obj is not None and hasattr(K_obj, "detach"))
            else (np.array(K_obj) if K_obj is not None else None)
        )

        Phi_inv_obj = model.get_Phi_inv()
        Phi_inv = (
            Phi_inv_obj.detach().cpu().numpy()
            if hasattr(Phi_inv_obj, "detach")
            else np.array(Phi_inv_obj)
        )

        eigvals, V_inner = np.linalg.eig(Lambda)
        _, W_inner = np.linalg.eig(Lambda.T)

        V = Phi_true @ V_inner

        v_norms = np.linalg.norm(V, axis=0)
        V = V / (v_norms + 1e-12)

        # Calculate W using the learned Encoder (get_Phi_inv)
        W = Phi_inv.T @ W_inner
        W = W * (v_norms + 1e-12)

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

    # ------------------------------------------------------------------
    # Case 2: Old scaled models that expose get_Phi_true and get_Lambda
    # ------------------------------------------------------------------
    if hasattr(model, "get_Phi_true") and hasattr(model, "get_Lambda"):
        Phi_true_obj = model.get_Phi_true()
        Lambda_obj = model.get_Lambda()
        K_obj = model.get_K_true() if hasattr(model, "get_K_true") else None

        Phi_true = (
            Phi_true_obj.detach().cpu().numpy()
            if hasattr(Phi_true_obj, "detach")
            else np.array(Phi_true_obj)
        )
        Lambda = (
            Lambda_obj.detach().cpu().numpy()
            if hasattr(Lambda_obj, "detach")
            else np.array(Lambda_obj)
        )
        K_true = (
            K_obj.detach().cpu().numpy()
            if (K_obj is not None and hasattr(K_obj, "detach"))
            else (np.array(K_obj) if K_obj is not None else None)
        )

        eigvals, V_inner = np.linalg.eig(Lambda)
        _, W_inner = np.linalg.eig(Lambda.T)

        V = Phi_true @ V_inner

        v_norms = np.linalg.norm(V, axis=0)
        V = V / (v_norms + 1e-12)

        Phi_inv_T = np.linalg.pinv(Phi_true).T
        W = Phi_inv_T @ W_inner

        W = W * (v_norms + 1e-12)

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

    # Case 2: Models that expose only the Koopman operator K (e.g., ML_LinearDynamics)
    if hasattr(model, "get_K_true"):
        K_obj = model.get_K_true()
        K_true = K_obj.detach().cpu().numpy() if hasattr(K_obj, "detach") else np.array(K_obj)

        eigvals, V = np.linalg.eig(K_true)
        _, W = np.linalg.eig(K_true.T)

        # Without an explicit Phi mapping, treat Phi_true as identity in lifted space
        Phi_true = np.eye(K_true.shape[0])
        Lambda = np.diag(eigvals)

        v_norms = np.linalg.norm(V, axis=0)
        V = V / (v_norms + 1e-12)
        W = W * (v_norms + 1e-12)

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



def get_system_matrices(system="saddle_point", print_matrices=False, plot_phi=False):
    
    # system values
    vp_mu = 1.5 # vanderpol
    lv_al = 1.1 # lotka-volterra
    lv_be = 0.4 
    lv_ga = 0.4 
    lv_de = 0.1
    pe_g = 9.81 # pendulum
    pe_l = 1.0
    du_al = -1.0 # duffing
    du_be = 1.0
    du_de = 0.2
    lo_sigma = 10.0 # lorenz
    lo_rho = 28.0
    lo_beta = 8.0 / 3.0
    cs_mu = 0.1 # closed_small
    cs_al = -1.0
    cl_mu = 0.1 # closed_large
    cl_al = -1.0 
    cl_be = 0.8
    cl_ga = -0.4
    cl_de = 0.2
    ct_om = 1.0 # closed_trig
    ct_alpha = -0.8
    ct_bs1 = 0.7
    ct_bc1 = -0.5
    ct_bs2 = 0.4
    ct_bc2 = 0.2
    ct_bs3 = -0.25
    ct_bc3 = 0.15
    ct_bx = 0.3
    ct_bx2 = -0.08



    A_cs = {
        "saddle_point": np.array([
            [0.2, 0], 
            [0, -0.2]]),
        "degenerate_node": np.array([
            [-0.7, 0.7], 
            [0, -0.7]]),
        "inward_spiral": np.array([
            [-0.5, -2], 
            [2, -0.5]]),
        "harmonic_oscillator": np.array([
            [0, 1.3], 
            [-1.3, 0]]),

        "vanderpol": np.array([
            [0,1,0],
            [-1,vp_mu, -vp_mu],
            [0,0,0]]),
        "lotka_volterra": np.array([
            [lv_al,0,-lv_be],
            [0, -lv_ga, lv_de],
            [0,0,0]]),
        "pendulum": np.array([
            [0,1,0],
            [0,0,-pe_g/pe_l],
            [0,0,0]]),
        "duffing": np.array([
            [0,1,0],
            [-du_al, -du_de, -du_be],
            [0,0,0]]),
        "lorenz": np.array([
            [-lo_sigma, lo_sigma, 0,0,0],
            [lo_rho, -1, 0,-1,0],
            [0, 0, -lo_beta,0,1],
            [0,0,0,0,0],
            [0,0,0,0,0]]),

        "closed_small": np.array([
            [cs_mu, 0, 0],
            [0, cs_al, -cs_al],
            [0, 0, 2*cs_mu]]),
        "closed_large": np.array([
            [cl_mu, 0, 0, 0, 0],
            [0, cl_al, cl_be, cl_ga, cl_de],
            [0, 0, 2*cl_mu, 0, 0],
            [0, 0, 0, 3*cl_mu, 0],
            [0, 0, 0, 0, 4*cl_mu]]),
        "closed_trig_small": np.array([
            [0, 0, 0, 0, 0, 0], 
            [ct_om, 0, 0, 0, 0,0],
            [0, ct_bx, ct_alpha, ct_bx2, ct_bs1, ct_bc1],
            [0, 2*ct_om, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, ct_om], 
            [0, 0, 0, 0, -ct_om, 0]]),
        "closed_trig_medium": np.array([
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [ct_om, 0, 0, 0, 0, 0, 0, 0],
            [0, ct_bx, ct_alpha, ct_bx2, ct_bs1, ct_bc1, ct_bs2, ct_bc2],
            [0, 2*ct_om, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, ct_om, 0, 0], 
            [0, 0, 0, 0, -ct_om, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 2*ct_om], 
            [0, 0, 0, 0, 0, 0, -2*ct_om, 0]]),
        "closed_trig_large": np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [ct_om, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, ct_bx, ct_alpha, ct_bx2, ct_bs1, ct_bc1, ct_bs2, ct_bc2, ct_bs3, ct_bc3],
            [0, 2*ct_om, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, ct_om, 0, 0, 0, 0], 
            [0, 0, 0, 0, -ct_om, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 2*ct_om, 0, 0], 
            [0, 0, 0, 0, 0, 0, -2*ct_om, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3*ct_om], 
            [0, 0, 0, 0, 0, 0, 0, 0, -3*ct_om, 0]])
    }

    expansion_names = {
        "saddle_point": ["x", "y"],
        "degenerate_node": ["x", "y"],
        "inward_spiral": ["x", "y"],
        "harmonic_oscillator": ["x", "y"],

        "vanderpol": ["x", "y", "x^2 y"],
        "lotka_volterra": ["x", "y", "xy"],
        "pendulum": ["x", "y", "sin(x)"],
        "duffing": ["x", "y", "x^3"],
        "lorenz": ["x", "y", "z", "xz", "xy"],

        "closed_small": ["x", "y", "x^2"],
        "closed_large": ["x", "y", "x^2", "x^3", "x^4"],
        "closed_trig_small": ["1", "x", "y", "x^2", "sin(x)", "cos(x)"],
        "closed_trig_medium": ["1", "x", "y", "x^2", "sin(x)", "cos(x)", "sin(2x)", "cos(2x)"],
        "closed_trig_large": ["1", "x", "y", "x^2", "sin(x)", "cos(x)", "sin(2x)", "cos(2x)", "sin(3x)", "cos(3x)"]
    }

    if system not in A_cs:
        raise ValueError(f"System '{system}' not found. Available systems: {list(A_cs.keys())}")

    # ==========================================================

    A_c = A_cs[system]
    system_expansion_names = expansion_names[system]

    # Compute A_d
    dt = 0.01
    A_d = expm(A_c * dt)
    eigvals, eigvecs = np.linalg.eig(A_d)

    if print_matrices:
        # compute A_d
        np.set_printoptions(precision=4, suppress=True) # print A_d with 4 decimal places
        print('='*80)
        print(f"\nSystem: {system}\n")
        print("A_d:")
        print(A_d)

        # eigen decomp on A_d
        np.set_printoptions(precision=4, suppress=True) # print A_d with 4 decimal places
        print("Lambda")
        print(eigvals)
        # print("Phi")
        # print(eigvecs)

        np.set_printoptions(precision=3, suppress=True)
        print('phi, real')
        print(np.real(eigvecs))
        print('phi, imag')
        print(np.imag(eigvecs))

    # Plot matrix
    if plot_phi:
        plt.title(f"Theoretical Phi for {system}")
        plt.imshow(abs(eigvecs), cmap='viridis')
        plt.colorbar()
        # write value in cells
        for i in range(eigvecs.shape[0]):
            for j in range(eigvecs.shape[1]):
                if abs(eigvecs[i,j]) > 1e-4: # only write values above a certain threshold for readability
                    if abs(np.imag(eigvecs[i,j])) > 1e-4 and abs(np.real(eigvecs[i,j])) > 1e-4:
                        plt.text(j, i, f"{np.real(eigvecs[i,j]):.3f}\n{np.imag(eigvecs[i,j]):.3f}j", ha='center', va='center', color='red')
                    elif abs(np.imag(eigvecs[i,j])) > 1e-4:
                        plt.text(j, i, f"\n{np.imag(eigvecs[i,j]):.3f}j", ha='center', va='center', color='red')
                    elif abs(np.real(eigvecs[i,j])) > 1e-4:
                        plt.text(j, i, f"{np.real(eigvecs[i,j]):.4f}\n", ha='center', va='center', color='red')
        plt.xlabel("Eigenvector Index")
        plt.ylabel("State Dimension")
        plt.show()

    return A_c, A_d, eigvals, eigvecs, system_expansion_names


def find_complex_pairs(
        Lambda, 
        threshold_off_diag=1e-3, 
        threshold_diag=1e-3,
        print_info=False):
    """
    Detects pairs of complex conjugate modes in the Lambda matrix.
    Looks for significant (> threshold) off-diagonal values in 2x2 blocks.
    In case of overlaps, keeps highest scoring blocks (sum of off-diagonal elements).
    """
    
    block_indices = []
    block_scores = []

    # Loop through 2x2 diagonal blocks
    for diag_idx in range(len(Lambda)-1):

        # Get 2x2 block <a,b; c,d>
        a = Lambda[diag_idx, diag_idx]
        b = Lambda[diag_idx, diag_idx+1]
        c = Lambda[diag_idx+1, diag_idx]
        d = Lambda[diag_idx+1, diag_idx+1]

        # Detect rotation blocks (significant off-diagonal values)
        if ((abs(b) > threshold_off_diag) and # off-diag val 1 significant
            (abs(c) > threshold_off_diag) and # off-diag val 2 significant
            (abs(a - d) < threshold_diag)): # diag vals similar 
            block_indices.append((diag_idx, diag_idx+1))

            # Give a significance score
            score_off_diag = (abs(b) + abs(c)) + abs(b - (-c)) # higher for more significant off-diagonal values and values are (conjugate) similar
            score_diag = abs(a - d) # higher if a and d are closer
            block_scores.append(score_off_diag + score_diag)

    # Sort blocks by score
    sorted_scores_idx = np.argsort(block_scores)[::-1] # descending order
    sorted_block_scores = [block_scores[i] for i in sorted_scores_idx]
    sorted_block_indices = [block_indices[i] for i in sorted_scores_idx]

    # Remove overlapping blocks (keep only the highest scoring)
    final_blocks_idx = []
    final_idxs = []
    for idx, score in zip(sorted_block_indices, sorted_block_scores):
        if not any(i in final_idxs for i in idx): # if not already included, include this block
            final_blocks_idx.append(idx)
            final_idxs.extend(idx)

    if print_info:
        print(f"Detected complex conjugate pair indices in model modes: {final_blocks_idx}")

        # print complex values for each block
        conjugate_pairs = []
        for idx in final_blocks_idx:
            a = Lambda[idx[0], idx[0]]
            b = Lambda[idx[0], idx[1]]
            c = Lambda[idx[1], idx[0]]
            d = Lambda[idx[1], idx[1]]
            val1 = a + 1j*b
            val2 = d + 1j*c
            conjugate_pairs.append((val1, val2))
        
        print("These are the detected complex conjugate pairs:")
        for i, (val1, val2) in enumerate(conjugate_pairs):
            print(f"{val1:.3f}, {val2:.3f}")

    return final_blocks_idx
    

def rotation_blocks_to_complex(Lambda, Phi, complex_pair_idx):
    """
    Converts detected 2x2 rotation blocks in Lambda and corresponding columns in Phi into complex conjugate pairs.
    """

    Lambda_complex = Lambda.copy().astype(np.complex128)
    Phi_complex = Phi.copy().astype(np.complex128)

    # Loop through complex blocks
    for idx in complex_pair_idx:
        i, j = idx

        # Get 2x2 block <a,b; c,d>
        a = Lambda[i, i]
        b = Lambda[i, j]
        c = Lambda[j, i]
        d = Lambda[j, j]

        # find complex value
        complex_val_1 = a + 1j*b
        complex_val_2 = d + 1j*c
        avg_complex_val = (a + d)/2 + 1j*(b - c)/2
        print(avg_complex_val)

        # In Lambda: replace block with complex values on diagonal
        Lambda_complex[i, i] = avg_complex_val
        Lambda_complex[j, j] = np.conj(avg_complex_val)
        Lambda_complex[i, j] = 0
        Lambda_complex[j, i] = 0

        # In Phi: v1 = re(mode), v2 = im(mode)
        real_part = Phi[:, i]
        imag_part = Phi[:, j]
        Phi_complex[:, i] = real_part + 1j*imag_part
        Phi_complex[:, j] = real_part - 1j*imag_part

    return Lambda_complex, Phi_complex


def plot_transition_matrices(matrices, title, model_expansion_names, analytic_expansion_names, threshold_include_val = 1e-3, save_path=None):
    fig = plt.figure(figsize=(18, 10))
    # Create a 2x3 grid
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1])
    axes = [fig.add_subplot(gs[i,j]) for i in range(2) for j in range(3)]

    fig.suptitle(title, fontsize=22)
    
    for i, (M, subtitle) in enumerate(matrices):
        ax = axes[i]
        M_mag = np.abs(M)
        im = ax.imshow(M_mag) # color by magnitude, but show values as real/imaginary parts
        ax.set_title(subtitle, fontsize=14)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Model or analytic expansion names (for x/y ticks)
        expansion_names = model_expansion_names if "model" in subtitle.lower() else analytic_expansion_names

        # x ticks for K and Phi
        if ("K" in subtitle) or ("Phi" in subtitle):
            ax.set_xticks(range(len(expansion_names)))
            ax.set_xticklabels(expansion_names, rotation=60, fontsize=9)
        else:
            ax.set_xticks([])

        # y ticks for K
        if ("K" in subtitle):
            ax.set_yticks(range(len(expansion_names)))
            ax.set_yticklabels(expansion_names, fontsize=9)
        else:
            ax.set_yticks([])

        # Dynamically scale font size based on matrix dimension
        n_cols = M.shape[1]
        f_size = 10 if n_cols <= 5 else (8 if n_cols <= 8 else 6)

        for (row, col), v in np.ndenumerate(M):
            if abs(v) > threshold_include_val:
                re_val, im_val = np.real(v), np.imag(v)
                
                if abs(im_val) < threshold_include_val:
                    txt = f"{re_val:.3f}"
                elif abs(re_val) < threshold_include_val:
                    txt = f"{im_val:.3f}j"
                else:
                    sign = "+" if im_val > 0 else "-"
                    txt = f"{re_val:.3f}\n{sign}{abs(im_val):.3f}j"

                ax.text(
                    col, row, txt,
                    ha="center", va="center",
                    fontsize=f_size, color="red"
                )

    # if len(matrices) != 5:
    #     for i in range(len(matrices), len(axes)):
    #         axes[i].axis("off")

    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.1, top=0.92, hspace=0.4, wspace=0.2)
        
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def get_data_bounds_and_grid_points(trajectories, grid_res=100, state_dim=2):
    # Find min and max dynamically for ALL dimensions
    state_bounds = []
    for dim in range(state_dim):
        dim_min, dim_max = trajectories[:, :, dim].min(), trajectories[:, :, dim].max()
        state_bounds.append((dim_min, dim_max))

    # Dynamic Grid setup based on true boundaries (taking 2D slice for 3D+ systems)
    
    x_range = np.linspace(state_bounds[0][0], state_bounds[0][1], grid_res)
    y_range = np.linspace(state_bounds[1][0], state_bounds[1][1], grid_res)
    X, Y = np.meshgrid(x_range, y_range)
    grid_cols = [X.ravel(), Y.ravel()]

    # Pad higher dimensions with their mean trajectory values
    for dim in range(2, state_dim):
        dim_mean = trajectories[:, :, dim].mean()
        grid_cols.append(np.full_like(X.ravel(), dim_mean))
    grid_points = np.column_stack(grid_cols)

    return state_bounds, grid_points


def get_real_representation(V, eigvals):
    """
    Converts complex Koopman modes and eigenvalues into their 
    real-valued block-diagonal form.
    """
    V_real = np.zeros_like(V, dtype=np.float64)
    Lambda_real = np.zeros((len(eigvals), len(eigvals)), dtype=np.float64)
    
    i = 0
    while i < len(eigvals):
        if abs(np.imag(eigvals[i])) < 1e-5:
            V_real[:, i] = np.real(V[:, i])
            Lambda_real[i, i] = np.real(eigvals[i])
            i += 1
        else:
            if i + 1 < len(eigvals):
                V_real[:, i] = np.real(V[:, i])
                V_real[:, i+1] = np.imag(V[:, i]) 
                
                a = np.real(eigvals[i])
                b = np.imag(eigvals[i])
                
                Lambda_real[i, i] = a
                Lambda_real[i, i+1] = b
                Lambda_real[i+1, i] = -b
                Lambda_real[i+1, i+1] = a
                i += 2
            else:
                V_real[:, i] = np.real(V[:, i])
                Lambda_real[i, i] = np.real(eigvals[i])
                i += 1
                
    return V_real, Lambda_real


def plot_complex_field(
        grid_points, 
        grid_points_expanded, 
        Phi,
        scores,
        complex_pair_idx,
        cmap="inferno", 
        save_path=None
    ):
    # Compute eigenfunction values on the grid for the top N modes
    eigenfunction_vals = grid_points_expanded @ Phi

    # Compute complex pairs for annotation
    complex_pairs = [sorted([int(i+1), int(j+1)]) for i, j in complex_pair_idx] # 1-based index and internal sort
    complex_pairs = sorted(complex_pairs, key=lambda x: x[0]) # sort by first index

    grid_n = int(np.sqrt(len(grid_points)))
    extent = [grid_points[:,0].min(), grid_points[:,0].max(), grid_points[:,1].min(), grid_points[:,1].max()]
    num_modes = eigenfunction_vals.shape[1]

    fig, axes = plt.subplots(4, num_modes, figsize=(3+ 2.5*num_modes, 10))
        
    for mode_idx in range(num_modes):
        data_map = {
            "Real": np.real(eigenfunction_vals[:, mode_idx]), 
            "Imag": np.imag(eigenfunction_vals[:, mode_idx]),
            "Mag": np.abs(eigenfunction_vals[:, mode_idx]), 
            "Phase": np.angle(eigenfunction_vals[:, mode_idx])
        }
    
        # Annotate complex pairs
        pair_string = ""
        for n_pair, pair in enumerate(complex_pairs):
            if mode_idx+1 in pair: # if this mode is part of a complex pair
                pair_string += f" {'*' * (n_pair+1)}\n"

        for i, (label, data) in enumerate(data_map.items()):
            if i==0:
                label = (
                    pair_string + 
                    rf"$\mathbf{{EF{mode_idx+1}}}$" + 
                    f"\nScore: {scores[mode_idx]:.4f}" +
                    f"\n{label}")
            ax = axes[i, mode_idx]
            im = ax.imshow(data.reshape(grid_n, grid_n), extent=extent, origin="lower", cmap=cmap, aspect='auto')
            ax.set_title(f"{label}")
            plt.colorbar(im, ax=ax)

    fig.suptitle(f"{num_modes} first eigenfunctions", fontsize=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def modes_by_quality(
    model, 
    W, 
    eigvals_analytic, 
    z_scale, 
    state_bounds, 
    n_modes_to_keep=8
    ):

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
        rhs = phi_roll[:-1, :] * eigvals_analytic[None, :] 

        num = np.linalg.norm(lhs - rhs, axis=0) 
        den = np.linalg.norm(phi_roll[:-1, :], axis=0) + 1e-12 
        residual_accum += num / den

    residual_mean = residual_accum / max(1, n_eval_traj)

    # Calculate Spatial Score (Taking a 2D slice if state_dim > 2)
    x_range = np.linspace(state_bounds[0][0], state_bounds[0][1], 50)
    y_range = np.linspace(state_bounds[1][0], state_bounds[1][1], 50)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    grid_cols = [X_grid.ravel(), Y_grid.ravel()]
    # Pad higher dimensions with their median values
    for dim in range(2, model.state_dim):
        dim_mean = (state_bounds[dim][0] + state_bounds[dim][1]) / 2.0
        grid_cols.append(np.full_like(X_grid.ravel(), dim_mean))
        
    pts = np.column_stack(grid_cols)
    
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

    best_ids = ranked_indices[:n_modes_to_keep]
    
    return best_ids, mode_score, residual_mean


def plot_quality_spectrum(eigvals, mode_scores, theme="dark", save_path=None):
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

    x_bounds = (eigvals.real.min() - 0.1, eigvals.real.max() + 0.1)
    y_bounds = (eigvals.imag.min() - 0.1, eigvals.imag.max() + 0.1)
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    
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
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_freq_magnitude(eigvals, mode_scores, theme="dark", save_path=None):
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
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_mode_trajectories(model, W, eigvals, z_scale, best_ids, real_traj, save_path=None):
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
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_mode_contributions_vs_quality(V, phi_traj, best_ids, scores, state_dim, n_top=10, save_path=None):
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
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_mode_energy_vs_quality(V, scores, state_dim, n_top=20, save_path=None):
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
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()