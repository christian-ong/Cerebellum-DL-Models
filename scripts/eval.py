import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from src.models.linear_baseline import rollout_linear_map
from src.models.dmd_baseline import rollout_dmd_eig
from src.models.ml_dmd import ML_DMD
from src.models.ml_eigen_dmd import MLEigenDMD
from src.models.manual_expansion_ml_dmd import ManualExpansion_MLDMD
from src.models.manual_expansion_manual_dmd import ManualExpansion_ManualDMD
from src.models.manual_expansion_eigen_dmd import ManualExpansion_EigenDMD

from src.eval.rollout_eval import evaluate_validation_rollouts, compute_single_rollout
from src.eval.plot_rollout import plot_time_series, plot_phase_space
from src.eval.plot_eigenvalues import plot_eigenvalues
from src.eval.plot_training_losses import plot_training_losses
from src.eval.plot_matrices import plot_transition_matrix

"""
Global options (defaults):
    --model {
        linear_baseline,
        dmd_baseline,
        ml_dmd,
        manual_expansion_ml_dmd,
        manual_expansion_manual_dmd,
        manual_expansion_eigen_dmd}
    --data_path data/trajectories/{system}_trajectory.npz
    --model_path data/models/{model}_{system}.pt
    --steps 5000
    --traj_index 0
    --name optional_suffix

---------------------------------------------------------------------------------------------

# Linear baseline
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/saddle_point_trajectory.npz --model_path data/models/linear_baseline_saddle_point.npz
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/degenerate_node_trajectory.npz --model_path data/models/linear_baseline_degenerate_node.npz
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/inward_spiral_trajectory.npz --model_path data/models/linear_baseline_inward_spiral.npz
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/harmonic_oscillator_trajectory.npz --model_path data/models/linear_baseline_harmonic_oscillator.npz

# DMD baseline
    python -m scripts.eval --model dmd_baseline --data_path data/trajectories/saddle_point_trajectory.npz --model_path data/models/dmd_baseline_saddle_point.npz
    python -m scripts.eval --model dmd_baseline --data_path data/trajectories/degenerate_node_trajectory.npz --model_path data/models/dmd_baseline_degenerate_node.npz
    python -m scripts.eval --model dmd_baseline --data_path data/trajectories/inward_spiral_trajectory.npz --model_path data/models/dmd_baseline_inward_spiral.npz
    python -m scripts.eval --model dmd_baseline --data_path data/trajectories/harmonic_oscillator_trajectory.npz --model_path data/models/dmd_baseline_harmonic_oscillator.npz

---------------------------------------------------------------------------------------------

# ML DMD
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/saddle_point_trajectory.npz --model_path data/models/ml_dmd_saddle_point.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/degenerate_node_trajectory.npz --model_path data/models/ml_dmd_degenerate_node.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/inward_spiral_trajectory.npz --model_path data/models/ml_dmd_inward_spiral.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/harmonic_oscillator_trajectory.npz --model_path data/models/ml_dmd_harmonic_oscillator.pt

# ML Eigen DMD
    python -m scripts.eval --model ml_eigen_dmd --data_path data/trajectories/saddle_point_trajectory.npz --model_path data/models/ml_eigen_dmd_saddle_point.pt
    python -m scripts.eval --model ml_eigen_dmd --data_path data/trajectories/degenerate_node_trajectory.npz --model_path data/models/ml_eigen_dmd_degenerate_node.pt
    python -m scripts.eval --model ml_eigen_dmd --data_path data/trajectories/inward_spiral_trajectory.npz --model_path data/models/ml_eigen_dmd_inward_spiral.pt
    python -m scripts.eval --model ml_eigen_dmd --data_path data/trajectories/harmonic_oscillator_trajectory.npz --model_path data/models/ml_eigen_dmd_harmonic_oscillator.pt

---------------------------------------------------------------------------------------------

# Manual expansion + Manual DMD
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/saddle_point_trajectory.npz --model_path data/models/manual_expansion_manual_dmd_saddle_point.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/degenerate_node_trajectory.npz --model_path data/models/manual_expansion_manual_dmd_degenerate_node.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/inward_spiral_trajectory.npz --model_path data/models/manual_expansion_manual_dmd_inward_spiral.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/harmonic_oscillator_trajectory.npz --model_path data/models/manual_expansion_manual_dmd_harmonic_oscillator.npz

# Manual expansion + ML DMD
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/saddle_point_trajectory.npz --model_path data/models/manual_expansion_ml_dmd_saddle_point.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/degenerate_node_trajectory.npz --model_path data/models/manual_expansion_ml_dmd_degenerate_node.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/inward_spiral_trajectory.npz --model_path data/models/manual_expansion_ml_dmd_inward_spiral.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/harmonic_oscillator_trajectory.npz --model_path data/models/manual_expansion_ml_dmd_harmonic_oscillator.pt

# Manual expansion + Eigen DMD
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/saddle_point_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd_saddle_point.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/degenerate_node_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd_degenerate_node.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/inward_spiral_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd_inward_spiral.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/harmonic_oscillator_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd_harmonic_oscillator.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/vanderpol_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd_vanderpol.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/lotka_volterra_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd_lotka_volterra.pt

---------------------------------------------------------------------------------------------

Output:
    data/figures/{model}/{system}/{name}/time_series_idx{traj_index}.png
    data/figures/{model}/{system}/{name}/rollout_idx{traj_index}.png
"""

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")

    parser.add_argument("--model", type=str, required=True,
                        choices=[
                            "linear_baseline",
                            "dmd_baseline",
                            "ml_dmd",
                            "ml_eigen_dmd",
                            "manual_expansion_ml_dmd",
                            "manual_expansion_manual_dmd",
                            "manual_expansion_eigen_dmd",
                        ])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--traj_index", type=int, default=0, help="Which validation trajectory to show")
    parser.add_argument("--name", type=str, help="Optional suffix for saved figure")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"


    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    data = np.load(args.data_path)
    X = data["X"]
    state_dim = X.shape[-1]

    if "val_idx" not in data:
        raise ValueError(
            "Dataset does not contain val_idx. "
            "Please regenerate it using simulate_data.py."
        )

    val_idx = data["val_idx"]

    if X.ndim != 3:
        raise ValueError("Evaluation expects multiple trajectories (X must be 3D).")

    if len(val_idx) == 0:
        raise ValueError("No validation trajectories available.")
    
    system = os.path.basename(args.data_path).replace("_trajectory.npz", "")
    print(f"Loaded {X.shape[1]} trajectories for system '{system}', with {len(val_idx)} validation trajectories.")
    # --------------------------------------------------
    # Load model ONCE
    # --------------------------------------------------

    if args.model == "linear_baseline":
        model_data = np.load(args.model_path)
        M = model_data["M"]
        model = None

    elif args.model == "dmd_baseline":
        model_data = np.load(args.model_path)
        Lambda = model_data["Lambda"]
        Phi = model_data["Phi"]
        model = None

    elif args.model == "manual_expansion_manual_dmd":
        model_data = np.load(args.model_path, allow_pickle=True)
        K = model_data["K"]

        if "C" not in model_data:
            raise ValueError(
                "Checkpoint is missing decoder matrix C. "
                "Please retrain manual_expansion_manual_dmd with the updated EDMD-style implementation."
            )
        C = model_data["C"]

        degree = int(model_data["expansion_degree"]) if "expansion_degree" in model_data else 3

        # Backward compatibility: old checkpoints used include_bias,
        # newer ones use constant_expansion
        if "constant_expansion" in model_data:
            constant_expansion = bool(np.asarray(model_data["constant_expansion"]).item())
        elif "include_bias" in model_data:
            constant_expansion = bool(np.asarray(model_data["include_bias"]).item())
        else:
            constant_expansion = True

        if "sine_cosine_expansion" in model_data:
            sine_cosine_expansion = bool(np.asarray(model_data["sine_cosine_expansion"]).item())
        else:
            sine_cosine_expansion = False

        expansion_type = str(model_data["expansion_type"]) if "expansion_type" in model_data else "general"

        if "system_basis" in model_data:
            system_basis = str(model_data["system_basis"])
            if system_basis == "":
                system_basis = None
        else:
            system_basis = system if expansion_type == "specific" else None

        model = ManualExpansion_ManualDMD(
            state_dim=state_dim,
            expansion_degree=degree,
            constant_expansion=constant_expansion,
            sine_cosine_expansion=sine_cosine_expansion,
            expansion_type=expansion_type,
            system=system_basis,
        ).to(device)
        model.eval()

    elif args.model == "ml_dmd":
        ckpt = torch.load(args.model_path, map_location=device)
        model = ML_DMD(state_dim=ckpt["state_dim"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
    
    elif args.model == "ml_eigen_dmd":
        ckpt = torch.load(args.model_path, map_location=device)
        model = MLEigenDMD(state_dim=ckpt["state_dim"],).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

    elif args.model == "manual_expansion_ml_dmd":
        ckpt = torch.load(args.model_path, map_location=device)
        train_args = ckpt["train_args"]

        model = ManualExpansion_MLDMD(
            state_dim=ckpt["state_dim"],
            expansion_degree=train_args["expansion_degree"],
            expansion_type=train_args["expansion_type"],
            system=ckpt["system"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

    elif args.model == "manual_expansion_eigen_dmd":

        ckpt = torch.load(args.model_path, map_location=device)

        train_args = ckpt["train_args"]

        model = ManualExpansion_EigenDMD(
            state_dim=ckpt["state_dim"],
            expansion_degree=train_args["expansion_degree"],
            expansion_type=train_args["expansion_type"],
            system=ckpt["system"],
        ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

    else:
        raise ValueError(f"Unknown model: {args.model}")

    # --------------------------------------------------
    # Evaluate on ALL validation trajectories
    # --------------------------------------------------

    mse_mean, mse_std = evaluate_validation_rollouts(
        X=X,
        val_idx=val_idx,
        model_name=args.model,
        model=model,
        steps=args.steps,
        rollout_linear_map=rollout_linear_map,
        rollout_dmd_eig=rollout_dmd_eig,
        M=locals().get("M"),
        Lambda=locals().get("Lambda"),
        Phi=locals().get("Phi"),
        K=locals().get("K"),
        C=locals().get("C"),
    )

    print(
        f"Validation rollout MSE over {len(val_idx)} trajectories: "
        f"{mse_mean:.6e} ± {mse_std:.6e}"
    )

    # --------------------------------------------------
    # Plot ONE validation trajectory
    # --------------------------------------------------

    if args.traj_index >= len(val_idx):
        raise IndexError(
            f"traj_index={args.traj_index} but only {len(val_idx)} validation trajectories exist."
        )

    traj_id = val_idx[args.traj_index]

    X_true, X_hat = compute_single_rollout(
        X=X,
        traj_id=traj_id,
        steps=args.steps,
        model_name=args.model,
        model=model,
        rollout_linear_map=rollout_linear_map,
        rollout_dmd_eig=rollout_dmd_eig,
        M=locals().get("M"),
        Lambda=locals().get("Lambda"),
        Phi=locals().get("Phi"),
        K=locals().get("K"),
        C=locals().get("C"),
    )

    # --------------------------------------------------
    # Figure directory
    # --------------------------------------------------

    figdir = f"data/figures/{args.model}/{system}/{args.name if args.name else 'default'}"
    os.makedirs(figdir, exist_ok=True)

    # --------------------------------------------------
    # Plots
    # --------------------------------------------------

    plot_time_series(X_true, X_hat, figdir, args.traj_index)

    plot_phase_space(
        X_true,
        X_hat,
        system,
        figdir,
        args.model,
        args.traj_index,
    )

    # --------------------------------------------------
    # Eigenvalue spectrum
    # --------------------------------------------------

    eigvals = None

    if args.model == "linear_baseline":
        eigvals = np.linalg.eigvals(M)

    elif args.model == "dmd_baseline":
        eigvals = Lambda

    elif args.model == "manual_expansion_manual_dmd":
        eigvals = np.linalg.eigvals(K)

    elif model is not None and hasattr(model, "K"):
        A = model.K.weight.detach().cpu().numpy()
        eigvals = np.linalg.eigvals(A)

    elif model is not None and hasattr(model, "Lambda"):
        eigvals = np.diag(model.Lambda.detach().cpu().numpy())

    plot_eigenvalues(eigvals, figdir)

    # --------------------------------------------------
    # Training loss plots
    # --------------------------------------------------

    loss_file = args.model_path.replace(".pt", "_losses.npz")

    if os.path.exists(loss_file):
        plot_training_losses(loss_file, figdir)
    else:
        print(f"No loss file found at {loss_file}, skipping loss plots.")

    # --------------------------------------------------
    # Transition matrix visualization
    # --------------------------------------------------

    model_name = os.path.basename(args.model_path).replace(".pt","")

    expand_names = None
    if "expand_names" in ckpt:
        expand_names = ckpt["expand_names"]

    plot_transition_matrix(
        model=model,
        model_name=model_name,
        figdir=figdir,
        expand_names=expand_names,
    )

if __name__ == "__main__":
    main()