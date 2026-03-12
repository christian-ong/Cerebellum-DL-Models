import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from src.models.linear_baseline import rollout_linear_map
from src.models.dmd_baseline import rollout_dmd_eig
from src.models.ml_dmd import ML_DMD
from src.models.manual_expansion_ml_dmd import ManualExpansion_MLDMD
from src.models.manual_expansion_manual_dmd import ManualExpansion_ManualDMD
from src.models.manual_expansion_eigen_dmd import ManualExpansion_EigenDMD
from src.eval.rollout import rollout_ae_model
from src.models.dmd_baseline import *
from src.models.ml_eigen_dmd import MLEigenDMD


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
        model_data = np.load(args.model_path)
        K = model_data["K"]
        if "C" not in model_data:
            raise ValueError(
                "Checkpoint is missing decoder matrix C. "
                "Please retrain manual_expansion_manual_dmd with the updated EDMD-style implementation."
            )
        C = model_data["C"]
        degree = int(model_data["expansion_degree"]) if "expansion_degree" in model_data else 3
        include_bias = bool(np.asarray(model_data["include_bias"]).item()) if "include_bias" in model_data else True
        model = ManualExpansion_ManualDMD(
            state_dim=state_dim,
            expansion_degree=degree,
            include_bias=include_bias,
        ).to(device)
        model.eval()

    elif args.model == "ml_dmd":
        ckpt = torch.load(args.model_path, map_location=device)
        model = ML_DMD(state_dim=ckpt["state_dim"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

    elif args.model == "manual_expansion_ml_dmd":
        ckpt = torch.load(args.model_path, map_location=device)
        model = ManualExpansion_MLDMD(state_dim=ckpt["state_dim"],).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

    elif args.model == "ml_eigen_dmd":
        ckpt = torch.load(args.model_path, map_location=device)
        model = MLEigenDMD(state_dim=ckpt["state_dim"],).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

    else:
        raise ValueError(f"Unknown model: {args.model}")

    # --------------------------------------------------
    # Evaluate on ALL validation trajectories
    # --------------------------------------------------

    mse_list = []

    for traj_id in val_idx:
        X_true = X[:, traj_id, :]
        steps = min(args.steps, X_true.shape[0] - 1)
        X_true = X_true[: steps + 1]

        x0 = X_true[0]

        if args.model == "linear_baseline":
            X_hat = rollout_linear_map(M, x0=x0, steps=steps)

        elif args.model == "dmd_baseline":
            X_hat = rollout_dmd_eig(Lambda, Phi, x0=x0, steps=steps)

        elif args.model == "manual_expansion_manual_dmd":
            X_hat = model.rollout(K=K, C=C, x0=x0, steps=steps).cpu().numpy()

        elif args.model == "manual_expansion_eigen_dmd":
            X_hat = model.rollout(x0=x0, steps=steps).cpu().numpy()
        
        elif args.model == "manual_expansion_ml_dmd":
            X_hat = model.rollout(x0=x0, steps=steps).cpu().numpy()

        elif "eigen" in args.model:
            X_hat = model.rollout(x0=x0, n_steps=steps).cpu().numpy()

        else:
            x0_torch = torch.tensor(x0, dtype=torch.float64)
            X_hat = rollout_ae_model(
                model,
                x0=x0_torch,
                steps=steps,
                device=device,
            ).cpu().numpy()

        mse = np.mean((X_hat - X_true) ** 2)
        mse_list.append(mse)

    mse_mean = np.mean(mse_list)
    mse_std = np.std(mse_list)

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
    X_true = X[:, traj_id, :]
    steps = min(args.steps, X_true.shape[0] - 1)
    X_true = X_true[: steps + 1]

    x0 = X_true[0]

    if args.model == "linear_baseline":
        X_hat = rollout_linear_map(M, x0=x0, steps=steps)

    elif args.model == "dmd_baseline":
        X_hat = rollout_dmd_eig(Lambda, Phi, x0=x0, steps=steps)

    elif args.model == "manual_expansion_manual_dmd":
        X_hat = model.rollout(K=K, C=C, x0=x0, steps=steps).cpu().numpy()
    elif args.model == "ml_eigen_dmd":
        X_hat = model.rollout(x0=x0, n_steps=steps).cpu().numpy()
    else:
        x0_torch = torch.tensor(x0, dtype=torch.float64)
        X_hat = rollout_ae_model(
            model,
            x0=x0_torch,
            steps=steps,
            device=device,
        ).cpu().numpy()
    
    system = os.path.basename(args.data_path).replace("_trajectory.npz", "")
    figdir = f"data/figures/{args.model}/{system}/{args.name if args.name else 'default'}"
    os.makedirs(figdir, exist_ok=True)

    # Plot x1 over time and x2 over time for X_hat
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(X_true[:, 0], label=f"True x1 ({system})")
    plt.plot(X_hat[:, 0], "--", label=f"{args.model}_{system} x1")
    plt.xlabel("Time step")
    plt.ylabel("x1")
    plt.title(f"x1 over time ({args.model}_{system})")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(X_true[:, 1], label=f"True x2 ({system})")
    plt.plot(X_hat[:, 1], "--", label=f"{args.model}_{system} x2")
    plt.xlabel("Time step")
    plt.ylabel("x2")
    plt.title(f"x2 over time ({args.model}_{system})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{figdir}/time_series_idx{args.traj_index}.png")
    plt.show()

    # Plot phase space (x1 vs x2)
    # If it is a Lorenz trajectory, plot x1 and x3 instead (since x2 is just noise around 0)
    if system == "lorenz":
        plt.figure(figsize=(6, 6))
        plt.plot(X_true[:, 0], X_true[:, 2], label=f"True ({system})")
        plt.plot(X_hat[:, 0], X_hat[:, 2], "--", label=f"{args.model}_{system}")
        plt.xlabel("x1")
        plt.ylabel("x3")
        plt.title(f"Phase space rollout ({args.model}_{system})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{figdir}/rollout_idx{args.traj_index}.png")
        plt.show()
        plt.close()
    
    else:
        plt.figure(figsize=(6, 6))
        plt.plot(X_true[:, 0], X_true[:, 1], label=f"True ({system})")
        plt.plot(X_hat[:, 0], X_hat[:, 1], "--", label=f"{args.model}_{system}")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"Phase space rollout ({args.model}_{system})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{figdir}/rollout_idx{args.traj_index}.png")
        plt.show()
        plt.close()
    
    

    # DMD mode
    if args.model == "dmd_baseline":
        
        dt = data["dt"]
        plot_dmd_eigenvalues(
            Lambda,
            savepath=f"{figdir}/eigs_complex.png",
            title=f"DMD Eigenvalues ({args.name if args.name else system})",
        )

        plot_mode_amplitudes(
            Lambda,
            Phi,
            x0,
            savepath=f"{figdir}/mode_amplitudes.png",
        )

        if Phi.shape[0] == 2:
            plot_dmd_modes_2d(
                Phi,
                Lambda,
                savepath=f"{figdir}/modes_geometry.png",
            )

        plot_continuous_spectrum(
            Lambda,
            dt,
            savepath=f"{figdir}/continuous_spectrum.png",
        )

        plot_conjugate_mode_reconstruction(
            Lambda,
            Phi,
            x0,
            steps,
            X_true,
            savepath=f"{figdir}/dominant_mode_reconstruction.png",
        )

if __name__ == "__main__":
    main()