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

        else:
            X_hat = model.rollout(x0=x0, steps=steps).detach().cpu().numpy()

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

    else:
        X_hat = model.rollout(x0=x0, steps=steps).detach().cpu().numpy()

    figdir = f"data/figures/{args.model}/{system}/{args.name if args.name else 'default'}"
    os.makedirs(figdir, exist_ok=True)

    state_dim = X_true.shape[1]

    # --------------------------------------------------
    # Time series plot
    # --------------------------------------------------

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
    plt.savefig(f"{figdir}/time_series_idx{args.traj_index}.png")
    plt.show()
    plt.close()

    # --------------------------------------------------
    # Phase space plot
    # --------------------------------------------------

    if system == "lorenz" and state_dim >= 3:
        i, j = 0, 2
    else:
        i, j = 0, 1

    plt.figure(figsize=(6, 6))
    plt.plot(X_true[:, i], X_true[:, j], label="True")
    plt.plot(X_hat[:, i], X_hat[:, j], "--", label="Prediction")
    plt.xlabel(f"x{i+1}")
    plt.ylabel(f"x{j+1}")
    plt.title(f"Phase space rollout ({args.model}_{system})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{figdir}/rollout_idx{args.traj_index}.png")
    plt.show()
    plt.close()

    # --------------------------------------------------
    # Eigenvalue spectrum (all linear models)
    # --------------------------------------------------

    eigvals = None

    if args.model == "linear_baseline":
        eigvals = np.linalg.eigvals(M)

    elif args.model == "dmd_baseline":
        eigvals = Lambda

    elif args.model == "manual_expansion_manual_dmd":
        eigvals = np.linalg.eigvals(K)

    elif hasattr(model, "K"):
        A = model.K.weight.detach().cpu().numpy()
        eigvals = np.linalg.eigvals(A)

    elif hasattr(model, "Lambda") and hasattr(model, "Phi") and hasattr(model, "Phi_inv"):
        # Phi_np = model.Phi.detach().cpu().numpy()
        # Phi_inv_np = model.Phi_inv.detach().cpu().numpy()
        # Lambda_np = model.Lambda.detach().cpu().numpy()
        # A = Phi_np @ Lambda_np @ Phi_inv_np
        # eigvals = np.linalg.eigvals(A)
        eigvals = np.diag(model.Lambda.detach().cpu().numpy())

    if eigvals is not None:
        plt.figure(figsize=(6, 6))
        plt.scatter(eigvals.real, eigvals.imag)

        circle = plt.Circle((0, 0), 1, color="gray", fill=False)
        plt.gca().add_artist(circle)

        plt.xlabel("Real")
        plt.ylabel("Imag")
        plt.title(f"Eigenvalues ({args.model})")
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.tight_layout()
        plt.savefig(f"{figdir}/eigenvalues.png")
        plt.close()

if __name__ == "__main__":
    main()