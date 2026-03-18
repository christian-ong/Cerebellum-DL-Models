import argparse
import numpy as np
import torch
import os

from src.eval.rollout_eval import evaluate_rollouts, compute_single_rollout
from src.eval.model_io import load_model, infer_run_name
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
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/linear_baseline/saddle_point/default/model.npz
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear/degenerate_node_trajectory.npz --model_path data/models/linear_baseline/degenerate_node/default/model.npz
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear/inward_spiral_trajectory.npz --model_path data/models/linear_baseline/inward_spiral/default/model.npz
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --model_path data/models/linear_baseline/harmonic_oscillator/default/model.npz

# DMD baseline
    python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/dmd_baseline/saddle_point/default/model.npz
    python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/degenerate_node_trajectory.npz --model_path data/models/dmd_baseline/degenerate_node/default/model.npz
    python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/inward_spiral_trajectory.npz --model_path data/models/dmd_baseline/inward_spiral/default/model.npz
    python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --model_path data/models/dmd_baseline/harmonic_oscillator/default/model.npz

# ML DMD
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/ml_dmd/saddle_point/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/degenerate_node_trajectory.npz --model_path data/models/ml_dmd/degenerate_node/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/inward_spiral_trajectory.npz --model_path data/models/ml_dmd/inward_spiral/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --model_path data/models/ml_dmd/harmonic_oscillator/default/model.pt

# ML Eigen DMD
    python -m scripts.eval --model ml_eigen_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/ml_eigen_dmd/saddle_point/default/model.pt
    python -m scripts.eval --model ml_eigen_dmd --data_path data/trajectories/linear/degenerate_node_trajectory.npz --model_path data/models/ml_eigen_dmd/degenerate_node/default/model.pt
    python -m scripts.eval --model ml_eigen_dmd --data_path data/trajectories/linear/inward_spiral_trajectory.npz --model_path data/models/ml_eigen_dmd/inward_spiral/default/model.pt
    python -m scripts.eval --model ml_eigen_dmd --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --model_path data/models/ml_eigen_dmd/harmonic_oscillator/default/model.pt
---------------------------------------------------------------------------------------------

# Manual expansion + Manual DMD
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/saddle_point/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/linear/degenerate_node_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/degenerate_node/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/linear/inward_spiral_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/inward_spiral/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/harmonic_oscillator/default/model.npz

    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/vanderpol_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/vanderpol/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/lotka_volterra_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/lotka_volterra/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/pendulum_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/pendulum/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/duffing_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/duffing/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/lorenz_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/lorenz/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/koopman_poly_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/koopman_poly/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/koopman_poly_large_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/koopman_poly_large/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/koopman_poly_trig_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/koopman_poly_trig/default/model.npz

# Manual expansion + ML DMD
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/saddle_point/default/model.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/linear/degenerate_node_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/degenerate_node/default/model.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/linear/inward_spiral_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/inward_spiral/default/model.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/harmonic_oscillator/default/model.pt

    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/koopman_poly_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/koopman_poly/default/model.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/koopman_poly_large_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/koopman_poly_large/default/model.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/koopman_poly_trig_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/koopman_poly_trig/default/model.pt

    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/vanderpol_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/vanderpol/default/model.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/lotka_volterra_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/lotka_volterra/default/model.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/pendulum_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/pendulum/default/model.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/duffing_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/duffing/default/model.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/lorenz_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/lorenz/default/model.pt

# Manual expansion + Eigen DMD
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/saddle_point/default/model.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/linear/degenerate_node_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/degenerate_node/default/model.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/linear/inward_spiral_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/inward_spiral/default/model.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/harmonic_oscillator/default/model.pt

    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/koopman_poly_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/koopman_poly/default/model.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/koopman_poly_large_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/koopman_poly_large/default/model.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/koopman_poly_trig_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/koopman_poly_trig/default/model.pt

    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/vanderpol_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/vanderpol/default/model.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/lotka_volterra_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/lotka_volterra/default/model.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/pendulum_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/pendulum/default/model.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/duffing_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/duffing/default/model.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/lorenz_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/lorenz/default/model.pt

# SINDy baseline
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/sindy_baseline/saddle_point/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/linear/degenerate_node_trajectory.npz --model_path data/models/sindy_baseline/degenerate_node/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/linear/inward_spiral_trajectory.npz --model_path data/models/sindy_baseline/inward_spiral/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --model_path data/models/sindy_baseline/harmonic_oscillator/default/model.npz

    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/vanderpol_trajectory.npz --model_path data/models/sindy_baseline/vanderpol/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/lotka_volterra_trajectory.npz --model_path data/models/sindy_baseline/lotka_volterra/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/pendulum_trajectory.npz --model_path data/models/sindy_baseline/pendulum/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/duffing_trajectory.npz --model_path data/models/sindy_baseline/duffing/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/lorenz_trajectory.npz --model_path data/models/sindy_baseline/lorenz/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/koopman_poly_trajectory.npz --model_path data/models/sindy_baseline/koopman_poly/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/koopman_poly_large_trajectory.npz --model_path data/models/sindy_baseline/koopman_poly_large/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/koopman_poly_trig_trajectory.npz --model_path data/models/sindy_baseline/koopman_poly_trig/default/model.npz
        
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
                            "sindy_baseline",
                        ])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--traj_index", type=int, default=0, help="Which test trajectory to show")
    parser.add_argument("--name", type=str, help="Optional suffix for saved figure")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    data = np.load(args.data_path)
    X = data["X"]
    state_dim = X.shape[-1]

    if "test_idx" not in data:
        raise ValueError(
            "Dataset does not contain test_idx. "
            "Please regenerate it using simulate_data.py."
        )

    test_idx = data["test_idx"]

    if X.ndim != 3:
        raise ValueError("Evaluation expects multiple trajectories (X must be 3D).")

    if len(test_idx) == 0:
        raise ValueError("No test trajectories available.")

    system = os.path.basename(args.data_path).replace("_trajectory.npz", "")
    print(f"Loaded {X.shape[1]} trajectories for system '{system}', with {len(test_idx)} test trajectories.")

    # --------------------------------------------------
    # Load model once via shared backend
    # --------------------------------------------------

    model, extras = load_model(
        model_name=args.model,
        model_path=args.model_path,
        data_path=args.data_path,
        state_dim=state_dim,
        system=system,
        device=device,
    )

    # --------------------------------------------------
    # Evaluate on all TEST trajectories
    # --------------------------------------------------

    mse_mean, mse_std, _ = evaluate_rollouts(
        X=X,
        traj_indices=test_idx,
        model_name=args.model,
        model=model,
        steps=args.steps,
        extras=extras,
    )

    print(
        f"Test rollout MSE over {len(test_idx)} trajectories: "
        f"{mse_mean:.6e} ± {mse_std:.6e}"
    )

    # --------------------------------------------------
    # Plot ONE test trajectory
    # --------------------------------------------------

    if args.traj_index >= len(test_idx):
        raise IndexError(
            f"traj_index={args.traj_index} but only {len(test_idx)} test trajectories exist."
        )

    traj_id = test_idx[args.traj_index]

    X_true, X_hat = compute_single_rollout(
        X=X,
        traj_id=traj_id,
        steps=args.steps,
        model_name=args.model,
        model=model,
        extras=extras,
    )

    # --------------------------------------------------
    # Figure directory
    # --------------------------------------------------

    run_name = infer_run_name(args.model_path, args.name)

    figdir = os.path.join("data", "figures", args.model, system, run_name)
    os.makedirs(figdir, exist_ok=True)

    # --------------------------------------------------
    # Save trajectory plots
    # --------------------------------------------------

    plot_time_series(
        X_true,
        X_hat,
        figdir,
        args.traj_index,
    )

    plot_phase_space(
        X_true,
        X_hat,
        system,
        figdir,
        args.model,
        args.traj_index,
    )

    # --------------------------------------------------
    # Plot eigenvalues if available
    # --------------------------------------------------

    eigvals = None

    if args.model == "linear_baseline":
        eigvals = np.linalg.eigvals(extras["M"])

    elif args.model == "dmd_baseline":
        eigvals = extras["Lambda"]

    elif args.model == "manual_expansion_manual_dmd":
        eigvals = np.linalg.eigvals(extras["K"])

    elif model is not None and hasattr(model, "Lambda"):
        lam = model.Lambda
        if torch.is_tensor(lam):
            eigvals = lam.detach().cpu().numpy()
        else:
            eigvals = np.asarray(lam)

    if eigvals is not None:
        plot_eigenvalues(eigvals, figdir)

    # --------------------------------------------------
    # Plot training losses if available
    # --------------------------------------------------

    if args.model_path.endswith(".pt") and "ckpt" in extras:
        ckpt = extras["ckpt"]
        if "train_losses" in ckpt and "val_losses" in ckpt:
            plot_training_losses(
                ckpt["train_losses"],
                ckpt["val_losses"],
                figdir,
            )

    # --------------------------------------------------
    # Plot transition matrix if available
    # --------------------------------------------------

    model_name = os.path.basename(args.model_path).replace(".pt", "").replace(".npz", "")

    expand_names = None
    matrix_to_plot = None

    if args.model == "manual_expansion_manual_dmd":
        matrix_to_plot = extras["K"]
        expand_names = model.expand_names if hasattr(model, "expand_names") else None

    elif args.model_path.endswith(".pt") and "ckpt" in extras:
        if "expand_names" in extras["ckpt"]:
            expand_names = extras["ckpt"]["expand_names"]

    plot_transition_matrix(
        model=None if matrix_to_plot is not None else model,
        matrix=matrix_to_plot,
        model_name=model_name,
        figdir=figdir,
        expand_names=expand_names,
    )

    # --------------------------------------------------
    # Compare learned state block with true A_d
    # --------------------------------------------------

    if model is not None and hasattr(model, "Phi") and hasattr(model, "Lambda"):

        print("\n--- Learned lifted operator ---")

        Phi = model.Phi.detach().cpu().numpy()
        Lambda = model.Lambda.detach().cpu().numpy()

        try:
            Phi_inv = np.linalg.inv(Phi)
        except np.linalg.LinAlgError:
            Phi_inv = np.linalg.pinv(Phi)

        K = Phi @ Lambda @ Phi_inv

        print("Full lifted transition matrix shape:", K.shape)

        # extract state indices (x,y or x,y,z)
        if hasattr(model, "state_indices"):
            state_idx = model.state_indices
            K_xx = K[np.ix_(state_idx, state_idx)]
        else:
            # no lifting → the whole matrix is the state block
            K_xx = K

        print("\nState-space block K_xx:")
        print(K_xx)

        # If linear system, also print true A_d if available
        if args.model in ["manual_expansion_eigen_dmd"] and system in [
            "saddle_point",
            "degenerate_node",
            "inward_spiral",
            "harmonic_oscillator",
        ]:
            print("\nCompare this with true A_d from Overleaf.")

    # --------------------------------------------------
    # Compare learned state block with true A_d
    # --------------------------------------------------

    if model is not None and hasattr(model, "K"):

        print("\n--- Learned lifted Koopman operator ---")

        # Full lifted operator
        K = model.K.weight.detach().cpu().numpy().T

        print("Full lifted transition matrix shape:", K.shape)

        # Extract state block
        if hasattr(model, "state_indices"):
            state_idx = model.state_indices
            K_xx = K[np.ix_(state_idx, state_idx)]
        else:
            # no lifting → the whole matrix is the state block
            K_xx = K

        print("\nState-space block K_xx:")
        print(K_xx)

        if system in [
            "saddle_point",
            "degenerate_node",
            "inward_spiral",
            "harmonic_oscillator",
        ]:
            print("\nCompare this with true A_d from Overleaf.")


if __name__ == "__main__":
    main()

# import argparse
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import os

# from src.models.linear_baseline import rollout_linear_map
# from src.models.dmd_baseline import rollout_dmd_eig
# from src.models.ml_dmd import ML_DMD
# from src.models.ml_eigen_dmd import MLEigenDMD
# from src.models.manual_expansion_ml_dmd import ManualExpansion_MLDMD
# from src.models.manual_expansion_manual_dmd import ManualExpansion_ManualDMD
# from src.models.manual_expansion_eigen_dmd import ManualExpansion_EigenDMD
# from src.models.sindy_baseline import SINDyBaseline

# from src.eval.rollout_eval import evaluate_validation_rollouts, compute_single_rollout
# from src.eval.plot_rollout import plot_time_series, plot_phase_space
# from src.eval.plot_eigenvalues import plot_eigenvalues
# from src.eval.plot_training_losses import plot_training_losses
# from src.eval.plot_matrices import plot_transition_matrix
# from torch.utils.data import DataLoader
# from src.data_generation.load_data import OneStepTrajectoryDataset

# """
# Global options (defaults):
#     --model {
#         linear_baseline,
#         dmd_baseline,
#         ml_dmd,
#         manual_expansion_ml_dmd,
#         manual_expansion_manual_dmd,
#         manual_expansion_eigen_dmd}
#     --data_path data/trajectories/{system}_trajectory.npz
#     --model_path data/models/{model}_{system}.pt
#     --steps 5000
#     --traj_index 0
#     --name optional_suffix

# ---------------------------------------------------------------------------------------------

# # Linear baseline
#     python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/linear_baseline/saddle_point/default/model.npz
#     python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear/degenerate_node_trajectory.npz --model_path data/models/linear_baseline/degenerate_node/default/model.npz
#     python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear/inward_spiral_trajectory.npz --model_path data/models/linear_baseline/inward_spiral/default/model.npz
#     python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --model_path data/models/linear_baseline/harmonic_oscillator/default/model.npz

# # DMD baseline
#     python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/dmd_baseline/saddle_point/default/model.npz
#     python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/degenerate_node_trajectory.npz --model_path data/models/dmd_baseline/degenerate_node/default/model.npz
#     python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/inward_spiral_trajectory.npz --model_path data/models/dmd_baseline/inward_spiral/default/model.npz
#     python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --model_path data/models/dmd_baseline/harmonic_oscillator/default/model.npz

# # ML DMD
#     python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/ml_dmd/saddle_point/default/model.pt
#     python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/degenerate_node_trajectory.npz --model_path data/models/ml_dmd/degenerate_node/default/model.pt
#     python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/inward_spiral_trajectory.npz --model_path data/models/ml_dmd/inward_spiral/default/model.pt
#     python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --model_path data/models/ml_dmd/harmonic_oscillator/default/model.pt

# # ML Eigen DMD
#     python -m scripts.eval --model ml_eigen_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/ml_eigen_dmd/saddle_point/default/model.pt
#     python -m scripts.eval --model ml_eigen_dmd --data_path data/trajectories/linear/degenerate_node_trajectory.npz --model_path data/models/ml_eigen_dmd/degenerate_node/default/model.pt
#     python -m scripts.eval --model ml_eigen_dmd --data_path data/trajectories/linear/inward_spiral_trajectory.npz --model_path data/models/ml_eigen_dmd/inward_spiral/default/model.pt
#     python -m scripts.eval --model ml_eigen_dmd --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --model_path data/models/ml_eigen_dmd/harmonic_oscillator/default/model.pt
# ---------------------------------------------------------------------------------------------

# # Manual expansion + Manual DMD
#     python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/saddle_point/default/model.npz
#     python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/linear/degenerate_node_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/degenerate_node/default/model.npz
#     python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/linear/inward_spiral_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/inward_spiral/default/model.npz
#     python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/harmonic_oscillator/default/model.npz

#     python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/vanderpol_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/vanderpol/default/model.npz
#     python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/lotka_volterra_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/lotka_volterra/default/model.npz
#     python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/pendulum_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/pendulum/default/model.npz
#     python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/duffing_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/duffing/default/model.npz
#     python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/lorenz_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/lorenz/default/model.npz
#     python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/koopman_poly_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/koopman_poly/default/model.npz
#     python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/koopman_poly_large_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/koopman_poly_large/default/model.npz
#     python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/koopman_poly_trig_trajectory.npz --model_path data/models/manual_expansion_manual_dmd/koopman_poly_trig/default/model.npz

# # Manual expansion + ML DMD
#     python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/saddle_point/default/model.pt
#     python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/linear/degenerate_node_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/degenerate_node/default/model.pt
#     python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/linear/inward_spiral_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/inward_spiral/default/model.pt
#     python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/harmonic_oscillator/default/model.pt

#     python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/koopman_poly_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/koopman_poly/default/model.pt
#     python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/koopman_poly_large_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/koopman_poly_large/default/model.pt
#     python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/koopman_poly_trig_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/koopman_poly_trig/default/model.pt

#     python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/vanderpol_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/vanderpol/default/model.pt
#     python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/lotka_volterra_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/lotka_volterra/default/model.pt
#     python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/pendulum_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/pendulum/default/model.pt
#     python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/duffing_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/duffing/default/model.pt
#     python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/lorenz_trajectory.npz --model_path data/models/manual_expansion_ml_dmd/lorenz/default/model.pt

# # Manual expansion + Eigen DMD
#     python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/saddle_point/default/model.pt
#     python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/linear/degenerate_node_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/degenerate_node/default/model.pt
#     python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/linear/inward_spiral_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/inward_spiral/default/model.pt
#     python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/harmonic_oscillator/default/model.pt

#     python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/koopman_poly_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/koopman_poly/default/model.pt
#     python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/koopman_poly_large_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/koopman_poly_large/default/model.pt
#     python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/koopman_poly_trig_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/koopman_poly_trig/default/model.pt

#     python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/vanderpol_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/vanderpol/default/model.pt
#     python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/lotka_volterra_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/lotka_volterra/default/model.pt
#     python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/pendulum_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/pendulum/default/model.pt
#     python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/duffing_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/duffing/default/model.pt
#     python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/lorenz_trajectory.npz --model_path data/models/manual_expansion_eigen_dmd/lorenz/default/model.pt

# # SINDy baseline
#     python -m scripts.eval --model sindy_baseline --data_path data/trajectories/linear/saddle_point_trajectory.npz --model_path data/models/sindy_baseline/saddle_point/default/model.npz
#     python -m scripts.eval --model sindy_baseline --data_path data/trajectories/linear/degenerate_node_trajectory.npz --model_path data/models/sindy_baseline/degenerate_node/default/model.npz
#     python -m scripts.eval --model sindy_baseline --data_path data/trajectories/linear/inward_spiral_trajectory.npz --model_path data/models/sindy_baseline/inward_spiral/default/model.npz
#     python -m scripts.eval --model sindy_baseline --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --model_path data/models/sindy_baseline/harmonic_oscillator/default/model.npz

#     python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/vanderpol_trajectory.npz --model_path data/models/sindy_baseline/vanderpol/default/model.npz
#     python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/lotka_volterra_trajectory.npz --model_path data/models/sindy_baseline/lotka_volterra/default/model.npz
#     python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/pendulum_trajectory.npz --model_path data/models/sindy_baseline/pendulum/default/model.npz
#     python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/duffing_trajectory.npz --model_path data/models/sindy_baseline/duffing/default/model.npz
#     python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/lorenz_trajectory.npz --model_path data/models/sindy_baseline/lorenz/default/model.npz
#     python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/koopman_poly_trajectory.npz --model_path data/models/sindy_baseline/koopman_poly/default/model.npz
#     python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/koopman_poly_large_trajectory.npz --model_path data/models/sindy_baseline/koopman_poly_large/default/model.npz
#     python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/koopman_poly_trig_trajectory.npz --model_path data/models/sindy_baseline/koopman_poly_trig/default/model.npz
        
# ---------------------------------------------------------------------------------------------

# Output:
#     data/figures/{model}/{system}/{name}/time_series_idx{traj_index}.png
#     data/figures/{model}/{system}/{name}/rollout_idx{traj_index}.png
# """

# def main():
#     parser = argparse.ArgumentParser(description="Evaluate trained models")

#     parser.add_argument("--model", type=str, required=True,
#                         choices=[
#                             "linear_baseline",
#                             "dmd_baseline",
#                             "ml_dmd",
#                             "ml_eigen_dmd",
#                             "manual_expansion_ml_dmd",
#                             "manual_expansion_manual_dmd",
#                             "manual_expansion_eigen_dmd",
#                             "sindy_baseline",
#                         ])
#     parser.add_argument("--data_path", type=str, required=True)
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--steps", type=int, default=5000)
#     parser.add_argument("--traj_index", type=int, default=0, help="Which validation trajectory to show")
#     parser.add_argument("--name", type=str, help="Optional suffix for saved figure")
    
#     args = parser.parse_args()

#     device = "cuda" if torch.cuda.is_available() else "cpu"


#     # --------------------------------------------------
#     # Load data
#     # --------------------------------------------------

#     data = np.load(args.data_path)
#     X = data["X"]
#     state_dim = X.shape[-1]

#     if "test_idx" not in data:
#         raise ValueError(
#             "Dataset does not contain test_idx. "
#             "Please regenerate it using simulate_data.py."
#         )

#     test_idx = data["test_idx"]

#     if X.ndim != 3:
#         raise ValueError("Evaluation expects multiple trajectories (X must be 3D).")

#     if len(test_idx) == 0:
#         raise ValueError("No test trajectories available.")
    
#     system = os.path.basename(args.data_path).replace("_trajectory.npz", "")
#     print(f"Loaded {X.shape[1]} trajectories for system '{system}', with {len(test_idx)} test trajectories.")
#     # --------------------------------------------------
#     # Load model ONCE
#     # --------------------------------------------------

#     if args.model == "linear_baseline":
#         model_data = np.load(args.model_path)
#         M = model_data["M"]
#         model = None

#     elif args.model == "dmd_baseline":
#         model_data = np.load(args.model_path)
#         Lambda = model_data["Lambda"]
#         Phi = model_data["Phi"]
#         model = None

#     elif args.model == "manual_expansion_manual_dmd":
#         model_data = np.load(args.model_path, allow_pickle=True)
#         K = model_data["K"]

#         if "C" not in model_data:
#             raise ValueError(
#                 "Checkpoint is missing decoder matrix C. "
#                 "Please retrain manual_expansion_manual_dmd with the updated EDMD-style implementation."
#             )
#         C = model_data["C"]

#         degree = int(model_data["expansion_degree"]) if "expansion_degree" in model_data else 3

#         # Backward compatibility: old checkpoints used include_bias,
#         # newer ones use constant_expansion
#         if "bias" in model_data:
#             bias = bool(np.asarray(model_data["bias"]).item())
#         elif "include_bias" in model_data:
#             bias = bool(np.asarray(model_data["include_bias"]).item())
#         else:
#             bias = True

#         if "sine_cosine_expansion" in model_data:
#             sine_cosine_expansion = bool(np.asarray(model_data["sine_cosine_expansion"]).item())
#         else:
#             sine_cosine_expansion = False

#         expansion_type = str(model_data["expansion_type"]) if "expansion_type" in model_data else "general"

#         if "system_basis" in model_data:
#             system_basis = str(model_data["system_basis"])
#             if system_basis == "":
#                 system_basis = None
#         else:
#             system_basis = system if expansion_type == "specific" else None

#         model = ManualExpansion_ManualDMD(
#             state_dim=state_dim,
#             expansion_degree=degree,
#             bias=bias,
#             sine_cosine_expansion=sine_cosine_expansion,
#             expansion_type=expansion_type,
#             system=system_basis,
#         ).to(device)
#         model.eval()

#     elif args.model == "ml_dmd":
#         ckpt = torch.load(args.model_path, map_location=device)
#         model = ML_DMD(state_dim=ckpt["state_dim"]).to(device)
#         model.load_state_dict(ckpt["model_state_dict"])
#         model.eval()
    
#     elif args.model == "ml_eigen_dmd":
#         ckpt = torch.load(args.model_path, map_location=device)
#         model = MLEigenDMD(state_dim=ckpt["state_dim"],).to(device)
#         model.load_state_dict(ckpt["model_state_dict"])
#         model.eval()

#     elif args.model == "manual_expansion_ml_dmd":
#         ckpt = torch.load(args.model_path, map_location=device)
#         train_args = ckpt["train_args"]

#         model = ManualExpansion_MLDMD(
#             state_dim=ckpt["state_dim"],
#             expansion_degree=train_args["expansion_degree"],
#             expansion_type=train_args["expansion_type"],
#             bias=train_args["bias"] == "true",
#             sine_cosine_expansion=train_args["sine_cosine_expansion"] == "true",
#             system=ckpt["system"],
#         ).to(device)
#         model.load_state_dict(ckpt["model_state_dict"])
#         model.eval()

#     elif args.model == "manual_expansion_eigen_dmd":

#         ckpt = torch.load(args.model_path, map_location=device)

#         train_args = ckpt["train_args"]

#         model = ManualExpansion_EigenDMD(
#             state_dim=ckpt["state_dim"],
#             expansion_degree=train_args["expansion_degree"],
#             bias=train_args["bias"] == "true",
#             sine_cosine_expansion=train_args["sine_cosine_expansion"] == "true",
#             expansion_type=train_args["expansion_type"],
#             system=ckpt["system"] if train_args["expansion_type"] == "specific" else None,
#         ).to(device)

#         model.load_state_dict(ckpt["model_state_dict"])
#         model.eval()

#     elif args.model == "sindy_baseline":
#         model_data = np.load(args.model_path, allow_pickle=True)

#         specific_system = str(np.asarray(model_data["specific_system"]).item())
#         if specific_system == "":
#             specific_system = None

#         specific_basis_size = int(np.asarray(model_data["specific_basis_size"]).item())
#         if specific_basis_size < 0:
#             specific_basis_size = None

#         model = SINDyBaseline(
#             discrete_time=bool(np.asarray(model_data["discrete_time"]).item()),
#             poly_order=int(np.asarray(model_data["poly_order"]).item()),
#             include_bias=bool(np.asarray(model_data["include_bias"]).item()),
#             include_interaction=bool(np.asarray(model_data["include_interaction"]).item()),
#             threshold=float(np.asarray(model_data["threshold"]).item()),
#             alpha=float(np.asarray(model_data["alpha"]).item()),
#             differentiation_method=str(np.asarray(model_data["diff_method"]).item()),
#             library_type=str(np.asarray(model_data["library_type"]).item()),
#             fourier_n_frequencies=int(np.asarray(model_data["fourier_n_frequencies"]).item()),
#             specific_system=specific_system,
#             specific_basis_size=specific_basis_size,
#         )

#         train_ds = OneStepTrajectoryDataset(args.data_path, split="train")
#         train_loader = DataLoader(train_ds, batch_size=4096, shuffle=False)

#         X_train_list, Y_train_list = [], []
#         for x, y in train_loader:
#             X_train_list.append(x.numpy())
#             Y_train_list.append(y.numpy())

#         X_train = np.vstack(X_train_list)
#         Y_train = np.vstack(Y_train_list)

#         if model.discrete_time:
#             model.fit_discrete_pairs(X_train, Y_train)
#         else:
#             meta_data = np.load(args.data_path)
#             X_all = meta_data["X"]
#             train_idx = meta_data["train_idx"]
#             dt = float(meta_data["dt"])
#             X_train = X_all[:, train_idx, :]
#             model.fit_continuous_trajectories(X_train, dt=dt)



#     else:
#         raise ValueError(f"Unknown model: {args.model}")

#     # --------------------------------------------------
#     # Evaluate on ALL validation trajectories
#     # --------------------------------------------------

#     mse_mean, mse_std = evaluate_validation_rollouts(
#         X=X,
#         test_idx=test_idx,
#         model_name=args.model,
#         model=model,
#         steps=args.steps,
#         rollout_linear_map=rollout_linear_map,
#         rollout_dmd_eig=rollout_dmd_eig,
#         M=locals().get("M"),
#         Lambda=locals().get("Lambda"),
#         Phi=locals().get("Phi"),
#         K=locals().get("K"),
#         C=locals().get("C"),
#     )

#     print(
#         f"Test rollout MSE over {len(test_idx)} trajectories: "
#         f"{mse_mean:.6e} ± {mse_std:.6e}"
#     )

#     # --------------------------------------------------
#     # Plot ONE test trajectory
#     # --------------------------------------------------

#     if args.traj_index >= len(test_idx):
#         raise IndexError(
#             f"traj_index={args.traj_index} but only {len(test_idx)} test trajectories exist."
#         )

#     traj_id = test_idx[args.traj_index]

#     X_true, X_hat = compute_single_rollout(
#         X=X,
#         traj_id=traj_id,
#         steps=args.steps,
#         model_name=args.model,
#         model=model,
#         rollout_linear_map=rollout_linear_map,
#         rollout_dmd_eig=rollout_dmd_eig,
#         M=locals().get("M"),
#         Lambda=locals().get("Lambda"),
#         Phi=locals().get("Phi"),
#         K=locals().get("K"),
#         C=locals().get("C"),
#     )

#     # --------------------------------------------------
#     # Figure directory
#     # --------------------------------------------------

#     if args.name is not None:
#         run_name = args.name
#     else:
#         # Infer run name from model path, e.g.
#         # data/models/manual_expansion_eigen_dmd/lorenz/deg7/model.pt -> deg7
#         run_name = os.path.basename(os.path.dirname(args.model_path))

#     figdir = os.path.join("data", "figures", args.model, system, run_name)
#     os.makedirs(figdir, exist_ok=True)

#     print(f"Saving figures to: {figdir}")
#     # --------------------------------------------------
#     # Plots
#     # --------------------------------------------------

#     plot_time_series(X_true, X_hat, figdir, args.traj_index)

#     plot_phase_space(
#         X_true,
#         X_hat,
#         system,
#         figdir,
#         args.model,
#         args.traj_index,
#     )

#     # --------------------------------------------------
#     # Eigenvalue spectrum
#     # --------------------------------------------------

#     eigvals = None

#     if args.model == "linear_baseline":
#         eigvals = np.linalg.eigvals(M)

#     elif args.model == "dmd_baseline":
#         eigvals = Lambda

#     elif args.model == "manual_expansion_manual_dmd":
#         eigvals = np.linalg.eigvals(K)

#     elif model is not None and hasattr(model, "K"):
#         A = model.K.weight.detach().cpu().numpy().T
#         eigvals = np.linalg.eigvals(A)

#     elif model is not None and hasattr(model, "Lambda"):
#         Lambda = model.Lambda.detach().cpu().numpy()
#         eigvals = np.linalg.eigvals(Lambda)

#     if eigvals is not None:
#         plot_eigenvalues(eigvals, figdir)

#     # --------------------------------------------------
#     # Training loss plots
#     # --------------------------------------------------
#     if ".pt" in args.model_path:
#         loss_file = args.model_path.replace("model.pt", "losses.npz")

#         if os.path.exists(loss_file):
#             plot_training_losses(loss_file, figdir)
#         else:
#             print(f"No loss file found at {loss_file}, skipping loss plots.")

#     # --------------------------------------------------
#     # Transition matrix visualization
#     # --------------------------------------------------
#     model_name = os.path.basename(args.model_path).replace(".pt", "").replace(".npz", "")

#     expand_names = None
#     matrix_to_plot = None

#     if args.model == "manual_expansion_manual_dmd":
#         matrix_to_plot = K
#         expand_names = model.expand_names if hasattr(model, "expand_names") else None

#     elif ".pt" in args.model_path and "ckpt" in locals():
#         if "expand_names" in ckpt:
#             expand_names = ckpt["expand_names"]

#     plot_transition_matrix(
#         model=None if matrix_to_plot is not None else model,
#         matrix=matrix_to_plot,
#         model_name=model_name,
#         figdir=figdir,
#         expand_names=expand_names,
#     )

#     # --------------------------------------------------
#     # Compare learned state block with true A_d
#     # --------------------------------------------------

#     if model is not None and hasattr(model, "Phi") and hasattr(model, "Lambda"):

#         print("\n--- Learned lifted operator ---")

#         Phi = model.Phi.detach().cpu().numpy()
#         Lambda = model.Lambda.detach().cpu().numpy()

#         try:
#             Phi_inv = np.linalg.inv(Phi)
#         except np.linalg.LinAlgError:
#             Phi_inv = np.linalg.pinv(Phi)

#         K = Phi @ Lambda @ Phi_inv

#         print("Full lifted transition matrix shape:", K.shape)

#         # extract state indices (x,y or x,y,z)
#         if hasattr(model, "state_indices"):
#             state_idx = model.state_indices
#             K_xx = K[np.ix_(state_idx, state_idx)]
#         else:
#             # no lifting → the whole matrix is the state block
#             K_xx = K

#         print("\nState-space block K_xx:")
#         print(K_xx)

#         # If linear system, also print true A_d if available
#         if args.model in ["manual_expansion_eigen_dmd"] and system in [
#             "saddle_point",
#             "degenerate_node",
#             "inward_spiral",
#             "harmonic_oscillator",
#         ]:
#             print("\nCompare this with true A_d from Overleaf.")
            
#     # --------------------------------------------------
#     # Compare learned state block with true A_d
#     # --------------------------------------------------

#     if model is not None and hasattr(model, "K"):

#         print("\n--- Learned lifted Koopman operator ---")

#         # Full lifted operator
#         K = model.K.weight.detach().cpu().numpy().T

#         print("Full lifted transition matrix shape:", K.shape)

#         # Extract state block
#         if hasattr(model, "state_indices"):
#             state_idx = model.state_indices
#             K_xx = K[np.ix_(state_idx, state_idx)]
#         else:
#             # no lifting → the whole matrix is the state block
#             K_xx = K

#         print("\nState-space block K_xx:")
#         print(K_xx)

#         if system in [
#             "saddle_point",
#             "degenerate_node",
#             "inward_spiral",
#             "harmonic_oscillator",
#         ]:
#             print("\nCompare this with true A_d from Overleaf.")

# if __name__ == "__main__":
#     main()