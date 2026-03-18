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