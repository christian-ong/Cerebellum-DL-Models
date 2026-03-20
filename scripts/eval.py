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
from src.eval.diagnostics import parse_int_list, run_diagnostics
from src.eval.metrics import (
    compute_one_step_metrics,
    compute_horizon_metrics,
    compute_full_rollout_metrics,
    compute_composite_validation_score,
    get_state_scale_from_train_split,
    save_summary_npz,
    build_rollout_cache,
)
from src.data_generation.load_data import resolve_split_npz_path
"""
Global options (defaults):
    --model {
        linear_baseline,
        dmd_baseline,
        ml_dmd,
        manual_expansion_ml_dmd,
        manual_expansion_manual_dmd,
        manual_expansion_eigen_dmd}
    --data_path data/trajectories/{linear|nonlinear}/{system}
    --model_path data/models/{model}/{system}/{name}/model.pt
    --steps 5000
    --traj_index 0
    --name optional_suffix

---------------------------------------------------------------------------------------------

# Linear baseline
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear/saddle_point --model_path data/models/linear_baseline/saddle_point/default/model.npz
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear/degenerate_node --model_path data/models/linear_baseline/degenerate_node/default/model.npz
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear/inward_spiral --model_path data/models/linear_baseline/inward_spiral/default/model.npz
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear/harmonic_oscillator --model_path data/models/linear_baseline/harmonic_oscillator/default/model.npz

# DMD baseline
    python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/saddle_point --model_path data/models/dmd_baseline/saddle_point/default/model.npz
    python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/degenerate_node --model_path data/models/dmd_baseline/degenerate_node/default/model.npz
    python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/inward_spiral --model_path data/models/dmd_baseline/inward_spiral/default/model.npz
    python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/harmonic_oscillator --model_path data/models/dmd_baseline/harmonic_oscillator/default/model.npz

# ML DMD
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/ml_dmd/saddle_point/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/degenerate_node --model_path data/models/ml_dmd/degenerate_node/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/inward_spiral --model_path data/models/ml_dmd/inward_spiral/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/harmonic_oscillator --model_path data/models/ml_dmd/harmonic_oscillator/default/model.pt

# ML Eigen DMD
    python -m scripts.eval --model ml_eigen_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/ml_eigen_dmd/saddle_point/default/model.pt
    python -m scripts.eval --model ml_eigen_dmd --data_path data/trajectories/linear/degenerate_node --model_path data/models/ml_eigen_dmd/degenerate_node/default/model.pt
    python -m scripts.eval --model ml_eigen_dmd --data_path data/trajectories/linear/inward_spiral --model_path data/models/ml_eigen_dmd/inward_spiral/default/model.pt
    python -m scripts.eval --model ml_eigen_dmd --data_path data/trajectories/linear/harmonic_oscillator --model_path data/models/ml_eigen_dmd/harmonic_oscillator/default/model.pt

---------------------------------------------------------------------------------------------

# Manual expansion + Manual DMD
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/manual_expansion_manual_dmd/saddle_point/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/linear/degenerate_node --model_path data/models/manual_expansion_manual_dmd/degenerate_node/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/linear/inward_spiral --model_path data/models/manual_expansion_manual_dmd/inward_spiral/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/linear/harmonic_oscillator --model_path data/models/manual_expansion_manual_dmd/harmonic_oscillator/default/model.npz

    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/manual_expansion_manual_dmd/vanderpol/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/lotka_volterra --model_path data/models/manual_expansion_manual_dmd/lotka_volterra/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/pendulum --model_path data/models/manual_expansion_manual_dmd/pendulum/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/duffing --model_path data/models/manual_expansion_manual_dmd/duffing/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/lorenz --model_path data/models/manual_expansion_manual_dmd/lorenz/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/koopman_poly --model_path data/models/manual_expansion_manual_dmd/koopman_poly/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/koopman_poly_large --model_path data/models/manual_expansion_manual_dmd/koopman_poly_large/default/model.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/koopman_poly_trig --model_path data/models/manual_expansion_manual_dmd/koopman_poly_trig/default/model.npz
    
    # Final test evaluation + print matching validation summary + save test_summary.npz
    python -m scripts.eval --model manual_expansion_manual_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/manual_expansion_manual_dmd/saddle_point/default/model.npz --print_validation_summary --horizons 1,2,5,10

# Manual expansion + ML DMD
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/manual_expansion_ml_dmd/saddle_point/default/model.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/linear/degenerate_node --model_path data/models/manual_expansion_ml_dmd/degenerate_node/default/model.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/linear/inward_spiral --model_path data/models/manual_expansion_ml_dmd/inward_spiral/default/model.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/linear/harmonic_oscillator --model_path data/models/manual_expansion_ml_dmd/harmonic_oscillator/default/model.pt

    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/koopman_poly --model_path data/models/manual_expansion_ml_dmd/koopman_poly/default/model.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/koopman_poly_large --model_path data/models/manual_expansion_ml_dmd/koopman_poly_large/default/model.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/koopman_poly_trig --model_path data/models/manual_expansion_ml_dmd/koopman_poly_trig/default/model.pt

    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/manual_expansion_ml_dmd/vanderpol/default/model.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/lotka_volterra --model_path data/models/manual_expansion_ml_dmd/lotka_volterra/default/model.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/pendulum --model_path data/models/manual_expansion_ml_dmd/pendulum/default/model.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/duffing --model_path data/models/manual_expansion_ml_dmd/duffing/default/model.pt
    python -m scripts.eval --model manual_expansion_ml_dmd --data_path data/trajectories/nonlinear/lorenz --model_path data/models/manual_expansion_ml_dmd/lorenz/default/model.pt

# Manual expansion + Eigen DMD
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/manual_expansion_eigen_dmd/saddle_point/default/model.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/linear/degenerate_node --model_path data/models/manual_expansion_eigen_dmd/degenerate_node/default/model.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/linear/inward_spiral --model_path data/models/manual_expansion_eigen_dmd/inward_spiral/default/model.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/linear/harmonic_oscillator --model_path data/models/manual_expansion_eigen_dmd/harmonic_oscillator/default/model.pt

    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/koopman_poly --model_path data/models/manual_expansion_eigen_dmd/koopman_poly/default/model.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/koopman_poly_large --model_path data/models/manual_expansion_eigen_dmd/koopman_poly_large/default/model.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/koopman_poly_trig --model_path data/models/manual_expansion_eigen_dmd/koopman_poly_trig/default/model.pt

    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/manual_expansion_eigen_dmd/vanderpol/default/model.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/lotka_volterra --model_path data/models/manual_expansion_eigen_dmd/lotka_volterra/default/model.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/pendulum --model_path data/models/manual_expansion_eigen_dmd/pendulum/default/model.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/duffing --model_path data/models/manual_expansion_eigen_dmd/duffing/default/model.pt
    python -m scripts.eval --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/lorenz --model_path data/models/manual_expansion_eigen_dmd/lorenz/default/model.pt

# SINDy baseline
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/linear/saddle_point --model_path data/models/sindy_baseline/saddle_point/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/linear/degenerate_node --model_path data/models/sindy_baseline/degenerate_node/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/linear/inward_spiral --model_path data/models/sindy_baseline/inward_spiral/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/linear/harmonic_oscillator --model_path data/models/sindy_baseline/harmonic_oscillator/default/model.npz

    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/sindy_baseline/vanderpol/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/lotka_volterra --model_path data/models/sindy_baseline/lotka_volterra/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/pendulum --model_path data/models/sindy_baseline/pendulum/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/duffing --model_path data/models/sindy_baseline/duffing/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/lorenz --model_path data/models/sindy_baseline/lorenz/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/koopman_poly --model_path data/models/sindy_baseline/koopman_poly/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/koopman_poly_large --model_path data/models/sindy_baseline/koopman_poly_large/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/koopman_poly_trig --model_path data/models/sindy_baseline/koopman_poly_trig/default/model.npz

Output:
    data/figures/{model}/{system}/{name}/time_series_idx{traj_index}.png
    data/figures/{model}/{system}/{name}/rollout_idx{traj_index}.png
"""
def get_default_summary_path(model_name: str, system: str, run_name: str) -> str:
    return os.path.join(
        "data", "figures", model_name, system, run_name, "diagnostics", "diagnostics_summary.npz"
    )


def print_validation_summary(summary_path: str) -> None:
    if not os.path.exists(summary_path):
        print(f"\nNo validation summary found at: {summary_path}")
        return

    d = np.load(summary_path, allow_pickle=True)

    print("\n--- Validation diagnostics summary ---")
    if "composite_validation_score" in d:
        print(f"Composite validation score : {float(d['composite_validation_score']):.6e}")
    if "one_step_mse" in d:
        print(f"One-step MSE               : {float(d['one_step_mse']):.6e}")
    if "one_step_rmse" in d:
        print(f"One-step RMSE              : {float(d['one_step_rmse']):.6e}")
    if "one_step_nrmse" in d:
        print(f"One-step NRMSE             : {float(d['one_step_nrmse']):.6e}")
    if "horizon_nrmse" in d:
        print(f"Mean horizon NRMSE         : {float(np.mean(d['horizon_nrmse'])):.6e}")
    if "rollout_nrmse" in d:
        print(f"Mean rollout NRMSE         : {float(np.mean(d['rollout_nrmse'])):.6e}")
    print(f"Summary file               : {summary_path}")

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
    parser.add_argument("--print_validation_summary",action="store_true",help="Print saved validation diagnostics summary for the same run if available.")
    parser.add_argument("--summary_path",type=str,default=None,help="Optional explicit path to diagnostics_summary.npz. If omitted, the default run-matched path is used.")
    parser.add_argument("--horizons",type=str,default="1,2,5,10,20,50,100",help="Comma-separated terminal horizons for test metrics.")
    parser.add_argument("--rollout_horizons",type=str,default="5,10,20,50,100",help="Comma-separated rollout horizons from x(0) for test metrics.")
    parser.add_argument("--max_one_step_pairs_per_traj",type=int,default=None, help="Optional cap on one-step pairs per test trajectory. Ignored when shared rollout cache is used.")
    parser.add_argument("--max_horizon_starts_per_traj",type=int,default=None,help="Optional cap on number of start points per test trajectory for horizon metrics." )
    
    parser.add_argument("--run_diagnostics",action="store_true",help="Run deeper diagnostic plots on the test split.")
    parser.add_argument("--phase_horizons",type=str,default="1,10,50",help="Comma-separated horizons for phase-space error maps.")
    parser.add_argument("--heatmap_horizon",type=int, default=50,help="Horizon for initial-condition error map.")
    parser.add_argument("--heatmap_mode",type=str, default="traj_initials",choices=["traj_initials", "all_valid_starts"],help="Use only test trajectory initials or all valid start points for the error heatmap.")
    parser.add_argument("--linear_error_scale",action="store_true", help="Use linear instead of log scale on the horizon-error plot when diagnostics are enabled.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load test split data
    test_data_path = resolve_split_npz_path(args.data_path, "test")
    data = np.load(test_data_path)
    X = data["X"]
    state_dim = X.shape[-1]
    system = str(data["system"])
    
    # Validate shape
    if X.ndim != 3:
        raise ValueError(f"Expected X to be 3D (T, n_traj, d), got {X.ndim}D")
    if X.shape[1] == 0:
        raise ValueError("No test trajectories found")
    
    # All trajectories in test split file are test trajectories
    test_indices = np.arange(X.shape[1])
    print(f"Loaded {len(test_indices)} test trajectories for system '{system}'")

    # Load model
    model, extras = load_model(
        model_name=args.model,
        model_path=args.model_path,
        data_path=args.data_path,
        state_dim=state_dim,
        system=system,
        device=device,
    )

    # Evaluate on all test trajectories
    mse_mean, mse_std, _ = evaluate_rollouts(
        X=X,
        traj_indices=test_indices,
        model_name=args.model,
        model=model,
        steps=args.steps,
        extras=extras,
    )

    print(
        f"Test rollout MSE over {len(test_indices)} trajectories: "
        f"{mse_mean:.6e} ± {mse_std:.6e}"
    )

    # Plot one test trajectory
    if args.traj_index >= len(test_indices):
        raise IndexError(
            f"traj_index={args.traj_index} but only {len(test_indices)} test trajectories exist"
        )

    traj_id = test_indices[args.traj_index]

    X_true, X_hat = compute_single_rollout(
        X=X,
        traj_id=traj_id,
        steps=args.steps,
        model_name=args.model,
        model=model,
        extras=extras,
    )

    # Setup figure directory
    run_name = infer_run_name(args.model_path, args.name)
    figdir = os.path.join("data", "figures", args.model, system, run_name)
    os.makedirs(figdir, exist_ok=True)
    
    if args.print_validation_summary:
        summary_path = args.summary_path
        if summary_path is None:
            summary_path = get_default_summary_path(args.model, system, run_name)
        print_validation_summary(summary_path)

    # Compute test metrics
    horizons = parse_int_list(args.horizons)
    rollout_horizons = parse_int_list(args.rollout_horizons)

    max_needed = max(max(horizons), max(rollout_horizons))
    if X.shape[0] <= max_needed:
        raise ValueError(
            f"Trajectory length T={X.shape[0]} is too short for max horizon {max_needed}. "
            "Use smaller horizons."
        )

    scales = get_state_scale_from_train_split(args.data_path)
    scale_std = scales["std"]
    
    diag_max_horizon = 1
    if args.run_diagnostics:
        phase_horizons = parse_int_list(args.phase_horizons)
        diag_max_horizon = max(max(phase_horizons), args.heatmap_horizon)

    metric_max_horizon = max(1, max(horizons), max(rollout_horizons), diag_max_horizon)

    rollout_cache = build_rollout_cache(
        X=X,
        traj_indices=test_indices,
        model_name=args.model,
        model=model,
        extras=extras,
        max_horizon=metric_max_horizon,
        start_stride=1,
        max_starts_per_traj=args.max_horizon_starts_per_traj,
    )

    one_step_metrics = compute_one_step_metrics(
        X=X,
        traj_indices=test_indices,
        model_name=args.model,
        model=model,
        extras=extras,
        scale_std=scale_std,
        max_pairs_per_traj=None,
        rollout_cache=rollout_cache,
    )

    horizon_metrics = compute_horizon_metrics(
        X=X,
        traj_indices=test_indices,
        horizons=horizons,
        model_name=args.model,
        model=model,
        extras=extras,
        scale_std=scale_std,
        max_starts_per_traj=args.max_horizon_starts_per_traj,
        rollout_cache=rollout_cache,
    )

    rollout_metrics = compute_full_rollout_metrics(
        X=X,
        traj_indices=test_indices,
        rollout_horizons=rollout_horizons,
        model_name=args.model,
        model=model,
        extras=extras,
        scale_std=scale_std,
        rollout_cache=rollout_cache,
    )

    test_composite_score = compute_composite_validation_score(
        one_step_nrmse=float(one_step_metrics["one_step_nrmse"]),
        horizon_nrmse=horizon_metrics["horizon_nrmse"],
        rollout_nrmse=rollout_metrics["rollout_nrmse"],
    )
    if args.run_diagnostics:
        max_diag_needed = max(max(phase_horizons), args.heatmap_horizon)
        if X.shape[0] <= max_diag_needed:
            raise ValueError(
                f"Trajectory length T={X.shape[0]} is too short for requested diagnostic horizon {max_diag_needed}. "
                "Use smaller --phase_horizons / --heatmap_horizon."
            )

        diagnostics_figdir = os.path.join(figdir, "diagnostics_test")
        os.makedirs(diagnostics_figdir, exist_ok=True)

        run_diagnostics(
            X=X,
            split_idx=test_indices,
            traj_id=traj_id,
            model_name=args.model,
            model=model,
            extras=extras,
            system=system,
            figdir=diagnostics_figdir,
            horizon_metrics=horizon_metrics,
            rollout_metrics=rollout_metrics,
            phase_horizons=phase_horizons,
            heatmap_horizon=args.heatmap_horizon,
            heatmap_mode=args.heatmap_mode,
            linear_error_scale=args.linear_error_scale,
            rollout_cache=rollout_cache,
        )

        print(f"Saved test diagnostics     : {diagnostics_figdir}")
    print("\n--- Test metric summary ---")
    print(f"One-step MSE              : {float(one_step_metrics['one_step_mse']):.6e}")
    print(f"One-step RMSE             : {float(one_step_metrics['one_step_rmse']):.6e}")
    print(f"One-step NRMSE            : {float(one_step_metrics['one_step_nrmse']):.6e}")
    print(f"Mean horizon RMSE         : {float(np.mean(horizon_metrics['horizon_rmse'])):.6e}")
    print(f"Mean horizon NRMSE        : {float(np.mean(horizon_metrics['horizon_nrmse'])):.6e}")
    print(f"Mean rollout RMSE         : {float(np.mean(rollout_metrics['rollout_rmse'])):.6e}")
    print(f"Mean rollout NRMSE        : {float(np.mean(rollout_metrics['rollout_nrmse'])):.6e}")
    print(f"Composite test score      : {test_composite_score:.6e}  (reporting only)")

    test_summary_path = os.path.join(figdir, "test_summary.npz")
    test_summary_payload = {
        "model_name": np.array(args.model),
        "system": np.array(system),
        "run_name": np.array(run_name),
        "split": np.array("test"),
        "test_indices": np.asarray(test_indices),
        "scale_std": scale_std,
        "test_composite_score": np.array(test_composite_score),
        **one_step_metrics,
        **horizon_metrics,
        **rollout_metrics,
    }
    save_summary_npz(test_summary_path, test_summary_payload)
    print(f"Saved test summary        : {test_summary_path}")

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