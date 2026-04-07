
import argparse
import os
import numpy as np
import torch

from src.eval.rollout_eval import compute_single_rollout
from src.eval.model_io import load_model, infer_run_name
from src.eval.plot_rollout import plot_time_series, plot_phase_space
from src.eval.plot_eigenvalues import plot_eigenvalues
from src.eval.plot_training_losses import plot_training_losses
from src.eval.plot_matrices import plot_transition_matrix
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
from src.eval.diagnostics import run_diagnostics
"""
Global options (defaults):
    --model {
        linear_baseline,
        dmd_baseline,
        regression_dmd,
        ml_lineardynamics,
        ml_dmd,
        sindy_baseline}
    --data_path data/trajectories/{linear|nonlinear}/{system}
    --model_path data/models/{model}/{system}/{run_name}/model.{npz|pt}
    --steps 500
    --traj_index 0
    --name optional_suffix
    --horizons 1,5
    --rollout_horizons 5

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

---------------------------------------------------------------------------------------------

# Regression DMD
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/regression_dmd/saddle_point/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/linear/degenerate_node --model_path data/models/regression_dmd/degenerate_node/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/linear/inward_spiral --model_path data/models/regression_dmd/inward_spiral/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/linear/harmonic_oscillator --model_path data/models/regression_dmd/harmonic_oscillator/default/model.npz

    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/regression_dmd/vanderpol/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/lotka_volterra --model_path data/models/regression_dmd/lotka_volterra/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/pendulum --model_path data/models/regression_dmd/pendulum/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/duffing --model_path data/models/regression_dmd/duffing/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/lorenz --model_path data/models/regression_dmd/lorenz/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/koopman_poly --model_path data/models/regression_dmd/koopman_poly/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/koopman_poly_large --model_path data/models/regression_dmd/koopman_poly_large/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/koopman_poly_trig --model_path data/models/regression_dmd/koopman_poly_trig/default/model.npz
    
    # Final test evaluation + also print the saved validation diagnostics summary for the same run
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/regression_dmd/saddle_point/default/model.npz --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10

# ML Linear Dynamics
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/linear/saddle_point --model_path data/models/ml_lineardynamics/saddle_point/default/model.pt
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/linear/degenerate_node --model_path data/models/ml_lineardynamics/degenerate_node/default/model.pt
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/linear/inward_spiral --model_path data/models/ml_lineardynamics/inward_spiral/default/model.pt
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/linear/harmonic_oscillator --model_path data/models/ml_lineardynamics/harmonic_oscillator/default/model.pt

    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/nonlinear/koopman_poly --model_path data/models/ml_lineardynamics/koopman_poly/default/model.pt
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/nonlinear/koopman_poly_large --model_path data/models/ml_lineardynamics/koopman_poly_large/default/model.pt
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/nonlinear/koopman_poly_trig --model_path data/models/ml_lineardynamics/koopman_poly_trig/default/model.pt

    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/ml_lineardynamics/vanderpol/default/model.pt
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/nonlinear/lotka_volterra --model_path data/models/ml_lineardynamics/lotka_volterra/default/model.pt
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/nonlinear/pendulum --model_path data/models/ml_lineardynamics/pendulum/default/model.pt
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/nonlinear/duffing --model_path data/models/ml_lineardynamics/duffing/default/model.pt
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/nonlinear/lorenz --model_path data/models/ml_lineardynamics/lorenz/default/model.pt

# ML DMD
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/ml_dmd/saddle_point/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/degenerate_node --model_path data/models/ml_dmd/degenerate_node/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/inward_spiral --model_path data/models/ml_dmd/inward_spiral/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/harmonic_oscillator --model_path data/models/ml_dmd/harmonic_oscillator/default/model.pt

    python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/koopman_poly --model_path data/models/ml_dmd/koopman_poly/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/koopman_poly_large --model_path data/models/ml_dmd/koopman_poly_large/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/koopman_poly_trig --model_path data/models/ml_dmd/koopman_poly_trig/default/model.pt

    python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/ml_dmd/vanderpol/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/lotka_volterra --model_path data/models/ml_dmd/lotka_volterra/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/pendulum --model_path data/models/ml_dmd/pendulum/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/duffing --model_path data/models/ml_dmd/duffing/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/lorenz --model_path data/models/ml_dmd/lorenz/default/model.pt

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
    
# Final test evaluation + print matching validation summary + save test_summary.npz.
# Add --run_diagnostics to also generate the deeper diagnostic plots on the test split.
# Saddle-point example:
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear/saddle_point --model_path data/models/linear_baseline/saddle_point/default/model.npz --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,5 --heatmap_horizon 5
    python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/saddle_point --model_path data/models/dmd_baseline/saddle_point/default/model.npz --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,5 --heatmap_horizon 5
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/regression_dmd/saddle_point/default/model.npz --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,5 --heatmap_horizon 5
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/linear/saddle_point --model_path data/models/ml_lineardynamics/saddle_point/default/model.pt --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,5 --heatmap_horizon 5
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/ml_dmd/saddle_point/default/model.pt --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,5 --heatmap_horizon 5
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/linear/saddle_point --model_path data/models/sindy_baseline/saddle_point/default/model.npz --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,5 --heatmap_horizon 5
--------------------------------------------------------------------------------------------
# Additional diagnostics examples
# --------------------------------
# --run_diagnostics generates the standard test-split diagnostics:
#    * error-vs-horizon plot
#    * phase-space error map(s)
#    * rollout error summary
#
# --run_true_grid_heatmap adds a dense error heatmap over a regular grid of initial states.
# This is a "true simulator vs trained model" comparison, so it is more global than the
# sampled-start initial-condition error map. When enabled, it is the main state-space heatmap.
#
# Useful flags:
#   --heatmap_horizon H      terminal prediction horizon used in the heatmap
#   --grid_resolution N      grid size per axis (N=100 -> 100x100 grid)
#   --phase_horizons ...     horizons shown in the phase-space error maps
#
# Van der Pol example: standard diagnostics + dense true-grid heatmap
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/regression_dmd/vanderpol/default/model.npz --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,10,50 --heatmap_horizon 1 --run_true_grid_heatmap

# Same as above, but with denser grid (slower, prettier figure)
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/regression_dmd/vanderpol/default/model.npz --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,10,50 --heatmap_horizon 1 --run_true_grid_heatmap --grid_resolution 150

# Saddle-point example with true-grid heatmap
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/regression_dmd/saddle_point/default/model.npz --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,5,10 --heatmap_horizon 1 --run_true_grid_heatmap

# The dense true-grid heatmap works for other evaluated models too, as long as they support
# rollout from an initial condition through the normal eval/model_io pipeline.
# Example with ml_dmd:
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/ml_dmd/vanderpol/default/model.pt --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,10,50 --heatmap_horizon 1 --run_true_grid_heatmap

# Quick debug version (faster, lower-resolution heatmap)
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/regression_dmd/vanderpol/default/model.npz --horizons 1,2 --rollout_horizons 5 --run_diagnostics --phase_horizons 1,5 --heatmap_horizon 1 --run_true_grid_heatmap --grid_resolution 50

Output:
    data/figures/{model}/{system}/{run_name}/time_series_idx{traj_index}.png
    data/figures/{model}/{system}/{run_name}/rollout_idx{traj_index}.png
    data/figures/{model}/{system}/{run_name}/test_summary.npz
    data/figures/{model}/{system}/{run_name}/diagnostics_test/* (if --run_diagnostics)
"""

def parse_int_list(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "linear_baseline",
            "dmd_baseline",
            "regression_dmd",
            "ml_lineardynamics",
            "ml_dmd",
            "sindy_baseline",
        ],
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--traj_index", type=int, default=0)
    parser.add_argument("--print_validation_summary", action="store_true")
    parser.add_argument("--horizons", type=str, default="1,5,20")
    parser.add_argument("--rollout_horizons", type=str, default="5,20")
    parser.add_argument("--metric_cap",type=int,default=64,help="Cap on sampled start points per trajectory for metrics. Use 0 for all.")
    parser.add_argument("--use_cache",action="store_true",help="Reuse rollout cache across metrics.")
    parser.add_argument("--run_diagnostics",action="store_true",help="Generate diagnostics with fixed defaults.")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_data_path = resolve_split_npz_path(args.data_path, "test")
    data = np.load(test_data_path)
    X = data["X"]
    state_dim = X.shape[-1]
    system = str(data["system"])

    if X.ndim == 2:
        X = X[:, None, :]
    elif X.ndim != 3:
        raise ValueError(f"Expected X to be 2D or 3D, got {X.ndim}D")

    if X.shape[1] == 0:
        raise ValueError("No test trajectories found")

    test_indices = np.arange(X.shape[1])
    print(f"Loaded {len(test_indices)} test trajectories for system '{system}'")

    model, extras = load_model(
        model_name=args.model,
        model_path=args.model_path,
        data_path=test_data_path,
        state_dim=state_dim,
        system=system,
        device=device,
    )

    if args.model == "regression_dmd":
        print(f"Regression_DMD rollout mode: {extras['rollout_mode']}")

    if args.traj_index >= len(test_indices):
        raise IndexError(
            f"traj_index={args.traj_index} but only {len(test_indices)} test trajectories exist"
        )

    traj_id = test_indices[args.traj_index]
    run_name = infer_run_name(args.model_path)
    figdir = os.path.join("data", "figures", args.model, system, run_name)
    os.makedirs(figdir, exist_ok=True)

    if args.print_validation_summary:
        summary_path = get_default_summary_path(args.model, system, run_name)
        print_validation_summary(summary_path)

    horizons = parse_int_list(args.horizons)
    rollout_horizons = parse_int_list(args.rollout_horizons)

    max_needed = max(max(horizons), max(rollout_horizons))
    if X.shape[0] <= max_needed:
        raise ValueError(
            f"Trajectory length T={X.shape[0]} is too short for max horizon {max_needed}."
        )

    scales = get_state_scale_from_train_split(args.data_path)
    scale_std = scales["std"]

    metric_cap = None if args.metric_cap == 0 else args.metric_cap

    diag_phase_horizons = [1, 10, 50]
    diag_heatmap_horizon = max(50, max(horizons), max(rollout_horizons)) if args.run_diagnostics else 1

    metric_max_horizon = max(
        1,
        max(horizons),
        max(rollout_horizons),
        diag_heatmap_horizon,
    )

    rollout_cache = None
    if args.use_cache or args.run_diagnostics:
        rollout_cache = build_rollout_cache(
            X=X,
            traj_indices=test_indices,
            model_name=args.model,
            model=model,
            extras=extras,
            max_horizon=metric_max_horizon,
            start_stride=1,
            max_starts_per_traj=metric_cap,
        )

    one_step_metrics = compute_one_step_metrics(
        X=X,
        traj_indices=test_indices,
        model_name=args.model,
        model=model,
        extras=extras,
        scale_std=scale_std,
        max_pairs_per_traj=metric_cap,
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
        max_starts_per_traj=metric_cap,
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
            phase_horizons=diag_phase_horizons,
            heatmap_horizon=diag_heatmap_horizon,
            heatmap_mode="traj_initials",
            linear_error_scale=False,
            rollout_cache=rollout_cache,
            data_path=test_data_path,
            run_true_grid_heatmap=False,
            grid_resolution=100,
        )

        print(f"Saved test diagnostics     : {diagnostics_figdir}")

    print("\n--- Test metric summary ---")
    print(f"One-step MSE              : {float(one_step_metrics['one_step_mse']):.6e}")
    print(f"One-step RMSE             : {float(one_step_metrics['one_step_rmse']):.6e}")
    print(f"One-step NRMSE            : {float(one_step_metrics['one_step_nrmse']):.6e}")

    print(f"Mean horizon RMSE         : {float(np.mean(horizon_metrics['horizon_rmse'])):.6e}")
    print(f"Mean horizon NRMSE        : {float(np.mean(horizon_metrics['horizon_nrmse'])):.6e}")
    for h, rmse, nrmse in zip(
        horizon_metrics["horizons"],
        horizon_metrics["horizon_rmse"],
        horizon_metrics["horizon_nrmse"],
    ):
        print(f"  Horizon h={int(h):>3d}        : RMSE={float(rmse):.6e}, NRMSE={float(nrmse):.6e}")

    print(f"Mean rollout RMSE         : {float(np.mean(rollout_metrics['rollout_rmse'])):.6e}")
    print(f"Mean rollout NRMSE        : {float(np.mean(rollout_metrics['rollout_nrmse'])):.6e}")
    for h, rmse, nrmse in zip(
        rollout_metrics["rollout_horizons"],
        rollout_metrics["rollout_rmse"],
        rollout_metrics["rollout_nrmse"],
    ):
        print(f"  Rollout h={int(h):>3d}        : RMSE={float(rmse):.6e}, NRMSE={float(nrmse):.6e}")

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

    X_true, X_hat = compute_single_rollout(
        X=X,
        traj_id=traj_id,
        steps=args.steps,
        model_name=args.model,
        model=model,
        extras=extras,
    )

    plot_time_series(X_true, X_hat, figdir, args.traj_index)
    plot_phase_space(X_true, X_hat, system, figdir, args.model, args.traj_index)

    eigvals = None
    if args.model == "linear_baseline":
        eigvals = np.linalg.eigvals(extras["M"])
    elif args.model == "dmd_baseline":
        eigvals = extras["Lambda"]
    elif args.model == "regression_dmd":
        if "Lambda" in extras:
            eigvals = extras["Lambda"]
        elif "K" in extras:
            eigvals = np.linalg.eigvals(extras["K"])
    elif model is not None and hasattr(model, "Lambda"):
        lam = model.Lambda
        lam = lam.detach().cpu().numpy() if torch.is_tensor(lam) else np.asarray(lam)
        eigvals = np.linalg.eigvals(lam)

    if eigvals is not None:
        plot_eigenvalues(eigvals, figdir)

    loss_file = args.model_path.replace("model.npz", "losses.npz").replace("model.pt", "losses.npz")
    if os.path.exists(loss_file):
        try:
            plot_training_losses(loss_file, figdir)
        except KeyError:
            print(f"Skipping training loss plot (invalid file format): {loss_file}")

    matrix_to_plot = None
    expand_names = None

    if args.model == "regression_dmd" and "K" in extras:
        matrix_to_plot = extras["K"]
        expand_names = model.expand_names if hasattr(model, "expand_names") else None
    elif args.model_path.endswith(".pt") and "ckpt" in extras and "expand_names" in extras["ckpt"]:
        expand_names = extras["ckpt"]["expand_names"]

    plot_transition_matrix(
        model=None if matrix_to_plot is not None else model,
        matrix=matrix_to_plot,
        model_name=os.path.basename(args.model_path).replace(".pt", "").replace(".npz", ""),
        figdir=figdir,
        expand_names=expand_names,
    )


if __name__ == "__main__":
    main()

