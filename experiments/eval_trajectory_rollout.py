"""
This script evaluates a trained model by plotting the model's rollout prediction vs the ground truth trajectory.

------------
Run commands

Standard
    python -m experiments.eval_trajectory_rollout --model_name ml_dmd --custom_name short --data_path data/trajectories/linear/saddle_point/short

"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data_generation.load_data import resolve_split_npz_path
from src.eval.model_io import load_model, predict_rollout_from_x0
from src.eval.delay_utils import get_model_delay_depth, delay_start_index, make_rollout_initial_condition
from src.eval.diagnostics import format_model_label

############################################### ARGUMENT PARSING ###############################################

parser = argparse.ArgumentParser(description="Evaluate trained models")

# Model and data selection
parser.add_argument("--model_name", type=str, default="ml_dmd", help="Name of the model to evaluate")
parser.add_argument("--custom_name", type=str, default="default", help="Custom given name of the model to evaluate")
parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory or split file")
parser.add_argument("--model_path", type=str, default=None, help="Optional explicit checkpoint path to evaluate")

# Trajectory rollout settings
parser.add_argument("--num_steps", type=int, default=100, help="Number of steps to rollout the model for")
parser.add_argument("--traj_id", type=int, default=0, help="ID of the trajectory to rollout (index in the test set)")
parser.add_argument("--outdir", type=str, default=None, help="Force a custom output directory for the saved rollout plot.")

args = parser.parse_args()

################################################################################################################

if __name__ == "__main__":

    # Parameters
    error_th = 5

    # 1. Load data first so we can dynamically extract the system name!
    test_data_path = resolve_split_npz_path(args.data_path, "test")
    data = np.load(test_data_path)
    X = data["X"]
    state_dim = X.shape[-1]
    system = str(data["system"])
    num_trajs = X.shape[1]

    print(f"System: {system}")

    # 2. Prefer an explicit checkpoint path; otherwise infer the conventional location.
    if args.model_path is not None:
        model_path = args.model_path
    elif "ml" in args.model_name:
        model_path = f"data/models/{args.model_name}/{system}/{args.custom_name}/model_best.pt"
    elif "hardcoded" in args.model_name:
        model_path = f"data/models/{args.model_name}/{system}/{args.custom_name}/model.pt"
    elif "regression" in args.model_name or args.model_name in {"linear_baseline", "dmd_baseline", "sindy_baseline", "mlp_baseline"}:
        model_path = f"data/models/{args.model_name}/{system}/{args.custom_name}/model.npz"
    else:
        raise ValueError(f"Unrecognized model name: {args.model_name}")

    # 3. Load the trained model
    model, extras = load_model(
        model_name=args.model_name,
        model_path=model_path,
        data_path=args.data_path,
        state_dim=state_dim,
        system=system,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    if hasattr(model, "expand_names"):
        print(f"Expanded Basis: {model.expand_names}")

    # 4. Evaluate all trajectories to find Best, Median, and Worst
    print(f"Evaluating all {num_trajs} test trajectories for {args.num_steps} steps...")
    delay_depth = get_model_delay_depth(args.model_name, model)
    t0 = int(delay_start_index(delay_depth))

    rollout_results = []
    
    for i in range(num_trajs):
        x0 = make_rollout_initial_condition(X_traj=X[:, i, :], t0=t0, model_name=args.model_name, model=model)
        
        # 1. REMOVE the [:-1, :] slice so we keep the final target step (e.g., t=100)
        trajectory = predict_rollout_from_x0(
            x0=x0,
            steps=args.num_steps,
            model_name=args.model_name,
            model=model,
            extras=extras,
        )
        
        # 2. Add + 1 to args.num_steps so we grab the full array (t=0 up to t=100)
        n_traj_steps = min(trajectory.shape[0], X.shape[0] - t0, args.num_steps + 1)
        X_trunc = X[t0 : t0 + n_traj_steps, i, :]
        rollout_trunc = trajectory[:n_traj_steps, :]

        # 3. Skip the t=0 initial condition [1:] to perfectly match the official future-prediction metric
        rmse = np.sqrt(np.mean((rollout_trunc[1:] - X_trunc[1:])**2))
        
        rollout_results.append({
            'idx': i,
            'rmse': rmse,
            'trajectory': trajectory,
            'X_trunc': X_trunc
        })

    # Sort results by RMSE
    rollout_results.sort(key=lambda x: x['rmse'])
    avg_rmse = np.mean([res['rmse'] for res in rollout_results])
    
    best_case = rollout_results[0]
    median_case = rollout_results[len(rollout_results) // 2]
    worst_case = rollout_results[-1]
    
    selected_cases = [
        ("Best Case", best_case),
        ("Median Case", median_case),
        ("Worst Case", worst_case)
    ]

    # 5. Plotting (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (case_name, res) in zip(axes, selected_cases):
        traj_idx = res['idx']
        rmse = res['rmse']
        trajectory = res['trajectory']
        X_trunc = res['X_trunc']

        # locate where error exceeds a threshold (th * system boundaries)
        x_min_system = np.min(X_trunc[:, 0])
        x_max_system = np.max(X_trunc[:, 0])
        x_range_system = x_max_system - x_min_system
        x_error = np.abs(trajectory[:X_trunc.shape[0], 0] - X_trunc[:, 0])

        y_min_system = np.min(X_trunc[:, 1])
        y_max_system = np.max(X_trunc[:, 1])
        y_range_system = y_max_system - y_min_system
        y_error = np.abs(trajectory[:X_trunc.shape[0], 1] - X_trunc[:, 1])

        x_error_indices = np.where(x_error > error_th * x_range_system)[0]
        y_error_indices = np.where(y_error > error_th * y_range_system)[0]

        error_index = None
        if len(x_error_indices)>0 or len(y_error_indices) > 0:
            if not len(x_error_indices) > 0:
                error_index = y_error_indices[0]
            elif not len(y_error_indices) > 0:
                error_index = x_error_indices[0]
            else:
                error_index = min(x_error_indices[0], y_error_indices[0])
            trajectory = trajectory[:error_index, :]
            X_trunc = X_trunc[:error_index, :]

        # Plot phase space
        ax.plot(X_trunc[:, 0], X_trunc[:, 1], label='Ground Truth', linestyle='-', alpha=0.7)
        ax.plot(trajectory[:X_trunc.shape[0], 0], trajectory[:X_trunc.shape[0], 1], label='Model Rollout', linestyle='--')
        
        # Mark the initial starting point
        ax.scatter(X_trunc[0, 0], X_trunc[0, 1], color='black', marker='o', s=40, label='Initial Condition', zorder=10)
        
        if error_index is not None:
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', label=f'Error > {error_th}x range (step {error_index})', zorder=5)
            
        ax.set_title(f"{case_name}\nRMSE: {rmse:.2e}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    model_label = format_model_label(args.model_name, model, extras, system=system)
    fig.suptitle(f"Trajectory Rollouts\n{model_label}\nAvg Test RMSE: {avg_rmse:.2e}", fontsize=14, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.93))

    # 6. Save Plot
    save_root = os.environ.get("EVAL_BASE_DIR", "experiments/figures")
    save_dir = os.path.join(save_root, args.model_name, system)
    if args.outdir:
        save_dir = args.outdir
    else:
        expansion_type = getattr(model, "expansion_type", None)
        expansion_folder = str(expansion_type) if expansion_type is not None else "none"
        save_dir = os.path.join(save_dir, expansion_folder)
        if args.model_name == "ml_dmd":
            l1_weight = None
            try:
                l1_weight = getattr(model, "l1_weight", None)
            except Exception:
                l1_weight = None
            if l1_weight is None and isinstance(extras, dict):
                l1_weight = extras.get("train_args", {}).get("l1_weight")
            if l1_weight is not None:
                try:
                    l1_value = float(l1_weight)
                    if l1_value == 0.0:
                        save_dir = os.path.join(save_dir, "0.0")
                    else:
                        save_dir = os.path.join(save_dir, "{:.0e}".format(l1_value))
                except Exception:
                    save_dir = os.path.join(save_dir, str(l1_weight))
        save_dir = os.path.join(save_dir, args.custom_name)
        
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"traj_rollout.png")
    fig.savefig(save_path, dpi=200)
    print(f"Plot saved to: {save_path}")