"""
This script evaluates a trained model by plotting the model's rollout prediction vs the ground truth trajectory.

------------
Run commands

Standard
    python -m experiments.eval_trajectory_rollout --model_name ml_dmd_free --custom_name short --data_path data/trajectories/linear/saddle_point/short

"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data_generation.load_data import resolve_split_npz_path
from src.eval.model_io import load_model

############################################### ARGUMENT PARSING ###############################################

parser = argparse.ArgumentParser(description="Evaluate trained models")

# Model and data selection
parser.add_argument("--model_name", type=str, default="ml_dmd_free", help="Name of the model to evaluate")
parser.add_argument("--custom_name", type=str, default="default", help="Custom given name of the model to evaluate")
parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory or split file")

# Trajectory rollout settings
parser.add_argument("--num_steps", type=int, default=200, help="Number of steps to rollout the model for")
parser.add_argument("--traj_id", type=int, default=0, help="ID of the trajectory to rollout (index in the test set)")

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

    print(f"System: {system}")

    # 2. Construct the model path dynamically based on the extracted system
    if "ml" in args.model_name:
        model_path = f"data/models/{args.model_name}/{system}/{args.custom_name}/model_best.pt"
    elif "hardcoded" in args.model_name:
        model_path = f"data/models/{args.model_name}/{system}/{args.custom_name}/model.pt"
    elif "regression" in args.model_name:
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

    # Start plotting
    fig, axes = plt.subplots(2, 2, figsize=(10,10))

    for i, ax in enumerate(axes.flatten()):
        # Select an initial state (e.g., the first state from the test set)
        x0 = X[0, i, :] # Shape: (state_dim,)
        
        # Roll out the model for a certain number of steps
        trajectory = model.rollout(x0, args.num_steps).detach().cpu().numpy()[:-1, :]
        
        # print("Model Rollout:")

        # locate where nan values start in the trajectory
        nan_indices = np.where(np.isnan(trajectory[:, 0]))[0]
        if len(nan_indices) > 0:
            # print(f"NaN values start at index: {nan_indices[0]}")  
            print(trajectory[nan_indices[0]-10 : nan_indices[0]+10, :])
        else:
            # print("No NaN values found in the trajectory.")
            pass

        # locate where error exceeds a threshold (th * system boundaries)
        x_min_system = np.min(X[:, i, 0])
        x_max_system = np.max(X[:, i, 0])
        x_range_system = x_max_system - x_min_system
        x_error = np.abs(trajectory[:, 0] - X[:args.num_steps, i, 0])

        y_min_system = np.min(X[:, i, 1])
        y_max_system = np.max(X[:, i, 1])
        y_range_system = y_max_system - y_min_system
        y_error = np.abs(trajectory[:, 1] - X[:args.num_steps, i, 1])

        x_error_indices = np.where(x_error > error_th * x_range_system)[0]
        y_error_indices = np.where(y_error > error_th * y_range_system)[0]
        MSE = np.mean(x_error**2 + y_error**2)
        # print(f"Mean Squared Error of the trajectory: {MSE:.2e}")

        error_index = None
        if len(x_error_indices)>0 or len(y_error_indices) > 0:
            if not len(x_error_indices) > 0:
                error_index = y_error_indices[0]
            elif not len(y_error_indices) > 0:
                error_index = x_error_indices[0]
            else:
                error_index = min(x_error_indices[0], y_error_indices[0])
            trajectory = trajectory[:error_index, :]

        # Plot rollout vs ground truth
        
        ax.plot(X[:, i, 0], X[:, i, 1], label='Ground Truth', linestyle='-', alpha=0.7)
        ax.plot(trajectory[:, 0], trajectory[:, 1], label=f'Model Rollout (MSE: {MSE:.2e})', linestyle='--')
        if error_index is not None: # Mark the point where error exceeds threshold
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', label=f'Error Threshold Exceeded (step {error_index})', zorder=5)
        ax.set_title(f"Trajectory {i+1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid()
        ax.legend()

    plt.suptitle(f"Trajectory Rollout vs Ground Truth\n{args.model_name}, {args.custom_name}\n{system}")
    plt.tight_layout()

    # Save the plot dynamically under the correct system
    save_dir = f"experiments/figures/{args.model_name}/{system}/{args.custom_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"traj_rollout.png")
    fig.savefig(save_path)
    print(f"Plot saved to: {save_path}")

    # plt.show()