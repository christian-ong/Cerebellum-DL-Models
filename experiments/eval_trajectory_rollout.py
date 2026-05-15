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
from src.models.ml_dmd_free import ML_DMD_FREE
from src.models.ml_dmd_band import ML_DMD_BAND
from src.models.ml_linear_dynamics import ML_LinearDynamics
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
    else:
        model_path = f"data/models/{args.model_name}/{system}/{args.custom_name}/model.pt"

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

    # Select an initial state (e.g., the first state from the test set)
    x0 = X[0, args.traj_id, :]  # Shape: (state_dim,)
    
    # Roll out the model for a certain number of steps
    trajectory = model.rollout(x0, args.num_steps).detach().cpu().numpy()
    
    print("Model Rollout:")

    # locate where nan values start in the trajectory
    nan_indices = np.where(np.isnan(trajectory[:, 0]))[0]
    if len(nan_indices) > 0:
        print(f"NaN values start at index: {nan_indices[0]}")  
        print(trajectory[nan_indices[0]-10 : nan_indices[0]+10, :])
    else:
        print("No NaN values found in the trajectory.")
    
    # Plot rollout vs ground truth
    plt.figure(figsize=(8, 6))
    plt.plot(X[:, args.traj_id, 0], X[:, args.traj_id, 1], label='Ground Truth', linestyle='-', alpha=0.7)
    plt.plot(trajectory[:, 0], trajectory[:, 1], label='Model Rollout',linestyle='--')
    plt.title(f"Trajectory Rollout vs Ground Truth\n{args.model_name}, {args.custom_name}\n{system}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()

    # Save the plot dynamically under the correct system
    save_dir = f"experiments/figures/{args.model_name}/{system}/{args.custom_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"traj_id_{args.traj_id}.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")

    # plt.show()