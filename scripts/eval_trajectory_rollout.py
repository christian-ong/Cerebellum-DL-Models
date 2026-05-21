"""
This script evaluates a trained model by plotting the model's rollout prediction vs the ground truth trajectory.

------------
Run commands

Standard
    python -m scripts.eval_trajectory_rollout

Specify parameters
    python -m scripts.eval_trajectory_rollout --model_name ml_dmd --custom_name specific --data_name lorenz --num_steps 100 --traj_id 0

"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data_generation.load_data import resolve_split_npz_path
from src.data_generation.load_data import resolve_split_npz_path
from src.eval.model_io import load_model

############################################### ARGUMENT PARSING ###############################################

parser = argparse.ArgumentParser(description="Evaluate trained models")

# Model and data selection
parser.add_argument("--model_name", type=str, default="ml_dmd_free", help="Name of the model to evaluate")
parser.add_argument("--custom_name", type=str, default="specific", help="Custom given name of the model to evaluate")
parser.add_argument("--data_name", type=str, default="lorenz", help="Name of the dataset to evaluate on")

# Trajectory rollout settings
parser.add_argument("--num_steps", type=int, default=100, help="Number of steps to rollout the model for")
parser.add_argument("--traj_id", type=int, default=0, help="ID of the trajectory to rollout (index in the test set)")

args = parser.parse_args()

################################################################################################################

model_path = f"data/models/{args.model_name}/{args.data_name}/{args.custom_name}/model.pt"
data_path = f"data/trajectories/nonlinear/{args.data_name}/test.npz"

if __name__ == "__main__":
    # Load data
    test_data_path = resolve_split_npz_path(data_path, "test")
    data = np.load(test_data_path)
    X = data["X"]
    state_dim = X.shape[-1]
    system = str(data["system"])

    print(system)

    # Load the trained model
    model, extras = load_model(
        model_name=args.model_name,
        model_path=model_path,
        data_path=data_path,
        state_dim=state_dim,
        system=system,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(model.expand_names)

    # Select an initial state (e.g., the first state from the test set)
    x0 = X[0, args.traj_id, :]  # Shape: (state_dim,)
    
    # Roll out the model for a certain number of steps
    trajectory = model.rollout(x0, args.num_steps).detach().numpy()
    
    # print(X.shape)
    # print(trajectory.shape)
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
    plt.plot(trajectory[:, 0], trajectory[:, 1], label='Model Rollout')
    plt.plot(X[:, args.traj_id, 0], X[:, args.traj_id, 1], label='Ground Truth')
    plt.title(f"Trajectory Rollout vs Ground Truth\n{args.model_name}, {args.custom_name}\n{system}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.show()
