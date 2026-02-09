import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from src.models.linear_baseline import rollout_linear_map
from src.models.dmd_baseline import rollout_dmd, rollout_edmd
from src.models.ae_linear import AELinearDynamics
from src.models.ae_koopman import AEKoopmanDynamics
from src.eval.rollout import rollout_ae_model

"""
Usage examples:

Global options (defaults):
    --model {linear_baseline,dmd_baseline,edmd_baseline,ae_linear,ae_koopman}
    --data_path data/trajectories/{system}_trajectory.npz
    --model_path data/models/{model}_{system}.pt
    --steps 5000
    --traj_index 0

Linear system (x' = A x):
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear_trajectory.npz --model_path data/models/linear_baseline.npz
    python -m scripts.eval --model dmd_baseline    --data_path data/trajectories/linear_trajectory.npz --model_path data/models/dmd_baseline_linear.npz
    python -m scripts.eval --model edmd_baseline   --data_path data/trajectories/linear_trajectory.npz --model_path data/models/edmd_baseline_linear.npz
    python -m scripts.eval --model ae_linear       --data_path data/trajectories/linear_trajectory.npz --model_path data/models/ae_linear.pt
    Options: --steps --traj_index

Van der Pol:
    python -m scripts.eval --model ae_koopman --data_path data/trajectories/vanderpol_trajectory.npz --model_path data/models/ae_koopman_vanderpol.pt
    Options: --steps --traj_index

Lotka-Volterra:
    python -m scripts.eval --model ae_koopman --data_path data/trajectories/lotka_volterra_trajectory.npz --model_path data/models/ae_koopman_lotka_volterra.pt
    Options: --steps --traj_index

Pendulum:
    python -m scripts.eval --model ae_koopman --data_path data/trajectories/pendulum_trajectory.npz --model_path data/models/ae_koopman_pendulum.pt
    Options: --steps --traj_index

Lorenz:
    python -m scripts.eval --model ae_koopman --data_path data/trajectories/lorenz_trajectory.npz --model_path data/models/ae_koopman_lorenz.pt
    Options: --steps --traj_index
"""

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")

    parser.add_argument("--model", type=str, required=True,
                        choices=["linear_baseline", "dmd_baseline", "edmd_baseline", "ae_linear", "ae_koopman"])

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

    # --------------------------------------------------
    # Load model ONCE
    # --------------------------------------------------

    if args.model == "linear_baseline":
        model_data = np.load(args.model_path)
        M = model_data["M"]
        model = None

    elif args.model == "dmd_baseline":
        model_data = np.load(args.model_path)
        A = model_data["A"]
        model = None

    elif args.model == "edmd_baseline":
        model_data = np.load(args.model_path)
        K = model_data["K"]
        C = model_data["C"]
        degree = int(model_data["degree"])
        model = None

    elif args.model == "ae_linear":
        ckpt = torch.load(args.model_path, map_location=device)
        model = AELinearDynamics(
            state_dim=ckpt["state_dim"],
            latent_dim=ckpt["latent_dim"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

    elif args.model == "ae_koopman":
        ckpt = torch.load(args.model_path, map_location=device)
        model = AEKoopmanDynamics(
            state_dim=ckpt["state_dim"],
            latent_dim=ckpt["latent_dim"],
            hidden_dim=ckpt["hidden_dim"],
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
            X_hat = rollout_dmd(A, x0=x0, steps=steps)

        elif args.model == "edmd_baseline":
            X_hat = rollout_edmd(K, C, degree=degree, x0=x0, steps=steps)

        else:
            x0_torch = torch.tensor(x0, dtype=torch.float32)
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
        X_hat = rollout_dmd(A, x0=x0, steps=steps)

    elif args.model == "edmd_baseline":
        X_hat = rollout_edmd(K, C, degree=degree, x0=x0, steps=steps)
    else:
        x0_torch = torch.tensor(x0, dtype=torch.float32)
        X_hat = rollout_ae_model(
            model,
            x0=x0_torch,
            steps=steps,
            device=device,
        ).cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.plot(X_true[:, 0], X_true[:, 1], label="True")
    plt.plot(X_hat[:, 0], X_hat[:, 1], "--", label=args.model)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Phase space rollout ({args.model})")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    os.makedirs("data/figures", exist_ok=True)
    suffix = f"_{args.name}" if args.name else ""
    plt.savefig(f"data/figures/rollout_{args.model}_{args.traj_index}{suffix}.png")
    plt.close()

if __name__ == "__main__":
    main()