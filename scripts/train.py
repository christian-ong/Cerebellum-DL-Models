import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_generation.load_data import OneStepTrajectoryDataset
from src.models.linear_baseline import fit_linear_map
from src.models.dmd_baseline import fit_dmd, fit_edmd
from src.models.ae_linear import AELinearDynamics
from src.models.ae_koopman import AEKoopmanDynamics
from src.train.train_ae_onestep import train_ae_onestep

# --------------------------------------------------
# Main
# --------------------------------------------------

"""
Global options (defaults):
    --model {linear_baseline,dmd_baseline,edmd_baseline,ae_linear,ae_koopman}
    --data_path data/trajectories/{system}_trajectory.npz
    --epochs 50
    --batch_size 512
    --lr 1e-3
    --weight_decay 1e-6
    --latent_dim 2
    --hidden_dim 64
    --seed 0
    --outdir data/models

Linear system (x' = A x):
    python -m scripts.train --model linear_baseline --data_path data/trajectories/linear_trajectory.npz
    python -m scripts.train --model dmd_baseline    --data_path data/trajectories/linear_trajectory.npz
    python -m scripts.train --model edmd_baseline   --data_path data/trajectories/linear_trajectory.npz
    python -m scripts.train --model ae_linear       --data_path data/trajectories/linear_trajectory.npz
    Options: (ae_linear uses --epochs --batch_size --lr --weight_decay)

Van der Pol:
    python -m scripts.train --model ae_koopman --data_path data/trajectories/vanderpol_trajectory.npz
    Options: --latent_dim --hidden_dim (+ training options)

Lotka-Volterra:
    python -m scripts.train --model ae_koopman --data_path data/trajectories/lotka_volterra_trajectory.npz
    Options: --latent_dim --hidden_dim (+ training options)

Pendulum:
    python -m scripts.train --model ae_koopman --data_path data/trajectories/pendulum_trajectory.npz
    Options: --latent_dim --hidden_dim (+ training options)

Lorenz:
    python -m scripts.train --model ae_koopman --data_path data/trajectories/lorenz_trajectory.npz
    Options: --latent_dim --hidden_dim (+ training options)

Output:
    data/models/{model}.pt   (AE models)
    data/models/linear_baseline.npz
"""

def main():
    parser = argparse.ArgumentParser(description="Train linear baselines or AE models")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["linear_baseline", "dmd_baseline", "edmd_baseline", "ae_linear", "ae_koopman"],
    )

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--name", type=str, default=None, help="Optional suffix added to the model filename")

    # Training params
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)

    # Model params (used by nonlinear AE)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=64)

    # DMD / EDMD params
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--edmd_degree", type=int, default=2)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=0)

    # Output
    parser.add_argument("--outdir", type=str, default="data/models")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    data = np.load(args.data_path)
    X = data["X"]

    if "system" not in data:
        raise ValueError("Dataset does not contain 'system' field.")

    system_name = str(data["system"])

    if "train_idx" not in data or "val_idx" not in data:
        raise ValueError(
            "Dataset does not contain train_idx / val_idx. "
            "Please regenerate it using simulate_data.py."
        )

    train_traj = data["train_idx"]
    val_traj   = data["val_idx"]

    # --------------------------------------------------
    # Linear least-squares baseline
    # --------------------------------------------------

    if args.model == "linear_baseline":
        print("Fitting linear least-squares baseline...")

        M = fit_linear_map(X)

        suffix = f"_{args.name}" if args.name else ""
        save_path = os.path.join(
            args.outdir,
            f"linear_baseline_{system_name}{suffix}.npz"
        )
        np.savez(
            save_path,
            M=M,
            data_path=args.data_path,
            model="linear_baseline",
        )

        print("Saved linear baseline to:", save_path)
        print("M =\n", M)
        return

    # --------------------------------------------------
    # DMD / EDMD baselines
    # --------------------------------------------------

    if args.model in {"dmd_baseline", "edmd_baseline"}:
        if X.ndim == 3:
            X_train = X[:, train_traj, :]
        else:
            X_train = X

        suffix = f"_{args.name}" if args.name else ""

        if args.model == "dmd_baseline":
            print("Fitting DMD baseline...")
            A = fit_dmd(X_train, rank=args.rank, ridge=args.ridge)
            save_path = os.path.join(
                args.outdir,
                f"dmd_baseline_{system_name}{suffix}.npz",
            )
            np.savez(
                save_path,
                A=A,
                rank=args.rank,
                ridge=args.ridge,
                data_path=args.data_path,
                model="dmd_baseline",
            )
            print("Saved DMD baseline to:", save_path)
            return

        print("Fitting EDMD baseline...")
        K, C = fit_edmd(
            X_train,
            degree=args.edmd_degree,
            rank=args.rank,
            ridge=args.ridge,
        )
        save_path = os.path.join(
            args.outdir,
            f"edmd_baseline_{system_name}{suffix}.npz",
        )
        np.savez(
            save_path,
            K=K,
            C=C,
            degree=args.edmd_degree,
            rank=args.rank,
            ridge=args.ridge,
            data_path=args.data_path,
            model="edmd_baseline",
        )
        print("Saved EDMD baseline to:", save_path)
        return

    # --------------------------------------------------
    # Autoencoder-based models (one-step training)
    # --------------------------------------------------

    print("Training autoencoder model with one-step loss...")

    train_ds = OneStepTrajectoryDataset(
        args.data_path,
        traj_indices=train_traj,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    if len(val_traj) > 0:
        val_ds = OneStepTrajectoryDataset(
            args.data_path,
            traj_indices=val_traj,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
        )
    else:
        val_loader = None

    state_dim = X.shape[-1]

    # -------------------------------
    # Linear AE (control)
    # -------------------------------

    if args.model == "ae_linear":
        latent_dim = state_dim  # intentional

        model = AELinearDynamics(
            state_dim=state_dim,
            latent_dim=latent_dim,
        ).to(device)

    # -------------------------------
    # Nonlinear Koopman AE
    # -------------------------------

    elif args.model == "ae_koopman":
        model = AEKoopmanDynamics(
            state_dim=state_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)

    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # -------------------------------
    # Train
    # -------------------------------

    model = train_ae_onestep(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # -------------------------------
    # Save
    # -------------------------------
    
    suffix = f"_{args.name}" if args.name else ""
    save_path = os.path.join(
        args.outdir,
        f"{args.model}_{system_name}{suffix}.pt"
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": state_dim,
            "latent_dim": (
                state_dim if args.model == "ae_linear" else args.latent_dim
            ),
            "hidden_dim": args.hidden_dim if args.model == "ae_koopman" else None,
            "system": system_name,
            "train_args": vars(args),
            "model": args.model,
        },
        save_path,
    )

    print("Saved model to:", save_path)


if __name__ == "__main__":
    main()
