import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_generation.load_data import OneStepTrajectoryDataset
from src.models.linear_baseline import fit_linear_map
from src.models.dmd_baseline import fit_dmd
from src.models.edmd_baseline import fit_edmd
from src.models.ml_dmd import LinearDynamics
from src.models.manual_expansion_ml_dmd import ManualExpansion_MLDMD
from src.models.ae_linear import AELinearDynamics
from src.models.ae_koopman import AEKoopmanDynamics
from src.models.manual_expansion_manual_dmd import ManualExpansion_ManualDMD
from src.train.train_ae_onestep import train_ae_onestep

"""
Global options (defaults):
    --model {
        linear_baseline,
        dmd_baseline,
        edmd_baseline,
        ae_linear,
        ae_koopman,
        ml_dmd,
        manual_expansion_ml_dmd,
        manual_expansion_manual_dmd,}
    --data_path data/trajectories/{system}_trajectory.npz
    --epochs 50
    --batch_size 512
    --lr 1e-3
    --weight_decay 1e-6
    --latent_dim 2
    --hidden_dim 64
    --seed 0
    --outdir data/models

Linear system (x' = A x): # OUTDATED NAMES (linear_trajectory xd)
    python -m scripts.train --model linear_baseline --data_path data/trajectories/linear_trajectory.npz
    python -m scripts.train --model dmd_baseline    --data_path data/trajectories/linear_trajectory.npz
    python -m scripts.train --model edmd_baseline   --data_path data/trajectories/linear_trajectory.npz
    python -m scripts.train --model ae_linear       --data_path data/trajectories/linear_trajectory.npz
    python -m scripts.train --model ml_dmd          --data_path data/trajectories/linear_trajectory.npz
    python -m scripts.train --model manual_expansion_ml_dmd --data_path data/trajectories/linear_trajectory.npz
    python -m scripts.train --model manual_expansion_manual_dmd --data_path data/trajectories/linear_trajectory.npz
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

Duffing:
    python -m scripts.train --model ae_koopman --data_path data/trajectories/duffing_trajectory.npz
    Options: --latent_dim --hidden_dim (+ training options)

Output:
    data/models/{model}.pt   (AE models)
    data/models/linear_baseline.npz
"""

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def dataloader_to_numpy(loader):
    """
    Collect all (x, y) pairs from a DataLoader into NumPy arrays.

    Returns:
        X : (N, d)
        Y : (N, d)
    """
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.vstack(xs), np.vstack(ys)

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train linear baselines, DMD/EDMD, or AE models"
    )

    # --------------------------------------------------
    # Model selection
    # --------------------------------------------------
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "linear_baseline",
            "dmd_baseline",
            "edmd_baseline",
            "ae_linear",
            "ae_koopman",
            "ml_dmd",
            "manual_expansion_ml_dmd",
            "manual_expansion_manual_dmd",
        ],
    )

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)

    # --------------------------------------------------
    # Training hyperparameters
    # --------------------------------------------------
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)

    # --------------------------------------------------
    # Model hyperparameters
    # --------------------------------------------------
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=64)

    # --------------------------------------------------
    # DMD / EDMD hyperparameters
    # --------------------------------------------------
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--edmd_degree", type=int, default=3)

    # --------------------------------------------------
    # Misc
    # --------------------------------------------------
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="data/models")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    suffix = f"_{args.name}" if args.name else ""

    # --------------------------------------------------
    # Load dataset metadata
    # --------------------------------------------------
    meta = np.load(args.data_path)
    system_name = str(meta["system"])
    state_dim = meta["X"].shape[-1]

    # --------------------------------------------------
    # Build datasets + loaders
    # --------------------------------------------------
    train_ds = OneStepTrajectoryDataset(
        args.data_path,
        split="train",
    )
    val_ds = OneStepTrajectoryDataset(
        args.data_path,
        split="val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size)
        if len(val_ds) > 0
        else None
    )

    # ==================================================
    # Linear least-squares baseline
    # ==================================================
    if args.model == "linear_baseline":
        print("Fitting linear least-squares baseline...")

        X, Y = dataloader_to_numpy(train_loader)
        M = fit_linear_map(X, Y)

        save_path = os.path.join(
            args.outdir,
            f"linear_baseline_{system_name}{suffix}.npz",
        )

        np.savez(
            save_path,
            M=M,
            model="linear_baseline",
            system=system_name,
            data_path=args.data_path,
        )

        print("Saved linear baseline to:", save_path)
        return

    # ==================================================
    # DMD / EDMD baselines
    # ==================================================
    if args.model in {"dmd_baseline", "edmd_baseline", "manual_expansion_manual_dmd"}:
        print(f"Fitting {args.model.upper()}...")

        X, Y = dataloader_to_numpy(train_loader)

        if args.model == "dmd_baseline":
            Lambda, Phi = fit_dmd(
                X,
                Y,
                rank=args.rank,
                ridge=args.ridge,
            )

            save_path = os.path.join(
                args.outdir,
                f"dmd_baseline_{system_name}{suffix}.npz",
            )

            np.savez(
                save_path,
                Lambda=Lambda,
                Phi=Phi,
                rank=args.rank,
                ridge=args.ridge,
                model="dmd_baseline",
                system=system_name,
                data_path=args.data_path,
            )

            print("Saved DMD baseline to:", save_path)
            return

        # EDMD
        elif args.model == "edmd_baseline":
            K, C = fit_edmd(
                X,
                Y,
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
                model="edmd_baseline",
                system=system_name,
                data_path=args.data_path,
            )

            print("Saved EDMD baseline to:", save_path)
            return
    
        # manual expansion manual DMD
        elif args.model == "manual_expansion_manual_dmd":
            model = ManualExpansion_ManualDMD(
                state_dim=state_dim,
                expansion_degree=args.edmd_degree,
                rank=args.rank,
                ridge=args.ridge,
            ).to(device)
            K = model.fit(X, Y)
            print(K)
        

            save_path = os.path.join(
                args.outdir,
                f"manual_expansion_manual_dmd_{system_name}{suffix}.npz",
            )

            np.savez(
                save_path,
                K=K.detach().cpu().numpy(),
                state_dim=state_dim,
                expansion_degree=args.edmd_degree,
                rank=args.rank,
                ridge=args.ridge,
                model="manual_expansion_manual_dmd",
                system=system_name,
                data_path=args.data_path,
            )

            print("Saved manual DMD manual expansion baseline to:", save_path)
            return

        else:
            raise ValueError(f"Unknown manual model: {args.model}")

    # ==================================================
    # Network models
    # ==================================================
    print("Training autoencoder model with one-step loss...")
    print(f"Model: {args.model}")
    if args.model == "ml_dmd":
        model = LinearDynamics(
            state_dim=state_dim,
        ).to(device)

    elif args.model == "manual_expansion_ml_dmd":
        model = ManualExpansion_MLDMD(
            state_dim=state_dim,
            expansion_degree=args.edmd_degree,
        ).to(device)

    elif args.model == "ae_linear":
        model = AELinearDynamics(
            state_dim=state_dim,
            latent_dim=state_dim,  # intentional
        ).to(device)

    elif args.model == "ae_koopman":
        model = AEKoopmanDynamics(
            state_dim=state_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)

    else:
        raise ValueError(f"Unknown model: {args.model}")

    model, (train_losses, batch_val_losses, epoch_val_losses) = train_ae_onestep(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    save_path = os.path.join(
        args.outdir,
        f"{args.model}_{system_name}{suffix}.pt",
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model": args.model,
            "system": system_name,
            "state_dim": state_dim,
            "latent_dim": (
                state_dim if args.model == "ae_linear" else args.latent_dim
            ),
            "hidden_dim": args.hidden_dim if args.model == "ae_koopman" else None,
            "train_args": vars(args),
            "data_path": args.data_path,
            "expand_names": model.expand_names if "expansion" in args.model else None,
        },
        save_path,
    )

    # Save training losses
    loss_path = save_path.replace(".pt", "_losses.npz")
    np.savez(
        loss_path, 
        train_losses=train_losses, 
        batch_val_losses=batch_val_losses, 
        epoch_val_losses=epoch_val_losses)
    print("Saved model and losses to:", save_path, loss_path)


if __name__ == "__main__":
    main()