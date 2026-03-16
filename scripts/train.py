import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_generation.load_data import OneStepTrajectoryDataset
from src.models.linear_baseline import fit_linear_map
from src.models.dmd_baseline import fit_dmd

from src.models.ml_eigen_dmd import MLEigenDMD
from src.models.ml_dmd import ML_DMD
from src.models.manual_expansion_ml_dmd import ManualExpansion_MLDMD
from src.models.manual_expansion_manual_dmd import ManualExpansion_ManualDMD
from src.models.manual_expansion_eigen_dmd import ManualExpansion_EigenDMD
from src.train.train_onestep import train_onestep

"""
Global options (defaults):
    --model {
        linear_baseline,
        dmd_baseline,
        ml_dmd,
        manual_expansion_ml_dmd,
        manual_expansion_manual_dmd,
        manual_expansion_eigen_dmd,
        ml_eigen_dmd}
    --data_path data/trajectories/{system}_trajectory.npz
    --epochs 50
    --subset 1.0
    --batch_size 64
    --lr 1e-3
    --weight_decay 1e-6
    --latent_dim 2
    --hidden_dim 64
    --seed 0
    --outdir data/models

# Linear baseline
    python -m scripts.train --model linear_baseline --data_path data/trajectories/linear/saddle_point_trajectory.npz --epochs 10
    python -m scripts.train --model linear_baseline --data_path data/trajectories/linear/degenerate_node_trajectory.npz --epochs 10
    python -m scripts.train --model linear_baseline --data_path data/trajectories/linear/inward_spiral_trajectory.npz --epochs 10
    python -m scripts.train --model linear_baseline --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --epochs 10

# DMD baseline
    python -m scripts.train --model dmd_baseline --data_path data/trajectories/linear/saddle_point_trajectory.npz --epochs 10
    python -m scripts.train --model dmd_baseline --data_path data/trajectories/linear/degenerate_node_trajectory.npz --epochs 10
    python -m scripts.train --model dmd_baseline --data_path data/trajectories/linear/inward_spiral_trajectory.npz --epochs 10
    python -m scripts.train --model dmd_baseline --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --epochs 10
 
---------------------------------------------------------------------------------------------

# ML DMD
    python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz --epochs 10
    python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/degenerate_node_trajectory.npz --epochs 10
    python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/inward_spiral_trajectory.npz --epochs 10
    python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --epochs 10

# ML Eigen DMD
    python -m scripts.train --model ml_eigen_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz --epochs 2
    python -m scripts.train --model ml_eigen_dmd --data_path data/trajectories/linear/degenerate_node_trajectory.npz --epochs 2
    python -m scripts.train --model ml_eigen_dmd --data_path data/trajectories/linear/inward_spiral_trajectory.npz --epochs 2
    python -m scripts.train --model ml_eigen_dmd --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --epochs 2

---------------------------------------------------------------------------------------------

# Manual expansion + Manual DMD
    python -m scripts.train --model manual_expansion_manual_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz 
    python -m scripts.train --model manual_expansion_manual_dmd --data_path data/trajectories/linear/degenerate_node_trajectory.npz 
    python -m scripts.train --model manual_expansion_manual_dmd --data_path data/trajectories/linear/inward_spiral_trajectory.npz 
    python -m scripts.train --model manual_expansion_manual_dmd --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz 

    python -m scripts.train --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/vanderpol_trajectory.npz --expansion_type specific --expansion_degree 10
    python -m scripts.train --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/lotka_volterra_trajectory.npz --expansion_type specific --expansion_degree 10
    python -m scripts.train --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/pendulum_trajectory.npz --expansion_type specific --expansion_degree 10
    python -m scripts.train --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/duffing_trajectory.npz --expansion_type specific --expansion_degree 10
    python -m scripts.train --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/lorenz_trajectory.npz --expansion_type specific --expansion_degree 10
    python -m scripts.train --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/koopman_poly_trajectory.npz --expansion_degree 2
    python -m scripts.train --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/koopman_poly_large_trajectory.npz --expansion_degree 4
    python -m scripts.train --model manual_expansion_manual_dmd --data_path data/trajectories/nonlinear/koopman_poly_trig_trajectory.npz --expansion_type specific --expansion_degree 10

# Manual expansion + ML DMD
    python -m scripts.train --model manual_expansion_ml_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz --epochs 10
    python -m scripts.train --model manual_expansion_ml_dmd --data_path data/trajectories/linear/degenerate_node_trajectory.npz --epochs 10
    python -m scripts.train --model manual_expansion_ml_dmd --data_path data/trajectories/linear/inward_spiral_trajectory.npz --epochs 10
    python -m scripts.train --model manual_expansion_ml_dmd --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --epochs 10

# Manual expansion + Eigen DMD
    python -m scripts.train --model manual_expansion_eigen_dmd --data_path data/trajectories/linear/saddle_point_trajectory.npz --epochs 10
    python -m scripts.train --model manual_expansion_eigen_dmd --data_path data/trajectories/linear/degenerate_node_trajectory.npz --epochs 10
    python -m scripts.train --model manual_expansion_eigen_dmd --data_path data/trajectories/linear/inward_spiral_trajectory.npz --epochs 10
    python -m scripts.train --model manual_expansion_eigen_dmd --data_path data/trajectories/linear/harmonic_oscillator_trajectory.npz --epochs 10

    python -m scripts.train --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/vanderpol_trajectory.npz --epochs 10 --expansion_type specific --expansion_degree 7 --name deg7
    python -m scripts.train --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/lotka_volterra_trajectory.npz --epochs 10 --expansion_type specific --expansion_degree 7 --name deg7
    python -m scripts.train --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/pendulum_trajectory.npz --epochs 10 --expansion_type specific --expansion_degree 7 --name deg7
    python -m scripts.train --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/duffing_trajectory.npz --epochs 10 --expansion_type specific --expansion_degree 7 --name deg7
    python -m scripts.train --model manual_expansion_eigen_dmd --data_path data/trajectories/nonlinear/lorenz_trajectory.npz --epochs 10 --expansion_type specific --expansion_degree 7 --name deg7

---------------------------------------------------------------------------------------------

Output:
    data/models/{model}.pt
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
            "ml_dmd",
            "ml_eigen_dmd",
            "manual_expansion_manual_dmd",
            "manual_expansion_ml_dmd",
            "manual_expansion_eigen_dmd",
        ],
    )

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)

    # --------------------------------------------------
    # Training hyperparameters
    # --------------------------------------------------
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of data to use for training (for ML models only)")
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
    parser.add_argument("--bias", type=str.lower, choices=["true", "false"], default="true", help="Include bias term in polynomial expansion")
    parser.add_argument("--manual_decoder", type=str, choices=["regressed", "fixed"], default="fixed")
    parser.add_argument("--manual_regression_method", type=str, default="svd")
    parser.add_argument("--expansion_type", type=str, default="general", choices=["general", "specific"], help="Whether to use general polynomial expansion (all combinations up to degree) or specific expansion (e.g. only x^2, y^2, xy) for the manual expansion models")
    parser.add_argument("--expansion_degree", type=int, default=3)
    parser.add_argument("--sine_cosine_expansion", type=str.lower,choices=["true", "false"], default="false",help="Include sin(x_i) and cos(x_i) terms in the manual expansion basis")

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

    # suffix = f"_{args.name}" if args.name else ""

    # --------------------------------------------------
    # Load dataset metadata
    # --------------------------------------------------
    meta = np.load(args.data_path)
    system_name = str(meta["system"])
    state_dim = meta["X"].shape[-1]
    run_name = args.name if args.name else "default"
    save_dir = os.path.join(args.outdir, args.model, system_name, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------
    # Build datasets + loaders
    # --------------------------------------------------
    train_ds = OneStepTrajectoryDataset(
        args.data_path,
        split="train",
        subset=args.subset,
    )
    val_ds = OneStepTrajectoryDataset(
        args.data_path,
        split="val",
        subset=args.subset,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size) if len(val_ds) > 0 else None
    

    # ==================================================
    # Linear least-squares baseline
    # ==================================================
    if args.model == "linear_baseline":
        print("Fitting linear least-squares baseline...")

        X, Y = dataloader_to_numpy(train_loader)
        M = fit_linear_map(X, Y)

        save_path = os.path.join(save_dir, "model.npz")

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
    if args.model in {"dmd_baseline", "manual_expansion_manual_dmd"}:
        print(f"Fitting {args.model.upper()}...")

        X, Y = dataloader_to_numpy(train_loader)

        if args.model == "dmd_baseline":
            Lambda, Phi = fit_dmd(
                X,
                Y,
                rank=args.rank,
                ridge=args.ridge,
            )

            save_path = os.path.join(save_dir, "model.npz")

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

        # manual expansion manual DMD
        elif args.model == "manual_expansion_manual_dmd":
            model = ManualExpansion_ManualDMD(
                state_dim=state_dim,
                expansion_degree=args.expansion_degree,
                rank=args.rank,
                ridge=args.ridge,
                constant_expansion=args.bias == "true",
                sine_cosine_expansion=args.sine_cosine_expansion == "true",
                expansion_type=args.expansion_type,
                system=system_name if args.expansion_type == "specific" else None,
                decoder_mode=args.manual_decoder,
            ).to(device)
            K, C = model.fit(X, Y, method=args.manual_regression_method)
            
            print("K shape:", K.shape, K)
            print("C shape:", C.shape, C)
            print("Model expand names:", model.expand_names)

            save_path = os.path.join(save_dir, "model.npz")

            np.savez(
                save_path,
                K=K.detach().cpu().numpy(),
                C=C.detach().cpu().numpy(),
                state_dim=state_dim,
                expansion_degree=args.expansion_degree,
                constant_expansion=args.bias == "true",
                sine_cosine_expansion=args.sine_cosine_expansion == "true",
                expansion_type=args.expansion_type,
                system_basis=system_name if args.expansion_type == "specific" else "",
                decoder_mode=args.manual_decoder,
                regression_method=args.manual_regression_method,
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
        model = ML_DMD(
            state_dim=state_dim,
        ).to(device)

    elif args.model == "ml_eigen_dmd":
        model = MLEigenDMD(
            state_dim=state_dim,
        ).to(device)

    elif args.model == "manual_expansion_ml_dmd":
        model = ManualExpansion_MLDMD(
            state_dim=state_dim,
            expansion_degree=args.expansion_degree,
            expansion_type=args.expansion_type,
            system=system_name,
        ).to(device)
    
    elif args.model == "manual_expansion_eigen_dmd":
        model = ManualExpansion_EigenDMD(
            state_dim=state_dim,
            expansion_degree=args.expansion_degree,
            expansion_type=args.expansion_type,
            system=system_name,
        ).to(device)

    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    if hasattr(model, "expansion_type"):
        print(f"Expansion type: {args.expansion_type}")
    if hasattr(model, "expansion_degree"):
        print(f"Expansion degree: {args.expansion_degree}")
    if hasattr(model, "expansion_type"):
        print(f"Expand names: {model.expand_names}")

    model, (train_losses, batch_val_losses, epoch_val_losses, loss_components_val) = train_onestep(
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
    save_path = os.path.join(save_dir, "model.pt")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model": args.model,
            "system": system_name,
            "state_dim": state_dim,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "train_args": vars(args),
            "data_path": args.data_path,
            "expand_names": model.expand_names if hasattr(model, "expand_names") else None,
            "latent_dim": model.latent_dim if hasattr(model, "latent_dim") else None,
        },
        save_path,
    )

    # Save training losses
    loss_path = os.path.join(save_dir, "losses.npz")
    np.savez(
        loss_path, 
        train_losses=train_losses, 
        batch_val_losses=batch_val_losses, 
        epoch_val_losses=epoch_val_losses,
        loss_components_val=loss_components_val)
    print("Saved model and losses to:", save_path, loss_path)


if __name__ == "__main__":
    main()