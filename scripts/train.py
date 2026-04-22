import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_generation.load_data import OneStepTrajectoryDataset, resolve_split_npz_path
from src.eval.sweep_utils import maybe_set_z_scale
from src.models.linear_baseline import fit_linear_map
from src.models.dmd_baseline import fit_dmd
from src.models.ml_linear_dynamics import ML_LinearDynamics
from src.models.regression_dmd import Regression_DMD
from src.models.ml_dmd import ML_DMD
from src.train.train_onestep import train_onestep
from src.models.sindy_baseline import SINDyBaseline

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
    --epochs 50
    --subset 1.0
    --batch_size 64
    --lr 1e-3
    --weight_decay 1e-6
    --seed 0
    --outdir data/models

# Linear baseline
    python -m scripts.train --model linear_baseline --data_path data/trajectories/linear/saddle_point
    python -m scripts.train --model linear_baseline --data_path data/trajectories/linear/degenerate_node
    python -m scripts.train --model linear_baseline --data_path data/trajectories/linear/inward_spiral
    python -m scripts.train --model linear_baseline --data_path data/trajectories/linear/harmonic_oscillator

# DMD baseline
    python -m scripts.train --model dmd_baseline --data_path data/trajectories/linear/saddle_point
    python -m scripts.train --model dmd_baseline --data_path data/trajectories/linear/degenerate_node
    python -m scripts.train --model dmd_baseline --data_path data/trajectories/linear/inward_spiral
    python -m scripts.train --model dmd_baseline --data_path data/trajectories/linear/harmonic_oscillator

---------------------------------------------------------------------------------------------
# ML Linear Dynamics
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/saddle_point --epochs 10
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/degenerate_node --epochs 10
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/inward_spiral --epochs 10
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/harmonic_oscillator --epochs 10

    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/koopman_poly --epochs 10
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/koopman_poly_large --epochs 10
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/koopman_poly_trig --epochs 10

    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/vanderpol --epochs 10
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/lotka_volterra --epochs 10
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/pendulum --epochs 10
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/duffing --epochs 10
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/lorenz --epochs 10


# ML DMD
    python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/saddle_point --epochs 10
    python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/degenerate_node --epochs 10
    python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/inward_spiral --epochs 10
    python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/harmonic_oscillator --epochs 10

    python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/koopman_poly --epochs 10
    python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/koopman_poly_large --epochs 10
    python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/koopman_poly_trig --epochs 10

    python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/vanderpol --epochs 10
    python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/lotka_volterra --epochs 10
    python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/pendulum --epochs 10
    python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/duffing --epochs 10
    python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/lorenz --epochs 10


-----------------------------------------------------------------------------------------------expansion_degree 3 --normalize_state false
# Regression DMD
    python -m scripts.train --model regression_dmd --data_path data/trajectories/linear/saddle_point --bias true --normalize_state true
    python -m scripts.train --model regression_dmd --data_path data/trajectories/linear/degenerate_node --bias true --normalize_state true
    python -m scripts.train --model regression_dmd --data_path data/trajectories/linear/inward_spiral --bias true --normalize_state true
    python -m scripts.train --model regression_dmd --data_path data/trajectories/linear/harmonic_oscillator --bias true --normalize_state true

    python -m scripts.train --model regression_dmd --data_path data/trajectories/nonlinear/vanderpol --expansion_type specific --expansion_degree 10
    python -m scripts.train --model regression_dmd --data_path data/trajectories/nonlinear/lotka_volterra --expansion_type specific --expansion_degree 10
    python -m scripts.train --model regression_dmd --data_path data/trajectories/nonlinear/pendulum --expansion_type specific --expansion_degree 10
    python -m scripts.train --model regression_dmd --data_path data/trajectories/nonlinear/duffing --expansion_type specific --expansion_degree 10
    python -m scripts.train --model regression_dmd --data_path data/trajectories/nonlinear/lorenz --expansion_type specific --expansion_degree 10
    python -m scripts.train --model regression_dmd --data_path data/trajectories/nonlinear/koopman_poly --expansion_type specific --expansion_degree 3
    python -m scripts.train --model regression_dmd --data_path data/trajectories/nonlinear/koopman_poly_large --expansion_type specific --expansion_degree 5
    python -m scripts.train --model regression_dmd --data_path data/trajectories/nonlinear/koopman_poly_trig --expansion_type specific --expansion_degree 10

# ML Linear Dynamics + Manual Expansion
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/saddle_point --epochs 10 --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/degenerate_node --epochs 10 --weight_decay 0.0 --expansion_degree 3 --bias true --sine_cosine_expansion false --lr 1e-5
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/inward_spiral --epochs 10 --weight_decay 0.0 --expansion_degree 3 --bias true --sine_cosine_expansion false --lr 1e-5
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/harmonic_oscillator --epochs 10 --weight_decay 0.0 --expansion_degree 3 --bias true --sine_cosine_expansion false --lr 1e-5

    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/koopman_poly --epochs 10 --expansion_type specific --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/koopman_poly_large --epochs 10 --expansion_type specific --expansion_degree 5 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/koopman_poly_trig --epochs 10 --expansion_type specific --expansion_degree 10 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-5

    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/vanderpol --epochs 10 --expansion_type specific --expansion_degree 10 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/lotka_volterra --epochs 10 --expansion_type specific --expansion_degree 10 --bias false --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/pendulum --epochs 10 --expansion_type specific --expansion_degree 10 --bias false --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/duffing --epochs 10 --expansion_type specific --expansion_degree 10 --bias false --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/lorenz --epochs 3 --expansion_type general --expansion_degree 10 --bias false --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4

# ML DMD + Manual Expansion
    python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/saddle_point --epochs 10 --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5
    python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/degenerate_node --epochs 10 --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5
    python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/inward_spiral --epochs 10 --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5
    python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/harmonic_oscillator --epochs 10 --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5

    python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/koopman_poly --epochs 10 --expansion_type specific --expansion_degree 3 --bias false --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5
    python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/koopman_poly_large --epochs 10 --expansion_type specific --expansion_degree 5 --bias false --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5
    python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/koopman_poly_trig --epochs 10 --expansion_type specific --expansion_degree 10 --bias false --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-5

    python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/vanderpol --epochs 10 --expansion_type specific --expansion_degree 10 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4
    python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/lotka_volterra --epochs 10 --expansion_type specific --expansion_degree 10 --bias false --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4
    python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/pendulum --epochs 10 --expansion_type specific --expansion_degree 10 --bias false --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-4
    python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/duffing --epochs 10 --expansion_type specific --expansion_degree 10 --bias false --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4
    python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/lorenz --epochs 10 --expansion_type specific --expansion_degree 10 --bias false --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4

# SINDy baseline
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/linear/saddle_point --sindy_discrete_time true --sindy_poly_order 1 --sindy_threshold 0.0 --sindy_alpha 0.0
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/linear/degenerate_node --sindy_discrete_time true --sindy_poly_order 1 --sindy_threshold 0.0 --sindy_alpha 0.0
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/linear/inward_spiral --sindy_discrete_time true --sindy_poly_order 1 --sindy_threshold 0.0 --sindy_alpha 0.0
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/linear/harmonic_oscillator --sindy_discrete_time true --sindy_poly_order 1 --sindy_threshold 0.0 --sindy_alpha 0.0

    python -m scripts.train --model sindy_baseline --data_path data/trajectories/nonlinear/vanderpol --sindy_discrete_time true --sindy_poly_order 3 --sindy_threshold 1e-4 --sindy_alpha 1e-6 --sindy_library_type specific --sindy_specific_basis_size 10
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/nonlinear/lotka_volterra --sindy_discrete_time true --sindy_poly_order 3 --sindy_threshold 0.01 --sindy_alpha 1e-6
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/nonlinear/pendulum --sindy_discrete_time true --sindy_poly_order 3 --sindy_threshold 0.01 --sindy_alpha 1e-6
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/nonlinear/duffing --sindy_discrete_time true --sindy_poly_order 3 --sindy_threshold 0.01 --sindy_alpha 1e-6
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/nonlinear/lorenz --sindy_discrete_time true --sindy_poly_order 3 --sindy_threshold 0.01 --sindy_alpha 1e-6
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/nonlinear/koopman_poly --sindy_discrete_time true --sindy_poly_order 2 --sindy_threshold 0.0 --sindy_alpha 0.0
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/nonlinear/koopman_poly_large --sindy_discrete_time true --sindy_poly_order 4 --sindy_threshold 0.0 --sindy_alpha 0.0
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/nonlinear/koopman_poly_trig --sindy_discrete_time true --sindy_poly_order 3 --sindy_threshold 0.01 --sindy_alpha 1e-6

---------------------------------------------------------------------------------------------

Output:
    data/models/{model}/{system}/{run_name}/model.{npz|pt}
"""

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def dataloader_to_numpy(loader):
    """Collect all (x, y) pairs from a DataLoader into NumPy arrays."""
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
            "regression_dmd",
            "ml_lineardynamics",
            "ml_dmd",
            "sindy_baseline",
        ],
    )

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)

    # --------------------------------------------------
    # Training hyperparameters
    # --------------------------------------------------
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of data to use for training (for ML models only)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # --------------------------------------------------
    # DMD / EDMD hyperparameters
    # --------------------------------------------------
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--bias", type=str.lower, choices=["true", "false"], default="false", help="Include bias term in polynomial expansion")
    parser.add_argument("--expansion_type", type=str, default="general", choices=["general", "specific"], help="Whether to use general polynomial expansion (all combinations up to degree) or specific expansion (e.g. only x^2, y^2, xy) for the manual expansion models")
    parser.add_argument("--expansion_degree", type=int, default=1)
    parser.add_argument("--sine_cosine_expansion", type=str.lower,choices=["true", "false"], default="false",help="Include sin(x_i) and cos(x_i) terms in the manual expansion basis")
    parser.add_argument("--normalize_state", type=str.lower, choices=["true", "false"], default="false")
    parser.add_argument("--normalize_lifted", type=str.lower, choices=["true", "false"], default="true")
    parser.add_argument("--regression_rollout_mode",type=str,default="DMD",choices=["linear_dynamics", "DMD","projected_DMD"],help="Default rollout mode for regression_dmd checkpoints.")
    # --------------------------------------------------
    # SINDy
    # --------------------------------------------------
    parser.add_argument("--sindy_discrete_time", type=str.lower, choices=["true", "false"], default="false")
    parser.add_argument("--sindy_poly_order", type=int, default=3)
    parser.add_argument("--sindy_threshold", type=float, default=0.1)
    parser.add_argument("--sindy_alpha", type=float, default=0.0)
    parser.add_argument("--sindy_include_bias", type=str.lower, choices=["true", "false"], default="true")
    parser.add_argument("--sindy_include_interaction", type=str.lower, choices=["true", "false"], default="true")
    parser.add_argument("--sindy_diff_method", type=str, default="finite_difference",choices=["finite_difference", "smoothed_finite_difference"])
    parser.add_argument("--sindy_library_type",type=str,default="polynomial",choices=["polynomial", "fourier", "poly_fourier", "specific"])
    parser.add_argument("--sindy_fourier_n_frequencies", type=int, default=1)
    parser.add_argument("--sindy_specific_basis_size",type=int,default=None,help="If using sindy_library_type='specific', use the first k basis terms for that system.")
    
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

    # Load metadata from train split (all splits have it)
    train_meta_path = resolve_split_npz_path(args.data_path, "train")
    meta = np.load(train_meta_path)
    system_name = str(meta["system"])
    state_dim = meta["X"].shape[-1]
    
    # Setup output directory
    run_name = args.name if args.name else "default"
    save_dir = os.path.join(args.outdir, args.model, system_name, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Load datasets
    train_ds = OneStepTrajectoryDataset(args.data_path, split="train", subset=args.subset)
    val_ds = OneStepTrajectoryDataset(args.data_path, split="val", subset=args.subset)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if len(val_ds) > 0 else None

    train_state_mean = torch.tensor(np.mean(meta["X"], axis=(0, 1)), dtype=torch.float32, device=device)
    train_state_scale = torch.tensor(np.std(meta["X"], axis=(0, 1)), dtype=torch.float32, device=device)
    train_state_scale = torch.clamp(train_state_scale, min=1e-6)
    

# Get training data
    X, Y = dataloader_to_numpy(train_loader)
    
    # ==================================================
    # Linear baseline
    # ==================================================
    if args.model == "linear_baseline":
        print("Fitting linear least-squares baseline...")
        X = train_ds.x.numpy()
        Y = train_ds.y.numpy()
        M = fit_linear_map(X, Y)

        save_path = os.path.join(save_dir, "model.npz")
        np.savez(save_path, M=M, model="linear_baseline", system=system_name, data_path=args.data_path)
        print(f"Saved to {save_path}")
        return
    
    # ==================================================
    # DMD baseline
    # ==================================================
    if args.model == "dmd_baseline":
        print(f"Fitting {args.model.upper()}...")

        X = train_ds.x.numpy()
        Y = train_ds.y.numpy()


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

    elif args.model == "regression_dmd":
        model = Regression_DMD(
            state_dim=state_dim,
            expansion_degree=args.expansion_degree,
            bias=args.bias == "true",
            sine_cosine_expansion=args.sine_cosine_expansion == "true",
            expansion_type=args.expansion_type,
            system=system_name if args.expansion_type == "specific" else None,
            normalize_state=args.normalize_state == "true",
            normalize_lifted=args.normalize_lifted == "true",
            rollout_mode=args.regression_rollout_mode,
            ridge=args.ridge,
            rank=args.rank,
        ).to(device)
        print(f"Expansion type: {args.expansion_type}")
        print(f"Expansion degree: {args.expansion_degree}")
        print(f"Expanded dim: {model.expanded_dim}")
        # print("Expansion library:")
        # for i, name in enumerate(model.expand_names):
        #     print(f"  [{i:02d}] {name}")

        K, C = model.fit(X, Y)
        phi_cond = np.linalg.cond(model.Phi_lift_fitted.detach().cpu().numpy())
        print(f"cond(Phi_lift): {phi_cond:.3e}")
        K_np = model.K_fitted.detach().cpu().numpy()
        Phi_np = model.Phi_lift_fitted.detach().cpu().numpy()
        Lambda_np = model.Lambda_fitted.detach().cpu().numpy()

        Phi_pinv = np.linalg.pinv(Phi_np)
        Lambda_mat = np.diag(Lambda_np)

        recon_err = np.linalg.norm(K_np - Phi_np @ Lambda_mat @ Phi_pinv) / np.linalg.norm(K_np)
        eig_resid = np.linalg.norm(K_np @ Phi_np - Phi_np @ Lambda_mat) / np.linalg.norm(K_np)
        spec_radius = np.max(np.abs(Lambda_np))

        print(f"recon_relerr(K vs PhiΛPhi^+): {recon_err:.3e}")
        print(f"eig_resid_relerr           : {eig_resid:.3e}")
        print(f"spectral_radius           : {spec_radius:.6f}")

        save_path = os.path.join(save_dir, "model.npz")
        
        # Only save what the model actually produces
        save_kwargs = dict(
            train_args=vars(args),

            model="regression_dmd",
            system=system_name,
            state_dim=state_dim,
            expansion_degree=args.expansion_degree,
            bias=args.bias == "true",
            sine_cosine_expansion=args.sine_cosine_expansion == "true",
            expansion_type=args.expansion_type,
            expand_names=model.expand_names,
            system_basis=system_name if args.expansion_type == "specific" else "",
            rollout_mode=args.regression_rollout_mode,
            ridge=args.ridge,
            rank=-1 if args.rank is None else args.rank,
            normalize_state=args.normalize_state == "true",
            normalize_lifted=args.normalize_lifted == "true",

            x_mean=model.x_mean.detach().cpu().numpy(),
            x_scale=model.x_scale.detach().cpu().numpy(),
            psi_scale=model.psi_scale.detach().cpu().numpy(),

            K=model.K_fitted.detach().cpu().numpy(),
            C=model.C_fitted.detach().cpu().numpy(),

            K_tilde=model.K_tilde_fitted.detach().cpu().numpy(),
            U_r=model.U_r_fitted.detach().cpu().numpy(),
            W_reduced=model.W_reduced_fitted.detach().cpu().numpy(),
            Lambda=model.Lambda_fitted.detach().cpu().numpy(),
            Phi_lift=model.Phi_lift_fitted.detach().cpu().numpy(),
            Phi_state=model.Phi_state_fitted.detach().cpu().numpy(),
        )

        if model.Lambda_fitted is not None:
            save_kwargs["Lambda"] = model.Lambda_fitted.detach().cpu().numpy()
            save_kwargs["Phi"] = model.Phi_fitted.detach().cpu().numpy()

        np.savez(save_path, **save_kwargs)
        print(f"Saved regression_dmd checkpoint to: {save_path}")
        return

    # ==================================================
    # SINDy baseline
    # ==================================================
    if args.model == "sindy_baseline":
        print("Fitting SINDy baseline...")
        
        sindy_discrete_time = args.sindy_discrete_time == "true"
        model = SINDyBaseline(
            discrete_time=sindy_discrete_time,
            poly_order=args.sindy_poly_order,
            include_bias=args.sindy_include_bias == "true",
            include_interaction=args.sindy_include_interaction == "true",
            threshold=args.sindy_threshold, alpha=args.sindy_alpha,
            differentiation_method=args.sindy_diff_method,
            library_type=args.sindy_library_type,
            fourier_n_frequencies=args.sindy_fourier_n_frequencies,
            specific_system=system_name if args.sindy_library_type == "specific" else None,
            specific_basis_size=args.sindy_specific_basis_size,
        )
        
        if sindy_discrete_time:
            model.fit_discrete_pairs(X, Y)
        else:
            train_split_path = resolve_split_npz_path(args.data_path, "train")
            meta_data = np.load(train_split_path)
            X_traj = meta_data["X"]
            dt = float(meta_data["dt"])
            if X_traj.ndim != 3:
                raise ValueError("Continuous SINDy expects X with shape (T, n_traj, d)")
            model.fit_continuous_trajectories(X_traj, dt=dt)
        
        save_path = os.path.join(save_dir, "model.npz")
        coeffs = model.get_coefficients()
        equations = np.array(model.equations(), dtype=object)
        np.savez(save_path,
            model="sindy_baseline", system=system_name, data_path=args.data_path,
            discrete_time=sindy_discrete_time, poly_order=args.sindy_poly_order,
            threshold=args.sindy_threshold, alpha=args.sindy_alpha,
            include_bias=args.sindy_include_bias == "true",
            include_interaction=args.sindy_include_interaction == "true",
            diff_method=args.sindy_diff_method, library_type=args.sindy_library_type,
            fourier_n_frequencies=args.sindy_fourier_n_frequencies,
            specific_system=system_name if args.sindy_library_type == "specific" else "",
            specific_basis_size=-1 if args.sindy_specific_basis_size is None else args.sindy_specific_basis_size,
            coefficients=coeffs, equations=equations)
        
        with open(os.path.join(save_dir, "equations.txt"), "w") as f:
            for eq in model.equations():
                f.write(eq + "\n")
        
        print(f"Saved to {save_path}")
        model.print()
        return
    
    # ==================================================
    # Neural network models
    # ==================================================
    print("Training autoencoder model with one-step loss...")
    print(f"Model: {args.model}")
        
    if args.model == "ml_lineardynamics":
        model = ML_LinearDynamics(
            state_dim=state_dim,
            expansion_degree=args.expansion_degree,
            expansion_type=args.expansion_type,
            bias=args.bias == "true",
            sine_cosine_expansion=args.sine_cosine_expansion == "true",
            system=system_name,
        ).to(device)
    
    elif args.model == "ml_dmd":
        model = ML_DMD(
            state_dim=state_dim,
            expansion_degree=args.expansion_degree,
            bias=args.bias == "true",
            sine_cosine_expansion=args.sine_cosine_expansion == "true",
            expansion_type=args.expansion_type,
            system=system_name if args.expansion_type == "specific" else None,
        ).to(device)

    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    if hasattr(model, "expansion_type"):
        print(f"Expansion type: {args.expansion_type}")
    if hasattr(model, "expansion_degree"):
        print(f"Expansion degree: {args.expansion_degree}")
    if hasattr(model, "expansion_type"):
        print(f"Expand names: {model.expand_names}")

    if hasattr(model, "set_state_scale"):
        model.set_state_scale(train_state_mean, train_state_scale)

    # --------------------------------------------------
    # Compute lifted scaling (only for expansion models)
    # --------------------------------------------------

    maybe_set_z_scale(model, train_loader, device)
    
    # Train
    model, (train_losses, batch_val_losses, epoch_val_losses, loss_components_val) = train_onestep(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)
    
    # Save
    save_path = os.path.join(save_dir, "model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model": args.model,
        "system": system_name,
        "state_dim": state_dim,
        "train_args": vars(args),
        "data_path": args.data_path,
        "expand_names": model.expand_names if hasattr(model, "expand_names") else None,
    }, save_path)
    
    loss_path = os.path.join(save_dir, "losses.npz")
    np.savez(loss_path, train_losses=train_losses, batch_val_losses=batch_val_losses,
             epoch_val_losses=epoch_val_losses, loss_components_val=loss_components_val)
    
    print(f"Saved model to {save_path}")
    print(f"Saved losses to {loss_path}")


if __name__ == "__main__":
    main()
