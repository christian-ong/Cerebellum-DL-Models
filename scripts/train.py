import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.data_generation.load_data import OneStepTrajectoryDataset, resolve_split_npz_path
from src.models.linear_baseline import fit_linear_map
from src.models.dmd_baseline import fit_dmd
from src.train.train_onestep import train_onestep
from src.models.ml_linear_dynamics import ML_LinearDynamics
from src.models.regression_dmd import Regression_DMD
from src.models.ml_dmd_band import ML_DMD_BAND
from src.models.ml_dmd_free import ML_DMD_FREE
from src.models.mlp_baseline import MLP_BlackBox
from src.models.ml_dmd_schur import ML_DMD_SCHUR
from src.models.ml_dmd_l1 import ML_DMD_L1
from src.models.sindy_baseline import SINDyBaseline

"""
Global options (defaults):
    --model {
        linear_baseline,
        dmd_baseline,
        regression_dmd,
        ml_lineardynamics,
        ml_dmd_free,
        ml_dmd_band,
        ml_dmd_l1,
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

    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/closed_small --epochs 10
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/closed_large --epochs 10
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/closed_trig --epochs 10

    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/vanderpol --epochs 10
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/lotka_volterra --epochs 10
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/pendulum --epochs 10
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/duffing --epochs 10
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/lorenz --epochs 10


# ML DMD FREE
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/linear/saddle_point --epochs 10
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/linear/degenerate_node --epochs 10
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/linear/inward_spiral --epochs 10
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/linear/harmonic_oscillator --epochs 10

    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/nonlinear/closed_small --epochs 10
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/nonlinear/closed_large --epochs 10
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/nonlinear/closed_trig --epochs 10

    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/nonlinear/vanderpol --epochs 10
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/nonlinear/lotka_volterra --epochs 10
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/nonlinear/pendulum --epochs 10
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/nonlinear/duffing --epochs 10
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/nonlinear/lorenz --epochs 10

# ML DMD BAND
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/linear/saddle_point --epochs 10
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/linear/degenerate_node --epochs 10
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/linear/inward_spiral --epochs 10
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/linear/harmonic_oscillator --epochs 10

    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/closed_small --epochs 10
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/closed_large --epochs 10
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/closed_trig --epochs 10

    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/vanderpol --epochs 10
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/lotka_volterra --epochs 10
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/pendulum --epochs 10
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/duffing --epochs 10
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/lorenz --epochs 10

-----------------------------------------------------------------------------------------------expansion_degree 3 --normalize_state false
# Regression DMD
    python -m scripts.train --model regression_dmd --data_path data/trajectories/linear/saddle_point --bias true --normalize_state true
    python -m scripts.train --model regression_dmd --data_path data/trajectories/linear/degenerate_node --bias true --normalize_state true
    python -m scripts.train --model regression_dmd --data_path data/trajectories/linear/inward_spiral --bias true --normalize_state true
    python -m scripts.train --model regression_dmd --data_path data/trajectories/linear/harmonic_oscillator --bias true --normalize_state true

    python -m scripts.train --model regression_dmd --data_path data/trajectories/nonlinear/closed_small --expansion_type specific --expansion_degree 3
    python -m scripts.train --model regression_dmd --data_path data/trajectories/nonlinear/closed_large --expansion_type specific --expansion_degree 5
    python -m scripts.train --model regression_dmd --data_path data/trajectories/nonlinear/closed_trig --expansion_type specific --expansion_degree 10

    python -m scripts.train --model regression_dmd --data_path data/trajectories/nonlinear/vanderpol --expansion_type specific --expansion_degree 10
    python -m scripts.train --model regression_dmd --data_path data/trajectories/nonlinear/lotka_volterra --expansion_type specific --expansion_degree 10
    python -m scripts.train --model regression_dmd --data_path data/trajectories/nonlinear/pendulum --expansion_type specific --expansion_degree 10
    python -m scripts.train --model regression_dmd --data_path data/trajectories/nonlinear/duffing --expansion_type specific --expansion_degree 10
    python -m scripts.train --model regression_dmd --data_path data/trajectories/nonlinear/lorenz --expansion_type specific --expansion_degree 10

# ML Linear Dynamics + Manual Expansion
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/saddle_point --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/degenerate_node --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/inward_spiral --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/harmonic_oscillator --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4

    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/closed_small --epochs 20 --expansion_type specific --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5 --name spec3
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/closed_large --epochs 20 --expansion_type specific --expansion_degree 5 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5 --name spec5
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/closed_trig_small --epochs 20 --expansion_type specific --expansion_degree 6 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-4 --name spec6_trig
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/closed_trig_medium --epochs 20 --expansion_type specific --expansion_degree 8 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-4 --name spec8_trig
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/closed_trig_large --epochs 20 --expansion_type specific --expansion_degree 10 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-4 --name spec10_trig

    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/vanderpol --epochs 20 --expansion_type general --expansion_degree 5 --bias true --sine_cosine_expansion false --name vanderpol_gen5
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/lotka_volterra --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --name lotkavolterra_gen3
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/pendulum --epochs 20 --expansion_type general --expansion_degree 5 --bias true --sine_cosine_expansion true --name pendulum_gen5_trig
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/duffing --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --name duffing_gen3
    python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/lorenz --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --name lorenz_gen3

# ML DMD + Manual Expansion
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/linear/saddle_point --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/linear/degenerate_node --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/linear/inward_spiral --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/linear/harmonic_oscillator --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4

    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/nonlinear/closed_small --epochs 20 --expansion_type specific --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5 --name spec3
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/nonlinear/closed_large --epochs 20 --expansion_type specific --expansion_degree 5 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5 --name spec5
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/nonlinear/closed_trig_small --epochs 20 --expansion_type specific --expansion_degree 6 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-4 --name spec6_trig
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/nonlinear/closed_trig_medium --epochs 20 --expansion_type specific --expansion_degree 8 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-4 --name spec8_trig
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/nonlinear/closed_trig_large --epochs 20 --expansion_type specific --expansion_degree 10 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-4 --name spec10_trig

    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/nonlinear/vanderpol --epochs 20 --expansion_type general --expansion_degree 5 --bias true --sine_cosine_expansion false --name vanderpol_gen5
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/nonlinear/lotka_volterra --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --name lotkavolterra_gen3
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/nonlinear/pendulum --epochs 20 --expansion_type general --expansion_degree 5 --bias true --sine_cosine_expansion true --name pendulum_gen5_trig
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/nonlinear/duffing --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --name duffing_gen3
    python -m scripts.train --model ml_dmd_free --data_path data/trajectories/nonlinear/lorenz --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --name lorenz_gen3

# ML DMD BAND + Manual Expansion
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/linear/saddle_point --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/linear/degenerate_node --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/linear/inward_spiral --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/linear/harmonic_oscillator --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4

    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/closed_small --epochs 20 --expansion_type specific --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5 --name spec3
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/closed_large --epochs 20 --expansion_type specific --expansion_degree 5 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5 --name spec5
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/closed_trig_small --epochs 20 --expansion_type specific --expansion_degree 6 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-4 --name spec6_trig
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/closed_trig_medium --epochs 20 --expansion_type specific --expansion_degree 8 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-4 --name spec8_trig
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/closed_trig_large --epochs 20 --expansion_type specific --expansion_degree 10 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-4 --name spec10_trig

    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/vanderpol --epochs 20 --expansion_type general --expansion_degree 5 --bias true --sine_cosine_expansion false --name vanderpol_gen5
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/lotka_volterra --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --name lotkavolterra_gen3
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/pendulum --epochs 20 --expansion_type general --expansion_degree 5 --bias true --sine_cosine_expansion true --name pendulum_gen5_trig
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/duffing --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --name duffing_gen3 
    python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/lorenz --epochs 20 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --name lorenz_gen3

    
# SINDy baseline
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/linear/saddle_point --sindy_discrete_time true --sindy_poly_order 1 --sindy_threshold 0.0 --sindy_alpha 0.0
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/linear/degenerate_node --sindy_discrete_time true --sindy_poly_order 1 --sindy_threshold 0.0 --sindy_alpha 0.0
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/linear/inward_spiral --sindy_discrete_time true --sindy_poly_order 1 --sindy_threshold 0.0 --sindy_alpha 0.0
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/linear/harmonic_oscillator --sindy_discrete_time true --sindy_poly_order 1 --sindy_threshold 0.0 --sindy_alpha 0.0

    python -m scripts.train --model sindy_baseline --data_path data/trajectories/nonlinear/closed_small --sindy_discrete_time true --sindy_poly_order 2 --sindy_threshold 0.0 --sindy_alpha 0.0
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/nonlinear/closed_large --sindy_discrete_time true --sindy_poly_order 4 --sindy_threshold 0.0 --sindy_alpha 0.0
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/nonlinear/closed_trig --sindy_discrete_time true --sindy_poly_order 3 --sindy_threshold 0.01 --sindy_alpha 1e-6    

    python -m scripts.train --model sindy_baseline --data_path data/trajectories/nonlinear/vanderpol --sindy_discrete_time true --sindy_poly_order 3 --sindy_threshold 1e-4 --sindy_alpha 1e-6 --sindy_library_type specific --sindy_specific_basis_size 10
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/nonlinear/lotka_volterra --sindy_discrete_time true --sindy_poly_order 3 --sindy_threshold 0.01 --sindy_alpha 1e-6
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/nonlinear/pendulum --sindy_discrete_time true --sindy_poly_order 3 --sindy_threshold 0.01 --sindy_alpha 1e-6
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/nonlinear/duffing --sindy_discrete_time true --sindy_poly_order 3 --sindy_threshold 0.01 --sindy_alpha 1e-6
    python -m scripts.train --model sindy_baseline --data_path data/trajectories/nonlinear/lorenz --sindy_discrete_time true --sindy_poly_order 3 --sindy_threshold 0.01 --sindy_alpha 1e-6

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
    for batch in loader:
        x, y = batch[0], batch[1]
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.vstack(xs), np.vstack(ys)

def prepare_ml_expander_and_lift_stats(
    *,
    model,
    train_ds,
    device,
    max_fit_samples: int = 20000,
):
    """
    Prepare data-dependent expanders and fixed lifted-feature normalization
    for ML Koopman models.

    This supports:
      - raw delay expansion: no fitting, but lifted stats are useful
      - hankel_svd: fit SVD basis on training delay histories
      - rbf: fit centers/sigmas if the expander supports fit()
    """
    expander = getattr(model, "expander", None)
    if expander is None:
        return

    if not hasattr(train_ds, "x"):
        return

    X_fit = train_ds.x

    if not torch.is_tensor(X_fit):
        X_fit = torch.as_tensor(X_fit)

    if max_fit_samples is not None and max_fit_samples > 0 and X_fit.shape[0] > max_fit_samples:
        idx = torch.linspace(
            0,
            X_fit.shape[0] - 1,
            steps=max_fit_samples,
            dtype=torch.long,
        )
        X_fit = X_fit[idx]

    X_fit = X_fit.to(device)

    with torch.no_grad():
        if hasattr(expander, "fit") and not getattr(expander, "is_fitted", False):
            print(
                f"Fitting data-dependent expander on {X_fit.shape[0]} samples "
                f"for expansion_type={getattr(model, 'expansion_type', 'unknown')}..."
            )
            expander.fit(X_fit)

        # Refresh public aliases after fitting.
        if hasattr(expander, "expand_names"):
            model.expand_names = expander.expand_names
        if hasattr(expander, "state_indices"):
            model.state_indices = expander.state_indices
        if hasattr(expander, "expanded_dim"):
            model.expanded_dim = expander.expanded_dim
            model.latent_dim = expander.expanded_dim

        if hasattr(model, "set_lifted_normalization_stats"):
            Z = expander.expand(X_fit)
            mean = Z.mean(dim=0)
            centered = Z - mean
            scale = torch.sqrt(torch.mean(centered * centered, dim=0))
            scale = torch.clamp(scale, min=getattr(model, "lift_norm_eps", 1e-6))

            # Do not destroy the constant feature.
            for j, name in enumerate(getattr(expander, "expand_names", [])):
                if name == "1":
                    mean[j] = 0.0
                    scale[j] = 1.0

            model.set_lifted_normalization_stats(mean, scale)
            print("Set fixed lifted-feature normalization stats.")
            model._lift_stats_initialized = True

def load_best_hyperparams(config_path, system_name, model_name, expansion_type, expansion_degree):
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path(__file__).resolve().parent.parent / config_file

    if not config_file.exists():
        return None

    with config_file.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    try:
        return config[system_name][model_name][expansion_type][str(expansion_degree)]
    except KeyError:
        return None


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
            "ml_dmd_free",
            "ml_dmd_band",
            "ml_dmd_schur",
            "ml_dmd_l1",
            "sindy_baseline",
            "mlp_baseline"
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
    parser.add_argument(
        "--rollout_horizon",
        type=int,
        default=5,
        help="Rollout supervision horizon for loss computation.",
    )
    parser.add_argument(
        "--log_phi_every",
        type=int,
        default=1,
        help="Print get_Phi() every N epochs.",
    )
    parser.add_argument(
        "--phi_print_max_dim",
        type=int,
        default=12,
        help="When Phi is larger than this, print only the top-left block with summary stats.",
    )

    # --------------------------------------------------
    # DMD / EDMD hyperparameters
    # --------------------------------------------------
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--bias", type=str.lower, choices=["true", "false"], default="true", help="Include bias term in polynomial expansion")
    parser.add_argument("--expansion_type", type=str, default="general", choices=["general", "specific", "rbf", "delay","hankel_svd"], help="Whether to use general polynomial expansion, specific expansion, radial basis functions, or a time-delay embedding.")
    parser.add_argument("--expansion_degree", type=int, default=1)
    parser.add_argument("--sine_cosine_expansion", type=str.lower,choices=["true", "false"], default="false",help="Include sin(x_i) and cos(x_i) terms in the manual expansion basis")
    parser.add_argument("--normalize_state", type=str.lower, choices=["true", "false"], default="false")
    parser.add_argument("--normalize_lifted", type=str.lower, choices=["true", "false"], default="true")
    parser.add_argument("--l1_weight", type=float, default=1e-6, help="L1 regularization weight for regression DMD")
    parser.add_argument("--regression_rollout_mode",type=str,default="DMD",choices=["linear_dynamics", "DMD","projected_DMD"],help="Default rollout mode for regression_dmd checkpoints.")
    
    parser.add_argument("--delay_depth", type=int, default=1, help="Number of stacked delay coordinates to use when expansion_type='delay'.")
    parser.add_argument("--hankel_rank",type=int,default=None,help="Number of SVD delay coordinates when expansion_type='hankel_svd'.")
    parser.add_argument("--expander_fit_samples",type=int,default=0,help="Max training samples used to fit data-dependent ML expanders/statistics. Use 0 for all available training samples. Smaller values are useful for quick tests.")
    
    parser.add_argument("--rbf_n_centers", type=int, default=50, help="Number of RBF centers when expansion_type='rbf'.")
    parser.add_argument("--rbf_center_selection", type=str, default="farthest", choices=["random", "farthest"], help="How to choose RBF centers from training states.")
    parser.add_argument("--rbf_bandwidth_mode", type=str, default="knn", choices=["global", "knn"], help="How to choose RBF widths (sigmas).")
    parser.add_argument("--rbf_knn_k", type=int, default=5, help="k for k-nearest-center bandwidth when expansion_type='rbf'.")
    parser.add_argument("--load_rbf_from", type=str, default=None, help="Path to a model.npz to load fixed RBF centers from.")
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
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/best_hyperparams.json",
        help="JSON file with best lr/weight_decay values keyed by system/model/expansion config",
    )
    parser.add_argument(
        "--use_best_hparams",
        action="store_true",
        help="If set, override lr/weight_decay from configs/best_hyperparams.json when available",
    )

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

    if args.use_best_hparams:
        best_hparams = load_best_hyperparams(
            args.config_path,
            system_name=system_name,
            model_name=args.model,
            expansion_type=args.expansion_type,
            expansion_degree=args.expansion_degree,
        )
        if best_hparams is not None:
            args.lr = float(best_hparams["lr"])
            args.weight_decay = float(best_hparams["weight_decay"])
            print(
                "Loaded lr/weight_decay from config for "
                f"{system_name}/{args.model}/{args.expansion_type}/{args.expansion_degree}: "
                f"lr={args.lr}, weight_decay={args.weight_decay}"
            )
    
    # Setup output directory
    run_name = args.name if args.name else "default"
    save_dir = os.path.join(args.outdir, args.model, system_name, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Load datasets
    # ML_DMD_BAND gets a short future window so training can optimize both one-step
    # prediction and short-horizon rollout consistency.
    is_ml_model = args.model in {"ml_lineardynamics", "ml_dmd_free", "ml_dmd_band", "ml_dmd_schur", "ml_dmd_l1"}

    if args.rollout_horizon >= 0:
        rollout_horizon = args.rollout_horizon
    else:
        rollout_horizon = 20 if is_ml_model else 0

    if args.log_phi_every >= 0:
        log_phi_every = args.log_phi_every
    else:
        log_phi_every = 1 if is_ml_model else 0
        
    if args.delay_depth > 1 and args.expansion_type not in {"delay", "hankel_svd"}:
        raise ValueError(
            "delay_depth > 1 requires --expansion_type delay or --expansion_type hankel_svd."
        )

    train_ds = OneStepTrajectoryDataset(
        args.data_path,
        split="train",
        subset=args.subset,
        rollout_horizon=rollout_horizon,
        delay_depth=args.delay_depth,
    )
    val_ds = OneStepTrajectoryDataset(
        args.data_path,
        split="val",
        subset=args.subset,
        rollout_horizon=rollout_horizon,
        delay_depth=args.delay_depth,
    )
    
    pin_memory = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=pin_memory) if len(val_ds) > 0 else None
   
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
            delay_depth=args.delay_depth,
            hankel_rank=args.hankel_rank,
            normalize_state=args.normalize_state == "true",
            normalize_lifted=args.normalize_lifted == "true",
            rollout_mode=args.regression_rollout_mode,
            ridge=args.ridge,
            rank=args.rank,
            rbf_n_centers=args.rbf_n_centers,
            rbf_center_selection=args.rbf_center_selection,
            rbf_bandwidth_mode=args.rbf_bandwidth_mode,
            rbf_knn_k=args.rbf_knn_k,
        ).to(device)

        if args.load_rbf_from:
            print(f"Loading fixed RBF centers and sigmas from: {args.load_rbf_from}")
            loaded_data = np.load(args.load_rbf_from)
            c_tensor = torch.as_tensor(loaded_data["rbf_centers"], dtype=torch.float32, device=device)
            s_tensor = torch.as_tensor(loaded_data["rbf_sigmas"], dtype=torch.float32, device=device)
            
            model.expander.centers.copy_(c_tensor)
            model.expander.sigmas.copy_(s_tensor)
            model.expander.is_fitted = True
            model.expander.freeze_centers = True
            
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
            delay_depth=args.delay_depth,
            hankel_rank=-1 if args.hankel_rank is None else args.hankel_rank,
            rbf_n_centers=args.rbf_n_centers,
            rbf_center_selection=args.rbf_center_selection,
            rbf_bandwidth_mode=args.rbf_bandwidth_mode,
            rbf_knn_k=args.rbf_knn_k,

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

        if args.expansion_type == "rbf":
            save_kwargs["rbf_centers"] = model.expander.centers.detach().cpu().numpy()
            save_kwargs["rbf_sigmas"] = model.expander.sigmas.detach().cpu().numpy()

        if args.expansion_type == "hankel_svd":
            save_kwargs["hankel_mean"] = model.expander.mean.detach().cpu().numpy()
            save_kwargs["hankel_components"] = model.expander.components.detach().cpu().numpy()
            save_kwargs["hankel_singular_values"] = model.expander.singular_values.detach().cpu().numpy()
            
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
            system=system_name if args.expansion_type == "specific" else None,
            delay_depth=args.delay_depth, # <--- ADD THIS LINE
            rbf_n_centers=args.rbf_n_centers,
            rbf_center_selection=args.rbf_center_selection,
            rbf_bandwidth_mode=args.rbf_bandwidth_mode,
            rbf_knn_k=args.rbf_knn_k,
            hankel_rank=args.hankel_rank,
        ).to(device)
    
    elif args.model == "ml_dmd_free":
        model = ML_DMD_FREE(
            state_dim=state_dim,
            expansion_degree=args.expansion_degree,
            bias=args.bias == "true",
            sine_cosine_expansion=args.sine_cosine_expansion == "true",
            expansion_type=args.expansion_type,
            system=system_name if args.expansion_type == "specific" else None,
            delay_depth=args.delay_depth, # <--- ADD THIS LINE
            rbf_n_centers=args.rbf_n_centers,
            rbf_center_selection=args.rbf_center_selection,
            rbf_bandwidth_mode=args.rbf_bandwidth_mode,
            rbf_knn_k=args.rbf_knn_k,
            hankel_rank=args.hankel_rank,
        ).to(device)

    elif args.model == "ml_dmd_band":
        model = ML_DMD_BAND(
            state_dim=state_dim,
            expansion_degree=args.expansion_degree,
            bias=args.bias == "true",
            sine_cosine_expansion=args.sine_cosine_expansion == "true",
            expansion_type=args.expansion_type,
            system=system_name if args.expansion_type == "specific" else None,
            rbf_n_centers=args.rbf_n_centers,
            rbf_center_selection=args.rbf_center_selection,
            rbf_bandwidth_mode=args.rbf_bandwidth_mode,
            rbf_knn_k=args.rbf_knn_k,
            hankel_rank=args.hankel_rank,
        ).to(device)

    elif args.model == "ml_dmd_schur":
        model = ML_DMD_SCHUR(
            state_dim=state_dim,
            expansion_degree=args.expansion_degree,
            bias=args.bias == "true",
            sine_cosine_expansion=args.sine_cosine_expansion == "true",
            expansion_type=args.expansion_type,
            system=system_name if args.expansion_type == "specific" else None,
            rbf_n_centers=args.rbf_n_centers,
            rbf_center_selection=args.rbf_center_selection,
            rbf_bandwidth_mode=args.rbf_bandwidth_mode,
            rbf_knn_k=args.rbf_knn_k,
        ).to(device)
    
    elif args.model == "ml_dmd_l1":
        model = ML_DMD_L1(
            state_dim=state_dim,
            expansion_degree=args.expansion_degree,
            bias=args.bias == "true",
            sine_cosine_expansion=args.sine_cosine_expansion == "true",
            expansion_type=args.expansion_type,
            system=system_name if args.expansion_type == "specific" else None,
            rbf_n_centers=args.rbf_n_centers,
            rbf_center_selection=args.rbf_center_selection,
            rbf_bandwidth_mode=args.rbf_bandwidth_mode,
            rbf_knn_k=args.rbf_knn_k,
            l1_weight=args.l1_weight,
        ).to(device)

    elif args.model == "mlp_baseline":
        model = MLP_BlackBox(
            state_dim=state_dim,
            hidden_dim=64,
            num_layers=4
        ).to(device)

    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    if hasattr(model, "expansion_type"):
        print(f"Expansion type: {args.expansion_type}")
    if hasattr(model, "expansion_degree"):
        print(f"Expansion degree: {args.expansion_degree}")
    if hasattr(model, "expansion_type"):
        print(f"Expand names: {model.expand_names}")
    
    if args.model in {"ml_lineardynamics", "ml_dmd_free", "ml_dmd_band"}:
        prepare_ml_expander_and_lift_stats(
            model=model,
            train_ds=train_ds,
            device=device,
            max_fit_samples=args.expander_fit_samples,
        )
    # Train
    model, (train_losses, epoch_val_losses, loss_components_val), best_checkpoint = train_onestep(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_phi_every=log_phi_every,
        phi_print_max_dim=args.phi_print_max_dim,
    )

    # Save both best-by-validation and final epoch checkpoints.
    checkpoint_base = {
        "model": args.model,
        "system": system_name,
        "state_dim": state_dim,
        "train_args": vars(args),
        "data_path": args.data_path,
        "expand_names": model.expand_names if hasattr(model, "expand_names") else None,
        "best_epoch": best_checkpoint["epoch"],
        "best_val_loss": best_checkpoint["val_loss"],
    }

    best_save_path = os.path.join(save_dir, "model_best.pt")
    torch.save(
        {
            **checkpoint_base,
            "model_state_dict": best_checkpoint["state_dict"],
            "checkpoint_type": "best",
        },
        best_save_path,
    )

    last_save_path = os.path.join(save_dir, "model_last.pt")
    torch.save(
        {
            **checkpoint_base,
            "model_state_dict": model.state_dict(),
            "checkpoint_type": "last",
        },
        last_save_path,
    )

    # Keep backward-compatible default path, now pointing to best checkpoint.
    save_path = os.path.join(save_dir, "model.pt")
    torch.save(
        {
            **checkpoint_base,
            "model_state_dict": best_checkpoint["state_dict"],
            "checkpoint_type": "best",
        },
        save_path,
    )
    
    loss_path = os.path.join(save_dir, "losses.npz")
    np.savez(loss_path, train_losses=train_losses, epoch_val_losses=epoch_val_losses, loss_components_val=loss_components_val)
    
    print(f"Saved best model to {best_save_path}")
    print(f"Saved last model to {last_save_path}")
    print(f"Saved default model to {save_path}")
    print(f"Saved losses to {loss_path}")


if __name__ == "__main__":
    main()
