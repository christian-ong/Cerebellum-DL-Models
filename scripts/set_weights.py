"""
python -m scripts.set_weights

This script lets you manually set model weights.
"""
import argparse
import os
import numpy as np
import torch
from torch import device
from torch.utils.data import DataLoader
from scipy.linalg import schur
from src.data_generation.load_data import OneStepTrajectoryDataset
from src.models.ml_dmd import ML_DMD
from src.eval.visualize_modes import get_system_matrices, get_sorted_jordan_form, get_real_representation

"""
python -m scripts.set_weights --system_name closed_trig_large --expansion_degree 3

python -m scripts.set_weights --system_name closed_large --expansion_degree 10 --custom_name jordan --decomp_method jordan
"""

# =========================================
parser = argparse.ArgumentParser(
        description="Train linear baselines, DMD/EDMD, or AE models"
    )
parser.add_argument("--expansion_degree", type=int, default=3, help="Degree of expansion for the model")
parser.add_argument("--system_name", type=str, default="closed_trig_large", help="Name of the system to use")
parser.add_argument("--custom_name", type=str, default="default", help="Custom name for the model (used in saving)")
parser.add_argument("--decomp_method", type=str, default="schur", choices=["numpy","jordan", "schur"], help="Method to use for decomposition (Jordan or Schur)")

parser.add_argument("--jordan_value", type=float, default=1.0, help="Value to use for Jordan block off-diagonal entries (set to 0 to disable)")

args = parser.parse_args()
expansion_degree = args.expansion_degree
system_name = args.system_name
custom_name = args.custom_name
decomp_method = args.decomp_method
jordan_value = args.jordan_value if decomp_method == "jordan" else 0
# =========================================
# Custom weights
A_c, A_d, Lambda, Phi, system_expansion_names = get_system_matrices(
    system=system_name,
    decomp_type=decomp_method,
    truncate_dim=expansion_degree
)

# To real representation
Phi, Lambda = get_real_representation(Phi, Lambda, jordan_value=jordan_value, threshold_jordan=1e-1)
custom_lambda = torch.tensor(Lambda, dtype=torch.float64)
custom_phi = torch.tensor(Phi, dtype=torch.float64)


model_name = "hardcoded_dmd"
expansion_type = "specific"
bias = True if "1" in system_expansion_names else False
sine_cosine_expansion=True if "sin" in system_expansion_names else False

# =========================================
state_dim = 3 if "lorenz" in system_name else 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ML_DMD(
    state_dim=state_dim,
    expansion_degree=expansion_degree,
    bias=bias,
    sine_cosine_expansion=sine_cosine_expansion,
    expansion_type=expansion_type,
    system=system_name,
).to(device)

# Set custom weights here
with torch.no_grad():
    # Update the learned parameters
    # We use .copy_() to update the tensor data in-place
    model.Phi.copy_(custom_phi)
    model.Lambda.copy_(custom_lambda)

# Save the model so you can use it in your evaluation scripts
save_path = f"data/models/{model_name}/{system_name}/{custom_name}/model.pt"

os.makedirs(os.path.dirname(save_path), exist_ok=True)

torch.save({
    "model": model_name,
    "system": system_name,
    "state_dim": state_dim,
    "model_state_dict": model.state_dict(),
    "expand_names": model.expand_names,
    "train_args": {
        "expansion_degree": expansion_degree,
        "expansion_type": expansion_type,
        "bias": bias,
        "sine_cosine_expansion": sine_cosine_expansion
    }
}, save_path)


print(model.expand_names)
print(model.Phi)
print(model.Lambda)

print(f"\nModel with custom weights saved to: {save_path}")