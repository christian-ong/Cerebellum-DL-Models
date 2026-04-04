import numpy as np
import torch

from src.models.ml_linear_dynamics import ML_LinearDynamics
from src.models.ml_dmd import ML_DMD


def maybe_set_z_scale(model, train_loader, device):
    if hasattr(model, "expand") and hasattr(model, "set_z_scale"):
        print("Setting z_scale from training data...")
        with torch.no_grad():
            zs = []
            for x_batch, _ in train_loader:
                x_batch = x_batch.to(device)
                z_batch = model.expand(x_batch)
                zs.append(z_batch)

            if len(zs) > 0:
                z_all = torch.cat(zs, dim=0)
                z_scale = torch.mean(torch.abs(z_all), dim=0) + 1e-6
                model.set_z_scale(z_scale)
        print("z_scale set.")


def build_run_name(args, system_name, run_id=None):
    parts = [system_name, args.model]
    parts.append(f"lr{args.lr:.0e}")
    parts.append(f"bs{args.batch_size}")

    if hasattr(args, "expansion_type"):
        parts.append(args.expansion_type)
        parts.append(f"deg{args.expansion_degree}")
        parts.append(f"trig{args.sine_cosine_expansion}")

    if run_id is not None:
        parts.append(run_id[:8])

    return "_".join(parts)


def build_model(args, state_dim, system_name, device):
    if args.model == "ml_linear_dynamics":
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
        raise ValueError(f"Unsupported model: {args.model}")

    return model


def compute_loader_metrics(model, loader, device, state_scale=None):
    if loader is None or len(loader.dataset) == 0:
        return None, None, None

    if state_scale is None:
        raise ValueError("state_scale must be provided for one-step NRMSE computation.")

    if isinstance(state_scale, np.ndarray):
        state_scale = torch.tensor(state_scale, dtype=torch.float32, device=device)
    else:
        state_scale = state_scale.to(device)

    state_scale = state_scale + 1e-8

    model.eval()

    total_loss = 0.0
    total_sq_err = 0.0
    total_sq_err_norm = 0.0
    total_numel = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_hat = model(x)
            diff = y_hat - y
            loss = torch.mean(diff ** 2)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            total_sq_err += torch.sum(diff ** 2).item()
            total_sq_err_norm += torch.sum((diff / state_scale) ** 2).item()
            total_numel += y.numel()

    mean_loss = total_loss / max(total_samples, 1)
    rmse = float(np.sqrt(total_sq_err / max(total_numel, 1)))
    nrmse = float(np.sqrt(total_sq_err_norm / max(total_numel, 1)))

    return mean_loss, rmse, nrmse

def compute_rollout_metrics(
    model,
    X,
    device,
    horizon=500,
    gamma=0.99,
    max_trajs=None,
    state_scale=None,
):
    """
    Computes:
      - horizon-N RMSE
      - horizon-N NRMSE
      - weighted cumulative horizon NRMSE-prediction error

    X expected shape: (T, N, d)
    rollout always starts from X[0] and compares predictions against X[t].

    NRMSE is normalized per state dimension using the empirical standard
    deviation of the target rollout window.
    """
    if X is None:
        return None

    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=device)
    else:
        X = X.to(device)

    T, N, d = X.shape

    if max_trajs is not None and N > max_trajs:
        X = X[:, :max_trajs, :]
        T, N, d = X.shape

    max_h = min(horizon, T - 1)
    if max_h < 1:
        return None

    model.eval()

    if state_scale is None:
        raise ValueError("state_scale must be provided for rollout NRMSE computation.")

    if isinstance(state_scale, np.ndarray):
        state_scale = torch.tensor(state_scale, dtype=torch.float32, device=device)
    else:
        state_scale = state_scale.to(device)

    state_scale = state_scale + 1e-8

    x = X[0]

    # store per-step errors
    rmse_list = []
    nrmse_list = []

    with torch.no_grad():
        for h in range(1, max_h + 1):
            x = model(x)
            diff = x - X[h]

            rmse_h = torch.sqrt(torch.mean(diff ** 2))
            nrmse_h = torch.sqrt(torch.mean((diff / state_scale) ** 2))

            rmse_list.append(rmse_h)
            nrmse_list.append(nrmse_h)

    rmse_tensor = torch.stack(rmse_list)     # (H,)
    nrmse_tensor = torch.stack(nrmse_list)   # (H,)

    results = {}

    # ---- extract specific horizons ----
    eval_points = [10, 100, 500]

    for h in eval_points:
        if h <= max_h:
            results[f"rollout_rmse_h{h}"] = float(rmse_tensor[h-1].item())
            results[f"rollout_nrmse_h{h}"] = float(nrmse_tensor[h-1].item())

            weights = torch.tensor(
                [gamma ** i for i in range(1, h + 1)],
                device=device
            )
            weighted = torch.sum(weights * nrmse_tensor[:h]) / torch.sum(weights)

            results[f"discounted_mean_nrmse_h{h}_g{gamma:.2f}"] = float(weighted.item())

    return results