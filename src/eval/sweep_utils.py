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

    if "manual_expansion" in args.model:
        parts.append(args.expansion_type)
        parts.append(f"deg{args.expansion_degree}")
        parts.append(f"trig{args.sine_cosine_expansion}")

    if run_id is not None:
        parts.append(run_id[:8])

    return "_".join(parts)


def build_model(args, state_dim, system_name, device):
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
        raise ValueError(f"Unsupported model: {args.model}")

    return model


def _get_loss_tensor(model, x, y, loss_fn):
    if hasattr(model, "compute_loss"):
        loss_out = model.compute_loss(x, y)
        return loss_out if isinstance(loss_out, torch.Tensor) else sum(loss_out)
    y_hat = model(x)
    return loss_fn(y_hat, y)


def compute_loader_loss_and_rmse(model, loader, device):
    if loader is None or len(loader.dataset) == 0:
        return None, None

    model.eval()
    loss_fn = torch.nn.MSELoss()

    total_loss = 0.0
    total_sq_err = 0.0
    total_numel = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_hat = model(x)

            if hasattr(model, "compute_loss"):
                loss_out = model.compute_loss(x, y)
                loss = loss_out if isinstance(loss_out, torch.Tensor) else sum(loss_out)
            else:
                loss = loss_fn(y_hat, y)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            sq_err = torch.sum((y_hat - y) ** 2).item()
            total_sq_err += sq_err
            total_numel += y.numel()

    mean_loss = total_loss / max(total_samples, 1)
    rmse = float(np.sqrt(total_sq_err / max(total_numel, 1)))
    return mean_loss, rmse


def compute_rollout_metrics(
    model,
    X,
    device,
    horizon=100,
    gamma=0.95,
    max_trajs=None,
):
    """
    Computes:
      - horizon-N RMSE
      - horizon-N NRMSE
      - weighted cumulative horizon NRMSE-prediction error

    X expected shape: (T, N, d)
    rollout always starts from X[0] and compares predictions against X[t].

    Returns dict with:
      {
        "rollout_rmse_h100": ...,
        "rollout_nrmse_h100": ...,
        "weighted_cumulative_nrmse_h100_g0.95": ...,
      }
    """
    if X is None:
        return None

    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=device)
    else:
        X = X.to(device)

    if X.ndim != 3:
        raise ValueError(f"Expected X to have shape (T, N, d), got {tuple(X.shape)}")

    T, N, d = X.shape
    if T < 2:
        return None

    if max_trajs is not None and N > max_trajs:
        X = X[:, :max_trajs, :]
        T, N, d = X.shape

    max_h = min(horizon, T - 1)
    if max_h < 1:
        return None

    model.eval()

    # denominator for NRMSE:
    # root mean square magnitude of the ground truth over rollout horizon
    target = X[1 : max_h + 1]  # (H, N, d)
    target_sq_mean = torch.mean(target ** 2)
    target_rms = torch.sqrt(torch.clamp(target_sq_mean, min=1e-12))

    x = X[0]  # (N, d)

    total_sq_err = torch.tensor(0.0, device=device)
    total_numel = 0

    weighted_num = torch.tensor(0.0, device=device)
    weight_sum = 0.0

    with torch.no_grad():
        for h in range(1, max_h + 1):
            x = model(x)
            diff = x - X[h]

            mse_h = torch.mean(diff ** 2)
            rmse_h = torch.sqrt(torch.clamp(mse_h, min=1e-12))
            nrmse_h = rmse_h / target_rms

            w = gamma ** h
            weighted_num += w * nrmse_h
            weight_sum += w

            total_sq_err += torch.sum(diff ** 2)
            total_numel += diff.numel()

    rollout_rmse = torch.sqrt(total_sq_err / max(total_numel, 1))
    rollout_nrmse = rollout_rmse / target_rms
    weighted_cumulative_nrmse = weighted_num / max(weight_sum, 1e-12)

    return {
        f"rollout_rmse_h{max_h}": float(rollout_rmse.item()),
        f"rollout_nrmse_h{max_h}": float(rollout_nrmse.item()),
        f"weighted_cumulative_nrmse_h{max_h}_g{gamma:.2f}": float(weighted_cumulative_nrmse.item()),
    }