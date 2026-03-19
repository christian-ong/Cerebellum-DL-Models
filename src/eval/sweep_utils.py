import numpy as np
import torch

from src.models.ml_dmd import ML_DMD
from src.models.ml_eigen_dmd import MLEigenDMD
from src.models.manual_expansion_ml_dmd import ManualExpansion_MLDMD
from src.models.manual_expansion_eigen_dmd import ManualExpansion_EigenDMD


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
            bias=args.bias == "true",
            sine_cosine_expansion=args.sine_cosine_expansion == "true",
            system=system_name,
        ).to(device)

    elif args.model == "manual_expansion_eigen_dmd":
        model = ManualExpansion_EigenDMD(
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


def compute_multi_horizon_rollout_rmse(
    model,
    X,
    traj_idx,
    device,
    fractions=(0.25, 0.5, 0.75, 1.0),
    max_trajs=None,
):
    if len(traj_idx) == 0:
        return None

    traj_idx = np.asarray(traj_idx)
    if max_trajs is not None and len(traj_idx) > max_trajs:
        traj_idx = traj_idx[:max_trajs]

    model.eval()

    X_sel = torch.tensor(X[:, traj_idx, :], dtype=torch.float32, device=device)
    T, N, d = X_sel.shape
    max_horizon = T - 1

    horizons = [max(1, int(f * max_horizon)) for f in fractions]

    x = X_sel[0]  # (N, d)

    sq_errors = {h: torch.tensor(0.0, device=device) for h in horizons}
    numels = {h: 0 for h in horizons}

    with torch.no_grad():
        for t in range(1, max_horizon + 1):
            x = model(x)
            diff = x - X_sel[t]

            diff_sq = torch.sum(diff * diff)
            n_el = diff.numel()

            for h in horizons:
                if t <= h:
                    sq_errors[h] += diff_sq
                    numels[h] += n_el

    # convert ONCE
    return {
        int(100 * h / max_horizon): float(torch.sqrt(sq_errors[h] / max(numels[h], 1)).item())
        for h in horizons
    }