import numpy as np
import torch

# from src.models.deprecated.ml_dmd import ML_DMD
# from src.models.deprecated.ml_eigen_dmd import MLEigenDMD
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
    # if args.model == "ml_dmd":
    #     model = ML_DMD(
    #         state_dim=state_dim,
    #     ).to(device)

    # elif args.model == "ml_eigen_dmd":
    #     model = MLEigenDMD(
    #         state_dim=state_dim,
    #     ).to(device)

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

            loss = _get_loss_tensor(model, x, y, loss_fn)
            y_hat = model(x)

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
    """
    Computes rollout RMSE at multiple fractions of trajectory length.

    Returns dict:
        {
            0.25: rmse_25,
            0.5: rmse_50,
            0.75: rmse_75,
            1.0: rmse_full,
        }
    """

    if len(traj_idx) == 0:
        return None

    T = X.shape[0]
    max_horizon = T - 1

    # Convert fractions → horizons
    horizons = [int(f * max_horizon) for f in fractions]
    horizons = [max(1, h) for h in horizons]

    if max_trajs is not None and len(traj_idx) > max_trajs:
        traj_idx = traj_idx[:max_trajs]

    model.eval()

    # storage for each horizon
    sq_errors = {h: 0.0 for h in horizons}
    numels = {h: 0 for h in horizons}

    with torch.no_grad():
        for traj in traj_idx:

            x_t = torch.tensor(X[0, traj], dtype=torch.float32, device=device).unsqueeze(0)

            preds = []

            # rollout FULL trajectory ONCE
            for _ in range(max_horizon):
                x_t = model(x_t)
                preds.append(x_t.squeeze(0).cpu().numpy())

            pred = np.stack(preds, axis=0)     # (T-1, d)
            true = X[1:, traj]                 # (T-1, d)

            # compute RMSE at each horizon
            for h in horizons:
                diff = pred[:h] - true[:h]
                sq_errors[h] += float(np.sum(diff ** 2))
                numels[h] += diff.size

    rmse_dict = {
        h: float(np.sqrt(sq_errors[h] / max(numels[h], 1)))
        for h in horizons
    }

    # map back to fractions for nicer logging
    return {
        int(100 * h / max_horizon): rmse_dict[h]
        for h in horizons
    }