import numpy as np
import torch

from src.models.ml_linear_dynamics import ML_LinearDynamics
from src.models.ml_dmd_free import ML_DMD_FREE
from src.models.ml_dmd_band import ML_DMD_BAND

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
            rbf_n_centers=getattr(args, "rbf_n_centers", 50),
            rbf_center_selection=getattr(args, "rbf_center_selection", "farthest"),
            rbf_bandwidth_mode=getattr(args, "rbf_bandwidth_mode", "knn"),
            rbf_knn_k=getattr(args, "rbf_knn_k", 5),
        ).to(device)

    elif args.model == "ml_dmd_free":
        model = ML_DMD_FREE(
            state_dim=state_dim,
            expansion_degree=args.expansion_degree,
            bias=args.bias == "true",
            sine_cosine_expansion=args.sine_cosine_expansion == "true",
            expansion_type=args.expansion_type,
            system=system_name if args.expansion_type == "specific" else None,
            rbf_n_centers=getattr(args, "rbf_n_centers", 50),
            rbf_center_selection=getattr(args, "rbf_center_selection", "farthest"),
            rbf_bandwidth_mode=getattr(args, "rbf_bandwidth_mode", "knn"),
            rbf_knn_k=getattr(args, "rbf_knn_k", 5),
        ).to(device)

    elif args.model == "ml_dmd_band":
        model = ML_DMD_BAND(
            state_dim=state_dim,
            expansion_degree=args.expansion_degree,
            bias=args.bias == "true",
            sine_cosine_expansion=args.sine_cosine_expansion == "true",
            expansion_type=args.expansion_type,
            system=system_name if args.expansion_type == "specific" else None,
            rbf_n_centers=getattr(args, "rbf_n_centers", 50),
            rbf_center_selection=getattr(args, "rbf_center_selection", "farthest"),
            rbf_bandwidth_mode=getattr(args, "rbf_bandwidth_mode", "knn"),
            rbf_knn_k=getattr(args, "rbf_knn_k", 5),
        ).to(device)

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    return model


def compute_loader_metrics(model, loader, device):
    if loader is None or len(loader.dataset) == 0:
        return None, None

    model.eval()

    total_loss = 0.0
    total_sq_err = 0.0
    total_numel = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            # Support datasets that return either (x, y) or (x, y, future_targets)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    x, y = batch
                elif len(batch) >= 3:
                    x, y = batch[0], batch[1]
                else:
                    raise ValueError(f"Unsupported batch format with length {len(batch)}")
            else:
                raise ValueError("Unsupported batch format from DataLoader")

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_hat = model(x)
            diff = y_hat - y
            loss = torch.mean(diff ** 2)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            total_sq_err += torch.sum(diff ** 2).item()
            total_numel += y.numel()

    mean_loss = total_loss / max(total_samples, 1)
    rmse = float(np.sqrt(total_sq_err / max(total_numel, 1)))

    return mean_loss, rmse

def compute_rollout_metrics(
    model,
    X,
    device,
    eval_horizons=[5, 50, 100],
    gamma=0.99,
    max_trajs=None
):
    if X is None:
        return None

    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=device)
    else:
        X = X.to(device)

    # Subsample trajectories if requested to speed up sweep evaluation
    if max_trajs is not None and max_trajs < X.shape[1]:
        X = X[:, :max_trajs, :]

    T, N, d = X.shape
    max_requested_h = max(eval_horizons)
    max_h = min(max_requested_h, T - 1)
    
    if max_h < 1:
        return None

    model.eval()
    results = {"rollout_failed": 0.0}
    rmse_list = []

    with torch.no_grad():
        rollout_pred = model.rollout(X[0], max_h)

        if not torch.isfinite(rollout_pred).all():
            results["rollout_failed"] = 1.0
            for hh in eval_horizons:
                results[f"rollout_rmse_h{hh}"] = np.nan
                results[f"discounted_mean_rmse_h{hh}_g{gamma:.2f}"] = np.nan
            return results

        for h in range(1, max_h + 1):
            diff = rollout_pred[h] - X[h]
            rmse_list.append(torch.sqrt(torch.mean(diff ** 2)))

    # 4. PROCESS HORIZONS
    rmse_tensor = torch.stack(rmse_list)
    for h in eval_horizons:
        if h <= max_h:
            results[f"rollout_rmse_h{h}"] = float(rmse_tensor[h-1].item())
            weights = torch.tensor([gamma ** i for i in range(1, h + 1)], device=device)
            weighted_rmse = torch.sum(weights * rmse_tensor[:h]) / torch.sum(weights)
            results[f"discounted_mean_rmse_h{h}_g{gamma:.2f}"] = float(weighted_rmse.item())

    return results