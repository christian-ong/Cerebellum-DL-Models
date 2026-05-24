import numpy as np
import torch

from src.models.ml_linear_dynamics import ML_LinearDynamics
from src.models.ml_dmd import ML_DMD
from src.models.regression_dmd import Regression_DMD
from src.models.mlp_baseline import MLP_BlackBox

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
    if args.model in {"ml_linear_dynamics", "ml_lineardynamics"}:
        model = ML_LinearDynamics(
            state_dim=state_dim,
            expansion_degree=args.expansion_degree,
            expansion_type=args.expansion_type,
            bias=args.bias == "true",
            sine_cosine_expansion=args.sine_cosine_expansion == "true",
            system=system_name if args.expansion_type == "specific" else None,
            delay_depth=getattr(args, "delay_depth", 1),
            hankel_rank=getattr(args, "hankel_rank", None),
            rbf_n_centers=getattr(args, "rbf_n_centers", 50),
            rbf_center_selection=getattr(args, "rbf_center_selection", "farthest"),
            rbf_bandwidth_mode=getattr(args, "rbf_bandwidth_mode", "knn"),
            rbf_knn_k=getattr(args, "rbf_knn_k", 5),
        ).to(device)

    elif args.model in {"ml_dmd"}:
        model = ML_DMD(
            state_dim=state_dim,
            expansion_degree=args.expansion_degree,
            bias=args.bias == "true",
            sine_cosine_expansion=args.sine_cosine_expansion == "true",
            expansion_type=args.expansion_type,
            system=system_name if args.expansion_type == "specific" else None,
            delay_depth=getattr(args, "delay_depth", 1),
            hankel_rank=getattr(args, "hankel_rank", None),
            rbf_n_centers=getattr(args, "rbf_n_centers", 50),
            rbf_center_selection=getattr(args, "rbf_center_selection", "farthest"),
            rbf_bandwidth_mode=getattr(args, "rbf_bandwidth_mode", "knn"),
            rbf_knn_k=getattr(args, "rbf_knn_k", 5),
            l1_weight=getattr(args, "l1_weight", 1e-6),
        ).to(device)

    elif args.model == "regression_dmd":
        model = Regression_DMD(
            state_dim=state_dim,
            expansion_degree=args.expansion_degree,
            bias=args.bias == "true",
            sine_cosine_expansion=args.sine_cosine_expansion == "true",
            expansion_type=args.expansion_type,
            system=system_name if args.expansion_type == "specific" else None,
            delay_depth=getattr(args, "delay_depth", 1),
            hankel_rank=getattr(args, "hankel_rank", None),
            normalize_state=getattr(args, "normalize_state", "false") == "true",
            normalize_lifted=getattr(args, "normalize_lifted", "true") == "true",
            rollout_mode=getattr(args, "regression_rollout_mode", "DMD"),
            ridge=getattr(args, "ridge", 0.0),
            rank=getattr(args, "rank", None),
            rbf_n_centers=getattr(args, "rbf_n_centers", 50),
            rbf_center_selection=getattr(args, "rbf_center_selection", "farthest"),
            rbf_bandwidth_mode=getattr(args, "rbf_bandwidth_mode", "knn"),
            rbf_knn_k=getattr(args, "rbf_knn_k", 5),
        ).to(device)

    elif args.model == "mlp_baseline":
        model = MLP_BlackBox(
            state_dim=state_dim,
            hidden_dim=getattr(args, "hidden_dim", 64),
            num_layers=getattr(args, "num_layers", 4),
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


def _build_rollout_initial_state(model, X):
    expander = getattr(model, "expander", None)
    delay_depth = int(getattr(expander, "delay_depth", 1)) if expander is not None else 1
    state_dim = int(model.state_dim)

    if delay_depth <= 1:
        return X[0], 0

    if X.shape[0] < delay_depth:
        raise ValueError(
            f"Cannot build a delay history with delay_depth={delay_depth} from only {X.shape[0]} time steps."
        )

    start_idx = delay_depth - 1
    history = X[:delay_depth].flip(0)  # [x(t), x(t-1), ..., x(t-q+1)]
    x0 = history.permute(1, 0, 2).reshape(history.shape[1], delay_depth * state_dim)
    return x0, start_idx

def compute_rollout_metrics(
    model,
    X,
    device,
    eval_horizons=[10, 20, 100],
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

    with torch.no_grad():
        x0, start_idx = _build_rollout_initial_state(model, X)
        max_h = min(max_h, T - 1 - start_idx)

        if max_h < 1:
            return None

        rollout_pred = model.rollout(x0, max_h)

        if not torch.isfinite(rollout_pred).all():
            results["rollout_failed"] = 1.0
            for hh in eval_horizons:
                results[f"rollout_mse_h{hh}"] = np.nan
            return results

        # Compute cumulative MSE for each horizon (matching training loss computation), then take RMSE
        for h in eval_horizons:
            if h <= max_h:
                cumulative_mse = torch.tensor(0.0, device=device)
                for k in range(1, h + 1):
                    diff = rollout_pred[k] - X[start_idx + k]
                    cumulative_mse += torch.mean(diff ** 2)
                avg_mse = cumulative_mse / h
                avg_rmse = torch.sqrt(avg_mse)
                results[f"rollout_rmse_h{h}"] = float(avg_rmse.item())

    return results