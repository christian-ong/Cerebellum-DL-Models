import numpy as np
import torch

from src.models.ml_linear_dynamics import ML_LinearDynamics
from src.models.ml_dmd import ML_DMD
from src.models.ml_dmd_drop import ML_DMD_DROP
from src.models.regression_dmd import Regression_DMD
from src.models.mlp_baseline import MLP_BlackBox
from src.models.sindy_baseline import SINDyBaseline
from src.models.regression_dmd import Regression_DMD

def build_run_name(args, system_name, run_id=None):
    parts = [system_name, args.model]
    parts.append(f"lr{args.lr:.0e}")
    parts.append(f"bs{args.batch_size}")

    if args.model == "sindy_baseline":
        parts.append(f"lib{getattr(args, 'sindy_library_type', 'polynomial')}")
        parts.append(f"dt{getattr(args, 'sindy_discrete_time', 'false')}")
        parts.append(f"p{getattr(args, 'sindy_poly_order', 3)}")
        parts.append(f"th{getattr(args, 'sindy_threshold', 0.1):.0e}")
        basis_size = getattr(args, "sindy_specific_basis_size", None)
        if basis_size is not None:
            parts.append(f"k{basis_size}")
    elif hasattr(args, "expansion_type"):
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

    elif args.model in {"ml_dmd_drop"}:
        model = ML_DMD_DROP(
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
            biorth_weight=getattr(args, "biorth_weight", 0.1),
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

    elif args.model == "sindy_baseline":
        model = SINDyBaseline(
            discrete_time=getattr(args, "sindy_discrete_time", "false") == "true",
            poly_order=getattr(args, "sindy_poly_order", 3),
            include_bias=getattr(args, "sindy_include_bias", "true") == "true",
            include_interaction=getattr(args, "sindy_include_interaction", "true") == "true",
            threshold=getattr(args, "sindy_threshold", 0.1),
            alpha=getattr(args, "sindy_alpha", 0.0),
            differentiation_method=getattr(args, "sindy_diff_method", "finite_difference"),
            library_type=getattr(args, "sindy_library_type", "polynomial"),
            fourier_n_frequencies=getattr(args, "sindy_fourier_n_frequencies", 1),
            specific_system=system_name if getattr(args, "sindy_library_type", "polynomial") == "specific" else None,
            specific_basis_size=getattr(args, "sindy_specific_basis_size", None),
        )

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    return model


def _predict_next_batch(model, x, device):
    if isinstance(model, Regression_DMD):
        try:
            return model(x)
        except Exception:
            # Best-effort fallback for edge cases where batched prediction
            # is unavailable for a loaded checkpoint.
            preds = []
            for xi in x:
                rollout = model.rollout(xi.detach().cpu().numpy(), 1)
                preds.append(torch.as_tensor(rollout[1], device=device, dtype=x.dtype))
            return torch.stack(preds, dim=0)

    if isinstance(model, torch.nn.Module):
        return model(x)

    if not hasattr(model, "rollout"):
        raise TypeError(f"Model {type(model).__name__} does not support batched prediction")

    preds = []
    for xi in x:
        rollout = model.rollout(xi.detach().cpu().numpy(), 1)
        preds.append(torch.as_tensor(rollout[1], device=device, dtype=x.dtype))
    return torch.stack(preds, dim=0)


def compute_loader_metrics(model, loader, device):
    if loader is None or len(loader.dataset) == 0:
        return None, None

    if isinstance(model, torch.nn.Module):
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

            y_hat = _predict_next_batch(model, x, device)
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
    state_dim = int(getattr(model, "state_dim", X.shape[-1]))

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

    results = {"rollout_failed": 0.0}

    # If no model provided, mark rollouts as failed and return NaNs to avoid calling None.rollout
    if model is None:
        results["rollout_failed"] = 1.0
        for h in eval_horizons:
            results[f"rollout_rmse_h{h}"] = np.nan
        return results

    with torch.no_grad():
        x0, start_idx = _build_rollout_initial_state(model, X)
        max_h = min(max_h, T - 1 - start_idx)

        if max_h < 1:
            return None

        if isinstance(model, torch.nn.Module):
            model.eval()
            rollout_pred = model.rollout(x0, max_h)
        else:
            # non-torch models may implement single-trajectory rollout; leave rollout_pred None
            rollout_pred = None

        if rollout_pred is not None:
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

        # Non-torch baselines like SINDy only support single-trajectory rollouts,
        # so evaluate trajectory-by-trajectory.
        n_traj = X.shape[1]
        for h in eval_horizons:
            if h > max_h:
                continue

            cumulative_rmse = 0.0
            counted = 0

            for traj_idx in range(n_traj):
                x0_single = X[start_idx, traj_idx].detach().cpu().numpy()
                rollout_pred_single = model.rollout(x0_single, h)
                rollout_pred_single = torch.as_tensor(rollout_pred_single, device=device, dtype=X.dtype)

                if not torch.isfinite(rollout_pred_single).all():
                    results["rollout_failed"] = 1.0
                    results[f"rollout_rmse_h{h}"] = np.nan
                    break

                cumulative_mse = torch.tensor(0.0, device=device)
                for k in range(1, h + 1):
                    diff = rollout_pred_single[k] - X[start_idx + k, traj_idx]
                    cumulative_mse += torch.mean(diff ** 2)

                avg_mse = cumulative_mse / h
                cumulative_rmse += float(torch.sqrt(avg_mse).item())
                counted += 1

            if counted > 0 and results.get(f"rollout_rmse_h{h}") is None:
                results[f"rollout_rmse_h{h}"] = cumulative_rmse / counted

    return results