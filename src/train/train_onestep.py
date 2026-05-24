import torch
import numpy as np
from tqdm import tqdm


def train_onestep(
    model,
    train_loader,
    val_loader,
    device="cpu",
    epochs=50,
    lr=1e-3,
    weight_decay=1e-6,
    log_phi_every=0,
    phi_print_max_dim=12,
    eval_callback=None,
    rollout_horizon=None,
):

    model = model.to(device)

    def initialize_lifted_normalization(model, loader):
        if getattr(model, "_lift_stats_initialized", False):
            return
        if not hasattr(model, "set_lifted_normalization_stats"):
            return
        if not hasattr(model, "expander") or not hasattr(model, "expand_names"):
            return

        feature_sum = None
        feature_sq_sum = None
        total_count = 0

        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device)
                z = model.expander.expand(x)

                if feature_sum is None:
                    feature_sum = torch.zeros(z.shape[1], dtype=torch.float64, device=z.device)
                    feature_sq_sum = torch.zeros(z.shape[1], dtype=torch.float64, device=z.device)

                z64 = z.to(torch.float64)
                feature_sum += z64.sum(dim=0)
                feature_sq_sum += (z64 ** 2).sum(dim=0)
                total_count += z64.shape[0]

        if total_count == 0:
            return

        # We do NOT subtract the mean (to preserve the Koopman origin).
        # Therefore, we MUST scale by the Root Mean Square (RMS), not Standard Deviation.
        mean = feature_sum / total_count
        rms_sq = torch.clamp(feature_sq_sum / total_count, min=1e-4) # E[x^2]
        scale = torch.sqrt(rms_sq)

        fixed_mask = []
        for name in model.expand_names:
            fixed_mask.append(name == "1" or ("sin(" in name) or ("cos(" in name))

        if any(fixed_mask):
            fixed_mask = torch.tensor(fixed_mask, device=mean.device, dtype=torch.bool)
            mean = mean.clone()
            scale = scale.clone()
            mean[fixed_mask] = 0.0
            scale[fixed_mask] = 1.0

        model.set_lifted_normalization_stats(mean, scale)
        model._lift_stats_initialized = True

    # 1. Fit the MaxAbs state scaler (supports both Koopman Expanders and standard MLPs)
    has_expander_scaler = hasattr(model, "expander") and hasattr(model.expander, "fit_state_scaler")
    has_model_scaler = hasattr(model, "fit_state_scaler")

    # --- NEW: Only fit if not already initialized by train.py ---
    if not getattr(model, "_state_scaler_initialized", False):
        if has_expander_scaler or has_model_scaler:
            all_x = []
            for batch in train_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                all_x.append(x)
                
            full_X_train = torch.cat(all_x, dim=0).to(device)
            
            if has_expander_scaler:
                model.expander.fit_state_scaler(full_X_train)
            elif has_model_scaler:
                model.fit_state_scaler(full_X_train)
            
            model._state_scaler_initialized = True

    # 2. THEN, calculate the lifted stats using the newly safe, bounded polynomials
    initialize_lifted_normalization(model, train_loader)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    warmup_epochs = min(5, max(1, int(epochs)))

    # 1. Warmup Phase: Linearly scale up to the initial lr
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs,
    )

    # 2/3. If there are epochs after warmup, add a cosine decay phase.
    # For short debug runs (e.g. epochs <= warmup), keep warmup-only scheduling.
    if int(epochs) > warmup_epochs:
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(epochs) - warmup_epochs,
            eta_min=0.0,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = warmup_scheduler
    loss_fn = torch.nn.MSELoss()

    def unpack_loss_output(loss_output):
        """Normalize model-specific compute_loss return values.

        Supported forms:
        - loss tensor
        - (loss,)
        - (loss, loss_dict)
        """
        if isinstance(loss_output, torch.Tensor):
            return loss_output, {}

        if isinstance(loss_output, (tuple, list)):
            if len(loss_output) == 0:
                raise ValueError("compute_loss returned an empty tuple/list")

            loss = loss_output[0]
            loss_dict = {}

            if len(loss_output) > 1 and isinstance(loss_output[1], dict):
                loss_dict = loss_output[1]

            return loss, loss_dict

        raise TypeError(
            "compute_loss must return a Tensor, a (loss,) tuple, or a (loss, loss_dict) tuple"
        )

    def maybe_log_matrices(epoch_idx):
        """Print physical Phi and Lambda during training when available.

        This gives immediate feedback on whether the descaled Phi and the 
        dynamics matrix Lambda are moving towards the expected structure.
        """
        if log_phi_every <= 0:
            return
        if (epoch_idx % log_phi_every) != 0 and epoch_idx != epochs - 1:
            return
        
        # We need both methods to log successfully
        if not (hasattr(model, "get_Phi") and hasattr(model, "get_Lambda")):
            return

        try:
            with torch.no_grad():
                phi = model.get_Phi().detach().cpu().float().numpy()
                lam = model.get_Lambda().detach().cpu().float().numpy()
        except Exception as exc:
            print(f"Matrix log skipped at epoch {epoch_idx:03d}: {exc}")
            return

        if phi.ndim != 2 or lam.ndim != 2:
            print(f"Epoch {epoch_idx:03d} | Matrices are not 2D.")
            return

        rows, cols = phi.shape
        diag_phi = np.diag(phi)
        offdiag_phi = phi - np.diag(diag_phi)
        
        try:
            cond_phi = float(np.linalg.cond(phi))
            cond_str = f"{cond_phi:.2e}"
        except Exception:
            cond_str = "nan"

        print(
            f"Epoch {epoch_idx:03d} | Shape {rows}x{cols} | "
            f"Phi Cond: {cond_str} | "
            f"Phi Diag Mean: {np.mean(np.abs(diag_phi)):.2e} | "
            f"Phi Off-Diag Mean: {np.mean(np.abs(offdiag_phi)):.2e}"
        )

        # Truncate for printing if necessary
        print_dim = min(rows, phi_print_max_dim)
        phi_block = phi[:print_dim, :print_dim]
        lam_block = lam[:print_dim, :print_dim]
        
        if rows > phi_print_max_dim:
             print(f"Showing top-left {print_dim}x{print_dim} block:")

        # We will format them as strings, split by line, and print side-by-side
        phi_str = np.array2string(phi_block, precision=3, suppress_small=True, max_line_width=120)
        lam_str = np.array2string(lam_block, precision=3, suppress_small=True, max_line_width=120)
        
        phi_lines = phi_str.split('\n')
        lam_lines = lam_str.split('\n')
        
        print(f"{'Phi (Observation -> Modes)'.center(60)} | {'Lambda (Modal Dynamics)'.center(60)}")
        print("-" * 123)
        
        # Zip them together, padding with empty strings if one is somehow longer
        for p_line, l_line in zip(phi_lines, lam_lines):
            # Pad the Phi line to 60 characters so the divider aligns
            p_padded = p_line.ljust(60)
            print(f"{p_padded} | {l_line}")
        print("-" * 123)

    all_train_losses = []
    epoch_val_losses = []
    best_state_dict = None
    best_epoch = -1
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.current_epoch = epoch
        model.train()

        # Set rollout horizon for this epoch
        if rollout_horizon is not None:
            model.rollout_horizon = int(rollout_horizon)

        train_loss = 0.0
        n_train = 0
        train_losses = []
        current_lr = optimizer.param_groups[0]["lr"]
        comp_sums = {}

        for batch in tqdm(train_loader):

            if len(batch) == 2:
                x, y = batch
                future_targets = None
            else:
                x, y, future_targets = batch

            x = x.to(device)
            y = y.to(device)
            if future_targets is not None:
                future_targets = future_targets.to(device)

            optimizer.zero_grad()

            # -------------------
            # Training loss
            # -------------------
            x_for_loss = x

            if hasattr(model, "compute_loss"):
                if future_targets is not None:
                    loss, loss_dict = unpack_loss_output(
                        model.compute_loss(x_for_loss, y, future_targets)
                    )
                else:
                    loss, loss_dict = unpack_loss_output(model.compute_loss(x_for_loss, y))
            else:
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                # Baselines that do not expose component losses only report state loss.
                loss_dict = {"state": loss.item()}

            # -------------------
            # Backprop
            # -------------------
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

            # Accumulate for terminal logging
            batch_size = x.size(0)
            train_loss += loss.item() * batch_size
            for k, v in loss_dict.items():
                comp_sums[k] = comp_sums.get(k, 0.0) + float(v) * batch_size
            n_train += batch_size

        train_loss /= n_train
        all_train_losses.extend(train_losses)
        avg_comps = {k: v / n_train for k, v in comp_sums.items()}

        preferred_order = [
            "state",
            "lift",       # Added lift to track the base Koopman loss
            "rollout",    # Added rollout
            "phi_ortho",
            "unit",
            "manifold",   # Replaced old lam_* keys with the new structural keys
            "same_sign",
            "lam_sp",
        ]
        ordered_comp_keys = [k for k in preferred_order if k in avg_comps]
        ordered_comp_keys.extend(sorted(k for k in avg_comps if k not in preferred_order))
        comp_str = " | ".join(f"{k} {avg_comps[k]:.2e}" for k in ordered_comp_keys)
        
        # -------------------
        # Full validation
        # -------------------
        val_loss = None

        if val_loader is not None:
            model.eval()
            val_loss_acc = 0.0
            val_state_loss_acc = 0.0
            n_val = 0

            with torch.no_grad():
                for val_batch in val_loader:
                    if len(val_batch) == 2:
                        xv, yv = val_batch
                        future_val_targets = None
                    else:
                        xv, yv, future_val_targets = val_batch

                    xv, yv = xv.to(device), yv.to(device)
                    if future_val_targets is not None:
                        future_val_targets = future_val_targets.to(device)

                    if hasattr(model, "compute_loss"):
                        if future_val_targets is not None:
                            batch_l, batch_loss_dict = unpack_loss_output(
                                model.compute_loss(xv, yv, future_val_targets)
                            )
                        else:
                            batch_l, batch_loss_dict = unpack_loss_output(model.compute_loss(xv, yv))
                    else:
                        y_hat = model(xv)
                        batch_l = loss_fn(y_hat, yv)
                        batch_loss_dict = {"state": batch_l.item()}

                    batch_size = xv.size(0)
                    val_loss_acc += batch_l.item() * batch_size
                    # Track state loss separately for scheduler
                    state_loss = batch_loss_dict.get("state", batch_l.item())
                    val_state_loss_acc += state_loss * batch_size
                    n_val += batch_size

            val_loss = val_loss_acc / n_val
            val_state_loss = val_state_loss_acc / n_val

        epoch_val_losses.append(val_loss)

        if val_state_loss is not None and val_state_loss < best_val_loss:
            best_val_loss = float(val_state_loss)
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        # -------------------
        # Scheduler Step
        # -------------------
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss is not None:
            msg = f"Epoch {epoch:03d} | lr {current_lr:.2e} | train {train_loss:.4e} | val {val_loss:.4e}"
            if comp_str:
                msg += f" | {comp_str}"
            print(msg)
        else:
            msg = f"Epoch {epoch:03d} | lr {current_lr:.2e} | train {train_loss:.4e}"
            if comp_str:
                msg += f" | {comp_str}"
            print(msg)

        if eval_callback is not None:
            eval_callback(epoch, train_loss, val_loss)

        maybe_log_matrices(epoch)

    losses = (
        all_train_losses,
        epoch_val_losses,
        None,
    )

    if best_state_dict is None:
        best_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_epoch = epochs - 1
        best_val_loss = None

    best_checkpoint = {
        "state_dict": best_state_dict,
        "epoch": best_epoch,
        "val_loss": best_val_loss,
    }

    return model, losses, best_checkpoint