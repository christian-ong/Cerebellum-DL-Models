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
):

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

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
    batch_val_losses = []
    best_state_dict = None
    best_epoch = -1
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.current_epoch = epoch
        model.train()

        train_loss = 0.0
        n_train = 0
        train_losses = []
        current_lr = optimizer.param_groups[0]["lr"]
        comp_sums = {}

        # Create validation iterator once per epoch
        val_iter = iter(val_loader) if val_loader is not None else None

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
            if hasattr(model, "compute_loss"):
                if future_targets is not None:
                    loss, loss_dict = unpack_loss_output(
                        model.compute_loss(x, y, future_targets)
                    )
                else:
                    loss, loss_dict = unpack_loss_output(model.compute_loss(x, y))
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

            # -------------------
            # Batch validation
            # -------------------
            if val_loader is not None:

                model.eval()

                with torch.no_grad():

                    # get next validation batch
                    try:
                        val_batch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        val_batch = next(val_iter)

                    if len(val_batch) == 2:
                        x_val, y_val = val_batch
                        future_val_targets = None
                    else:
                        x_val, y_val, future_val_targets = val_batch

                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    if future_val_targets is not None:
                        future_val_targets = future_val_targets.to(device, non_blocking=True)
                    if hasattr(model, "compute_loss"):
                        if future_val_targets is not None:
                            val_loss, _ = unpack_loss_output(
                                model.compute_loss(x_val, y_val, future_val_targets)
                            )
                        else:
                            val_loss, _ = unpack_loss_output(model.compute_loss(x_val, y_val))
                    else:
                        y_val_hat = model(x_val)
                        val_loss = loss_fn(y_val_hat, y_val)

                    batch_val_losses.append(val_loss.item())

                model.train()

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
            val_loss_acc = 0.0 # Use a unique name for the accumulator
            n_val = 0

            with torch.no_grad():
                for val_batch in val_loader: # Use val_batch to support optional rollout windows
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
                            batch_l, _ = unpack_loss_output(
                                model.compute_loss(xv, yv, future_val_targets)
                            )
                        else:
                            batch_l, _ = unpack_loss_output(model.compute_loss(xv, yv))
                    else:
                        y_hat = model(xv)
                        batch_l = loss_fn(y_hat, yv)

                    batch_size = xv.size(0)
                    val_loss_acc += batch_l.item() * batch_size # Accumulate batch_l
                    n_val += batch_size

            val_loss = val_loss_acc / n_val # Calculate the final average

        epoch_val_losses.append(val_loss)

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step(val_loss)

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

        maybe_log_matrices(epoch)

    losses = (
        all_train_losses,
        batch_val_losses,
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