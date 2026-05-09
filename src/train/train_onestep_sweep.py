import torch
import numpy as np
from tqdm import tqdm

def train_onestep_sweep(
    model,
    train_loader,
    val_loader,
    device="cpu",
    epochs=50,
    lr=1e-3,
    weight_decay=1e-6,
    eval_callback=None, 
):

    model = model.to(device)

    def initialize_lifted_normalization(model, loader):
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

        mean = feature_sum / total_count
        var = torch.clamp(feature_sq_sum / total_count - mean**2, min=1e-12)
        scale = torch.sqrt(var)

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

    initialize_lifted_normalization(model, train_loader)

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

    all_train_losses = []
    epoch_val_losses = []
    
    # REMOVED best_state_dict and best_val_loss tracking here

    for epoch in range(epochs):
        model.current_epoch = epoch
        model.train()

        train_loss = 0.0
        n_train = 0
        train_losses = []
        comp_sums = {}

        for batch in tqdm(train_loader):
            if len(batch) == 2:
                x, y = batch
                future_targets = None
            else:
                x, y, future_targets = batch

            x, y = x.to(device), y.to(device)
            if future_targets is not None:
                future_targets = future_targets.to(device)

            optimizer.zero_grad()

            if hasattr(model, "compute_loss"):
                if future_targets is not None:
                    loss, loss_dict = unpack_loss_output(model.compute_loss(x, y, future_targets))
                else:
                    loss, loss_dict = unpack_loss_output(model.compute_loss(x, y))
            else:
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                loss_dict = {"state": loss.item()}

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            batch_size = x.size(0)
            train_loss += loss.item() * batch_size
            for k, v in loss_dict.items():
                comp_sums[k] = comp_sums.get(k, 0.0) + float(v) * batch_size
            n_train += batch_size

        train_loss /= n_train
        all_train_losses.extend(train_losses)
        avg_comps = {k: v / n_train for k, v in comp_sums.items()}

        preferred_order = ["state", "lift", "rollout", "phi_ortho", "unit", "manifold", "same_sign", "lam_sp"]
        ordered_comp_keys = [k for k in preferred_order if k in avg_comps]
        ordered_comp_keys.extend(sorted(k for k in avg_comps if k not in preferred_order))
        comp_str = " | ".join(f"{k} {avg_comps[k]:.2e}" for k in ordered_comp_keys)
        
        # Full validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss_acc = 0.0 
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
                            batch_l, _ = unpack_loss_output(model.compute_loss(xv, yv, future_val_targets))
                        else:
                            batch_l, _ = unpack_loss_output(model.compute_loss(xv, yv))
                    else:
                        y_hat = model(xv)
                        batch_l = loss_fn(y_hat, yv)

                    batch_size = xv.size(0)
                    val_loss_acc += batch_l.item() * batch_size
                    n_val += batch_size

            val_loss = val_loss_acc / n_val 

        epoch_val_losses.append(val_loss)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss is not None:
            msg = f"Epoch {epoch:03d} | lr {current_lr:.2e} | train {train_loss:.4e} | val {val_loss:.4e}"
        else:
            msg = f"Epoch {epoch:03d} | lr {current_lr:.2e} | train {train_loss:.4e}"
            
        if comp_str:
            msg += f" | {comp_str}"
        print(msg)

        # Trigger our callback to log to W&B
        if eval_callback is not None:
            eval_callback(epoch, train_loss, val_loss)

    losses = (all_train_losses, epoch_val_losses, None)

    return model, losses