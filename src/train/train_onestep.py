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
):

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=100,
        gamma=0.5,
    )

    loss_fn = torch.nn.MSELoss()

    all_train_losses = []
    epoch_val_losses = []
    batch_val_losses = []

    for epoch in range(epochs):

        model.train()

        train_loss = 0.0
        n_train = 0
        train_losses = []

        # Create validation iterator once per epoch
        val_iter = iter(val_loader) if val_loader is not None else None

        for x, y in tqdm(train_loader):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            # -------------------
            # Training loss
            # -------------------
            if hasattr(model, "compute_loss"):

                loss_tuple = model.compute_loss(x, y)

                if isinstance(loss_tuple, torch.Tensor):
                    loss = loss_tuple
                else:
                    loss = sum(loss_tuple)

            else:

                y_hat = model(x)
                loss = loss_fn(y_hat, y)

            # -------------------
            # Backprop
            # -------------------
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

            batch_size = x.size(0)
            train_loss += loss.item() * batch_size
            n_train += batch_size

            # -------------------
            # Batch validation
            # -------------------
            if val_loader is not None:

                model.eval()

                with torch.no_grad():

                    # get next validation batch
                    try:
                        x_val, y_val = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        x_val, y_val = next(val_iter)

                    x_val = x_val.to(device)
                    y_val = y_val.to(device)

                    if hasattr(model, "compute_loss"):

                        val_tuple = model.compute_loss(x_val, y_val)

                        if isinstance(val_tuple, torch.Tensor):
                            val_loss = val_tuple
                        else:
                            val_loss = sum(val_tuple)

                    else:

                        y_val_hat = model(x_val)
                        val_loss = loss_fn(y_val_hat, y_val)

                    batch_val_losses.append(val_loss.item())

                model.train()

        train_loss /= n_train
        all_train_losses.extend(train_losses)

        # -------------------
        # Full validation
        # -------------------
        val_loss = None

        if val_loader is not None:

            model.eval()

            val_loss = 0.0
            n_val = 0

            with torch.no_grad():

                for x, y in val_loader:

                    x = x.to(device)
                    y = y.to(device)

                    if hasattr(model, "compute_loss"):

                        val_tuple = model.compute_loss(x, y)

                        if isinstance(val_tuple, torch.Tensor):
                            loss = val_tuple
                        else:
                            loss = sum(val_tuple)

                    else:

                        y_hat = model(x)
                        loss = loss_fn(y_hat, y)

                    batch_size = x.size(0)
                    val_loss += loss.item() * batch_size
                    n_val += batch_size

            val_loss /= n_val

        epoch_val_losses.append(val_loss)

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss is not None:
            print(
                f"Epoch {epoch:03d} | "
                f"lr {current_lr:.2e} | "
                f"train {train_loss:.6e} | "
                f"val {val_loss:.6e}"
            )
        else:
            print(
                f"Epoch {epoch:03d} | "
                f"lr {current_lr:.2e} | "
                f"train {train_loss:.6e}"
            )

    losses = (
        all_train_losses,
        batch_val_losses,
        epoch_val_losses,
        None,
    )

    return model, losses