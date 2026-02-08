import torch


def train_ae_onestep(
    model,
    train_loader,
    val_loader,
    device="cpu",
    epochs=50,
    lr=1e-3,
    weight_decay=1e-6,
):
    """
    Train an autoencoder-based dynamics model using ONE-STEP prediction loss.

    This function is intentionally simple and model-agnostic:
    it works for both linear and nonlinear AE models.

    The model is assumed to return:
        x_next, z, z_next = model(x)
    """

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

    for epoch in range(epochs):
        # -------------------
        # Training
        # -------------------
        model.train()
        train_loss = 0.0
        n_train = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            # One-step prediction
            y_hat, _, _ = model(x)

            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            train_loss += loss.item() * batch_size
            n_train += batch_size

        train_loss /= n_train

        # -------------------
        # Validation
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

                    y_hat, _, _ = model(x)
                    loss = loss_fn(y_hat, y)

                    batch_size = x.size(0)
                    val_loss += loss.item() * batch_size
                    n_val += batch_size

            val_loss /= n_val

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

    return model
