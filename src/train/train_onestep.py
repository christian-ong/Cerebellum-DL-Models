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
    all_train_losses = []
    epoch_val_losses = []
    batch_val_losses = []
    loss_components_val = {"predict": [], "orthogonal": [], "phi_inv": [], "unit_length": []}

    for epoch in range(epochs):
        # -------------------
        # Training
        # -------------------
        train_loss = 0.0
        n_train = 0
        train_losses = []

        for x, y in tqdm(train_loader):
            model.train()
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            # One-step prediction loss
            if hasattr(model, "compute_loss"): # multiple loss components
                loss = model.compute_loss(x, y)
                loss_predict, loss_orthogonal, loss_phi_inv, loss_unit_length = loss # can plot/weight these
                loss = sum(loss)

            else:
                y_hat, _, _ = model(x)
                loss = loss_fn(y_hat, y)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            batch_size = x.size(0)
            train_loss += loss.item() * batch_size
            n_train += batch_size

            # Validation loss on single batch (during training)
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    x_val, y_val = next(iter(val_loader))
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    if hasattr(model, "compute_loss"): # multiple loss components
                        val_loss = model.compute_loss(x_val, y_val)
                        loss_predict, loss_orthogonal, loss_phi_inv, loss_unit_length = val_loss # can plot/weight these
                        val_loss = sum(val_loss)
                        batch_val_losses.append(val_loss.item())

                        # Individual loss components
                        loss_components_val["predict"].append(loss_predict.item())
                        loss_components_val["orthogonal"].append(loss_orthogonal.item())
                        loss_components_val["phi_inv"].append(loss_phi_inv.item())
                        loss_components_val["unit_length"].append(loss_unit_length.item())

                    else:
                        y_val_hat, _, _ = model(x_val)
                        val_loss = loss_fn(y_val_hat, y_val)
                        batch_val_losses.append(val_loss.item())
                model.train()

        train_loss /= n_train
        all_train_losses.extend(train_losses)

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

                    if hasattr(model, "compute_loss"): # multiple loss components
                        loss = model.compute_loss(x, y)
                        loss = sum(loss)
                    else:
                        y_hat, _, _ = model(x)
                        loss = loss_fn(y_hat, y)

                    batch_size = x.size(0)
                    val_loss += loss.item() * batch_size
                    n_val += batch_size

            val_loss /= n_val

        epoch_val_losses.append(val_loss)

        scheduler.step()

        # Print progress
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
        
    if hasattr(model, "K"):    
        K_matrix = model.K.weight.detach().T
        print(f"K matrix:\n{K_matrix.cpu().numpy()}")
        eigvals, eigvecs = torch.linalg.eig(K_matrix)
        print(f"Eigenvalues:\n{eigvals.cpu().numpy()}")
        print(f"Eigenvectors:\n{eigvecs.cpu().numpy()}")
    
    losses = all_train_losses, batch_val_losses, epoch_val_losses, loss_components_val

    return model, losses
                                                                                                                 