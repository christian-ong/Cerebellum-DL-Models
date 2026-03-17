import numpy as np
import matplotlib.pyplot as plt
import os


def plot_training_losses(loss_file, figdir, ignore_first_epochs=0):

    data = np.load(loss_file, allow_pickle=True)

    train_losses = data["train_losses"]
    batch_val_losses = data["batch_val_losses"]
    epoch_val_losses = data["epoch_val_losses"]

    # Loss components may not exist for all models
    loss_components = None
    if "loss_components_val" in data:
        try:
            loss_components = data["loss_components_val"].item()
        except:
            loss_components = None

    model_name = os.path.basename(loss_file).replace("_losses.npz", "")
    model_name_clean = model_name.replace("_", " ")

    n_epochs = len(epoch_val_losses)
    steps_per_epoch = len(train_losses) // n_epochs

    if ignore_first_epochs != 0:

        train_losses = train_losses[steps_per_epoch * ignore_first_epochs:]
        batch_val_losses = batch_val_losses[steps_per_epoch * ignore_first_epochs:]
        epoch_val_losses = epoch_val_losses[ignore_first_epochs:]

        if loss_components is not None:
            for key in loss_components:
                loss_components[key] = loss_components[key][ignore_first_epochs:]

        n_epochs -= ignore_first_epochs

    # --------------------------------------------------
    # Plot 1: Training losses
    # --------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    plt.suptitle(f"Training Losses\nModel: {model_name_clean}")

    axes[0].plot(train_losses)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid()

    axes[1].plot(batch_val_losses)
    axes[1].set_title("Batch Validation Loss")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].grid()

    axes[2].plot(epoch_val_losses)
    axes[2].set_title("Epoch Validation Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].grid()

    plt.tight_layout()
    plt.savefig(f"{figdir}/training_losses.png")
    plt.close()

    # --------------------------------------------------
    # Plot 2: Loss components (only if they exist)
    # --------------------------------------------------

    if loss_components is None or len(loss_components) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    plt.suptitle(f"Validation Loss Components\nModel: {model_name_clean}")

    component_names = list(loss_components.keys())

    for i, key in enumerate(component_names):

        row = i // 2
        col = i % 2

        axes[row, col].plot(loss_components[key])
        axes[row, col].set_title(key)
        axes[row, col].set_xlabel("Epoch")
        axes[row, col].set_ylabel("Loss")
        axes[row, col].grid()

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    plt.savefig(f"{figdir}/loss_components.png")
    plt.close()