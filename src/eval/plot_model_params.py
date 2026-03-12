import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_training_losses(
        file_name="manual_expansion_ml_dmd_harmonic_oscillator", 
        ignore_first_epochs=0
    ):

    data = np.load(f"data/models/{file_name}_losses.npz")

    for key in data.keys():
        print(f"{key}:    \tshape {data[key].shape}, dtype {data[key].dtype}")

    train_losses = data["train_losses"]
    batch_val_losses = data["batch_val_losses"]
    epoch_val_losses = data["epoch_val_losses"]
    n_epochs = len(epoch_val_losses)
    steps_per_epoch = len(train_losses) // n_epochs

    if ignore_first_epochs != 0:
        train_losses = train_losses[steps_per_epoch * ignore_first_epochs:]
        batch_val_losses = batch_val_losses[steps_per_epoch * ignore_first_epochs:]
        epoch_val_losses = epoch_val_losses[ignore_first_epochs:]
        n_epochs -= ignore_first_epochs

    print(f"Train losses shape: {train_losses.shape}")
    print(f"Batch validation losses shape: {batch_val_losses.shape}")
    print(f"Epoch validation losses shape: {epoch_val_losses.shape}")

    plt.figure(figsize=(8, 6))
    plt.title(f"Training Loss, model: {file_name.replace('_', ' ')}")
    plt.plot(train_losses, label="Training Loss")
    plt.plot(batch_val_losses, label="Batch validation Loss")
    # plot epoch_val_losses as a step function (constant between epochs)
    plt.step(np.arange(0, n_epochs * steps_per_epoch, steps_per_epoch), epoch_val_losses, where='post', label="Epoch validation Loss")

    plt.xlabel("step")
    plt.ylabel("MSE Loss")
    plt.xticks(
        np.arange(
            0, 
            n_epochs*len(train_losses)//n_epochs+1, 
            len(train_losses)//n_epochs), 
        np.arange(
            ignore_first_epochs, 
            n_epochs + ignore_first_epochs + 1)
        )
    plt.grid()
    plt.legend()
    plt.show()


def plot_transition_matrix(
        model_name="manual_expansion_ml_dmd_harmonic_oscillator", 
        print_K_xy=False
        ):
    
    data = torch.load(f"data/models/{model_name}.pt")
    print(data['expand_names'])
    print(data.keys())
    K = data['model_state_dict']['K.weight']
    expand_names = [f"${e}$" for e in data['expand_names']]

    # Print the top-left 2x2 (x,y)
    if print_K_xy:
        print("K matrix:")
        print(K[:2, :2])

    # Plot K matrix as a heatmap
    plt.figure(figsize=(8, 6))
    plt.title(f"K matrix\nmodel: {model_name.replace('_', ' ')}\n (showing values > 1e-3)")
    plt.imshow(abs(K), cmap='viridis', aspect='auto')

    # show numbers on the heatmap
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if abs(K[i, j]) > 1e-3:  # only print values above a threshold for readability
                plt.text(j, i, f"{K[i, j]:.2e}", ha='center', va='center', color='red', fontsize=8, rotation=30)

    plt.colorbar(label='|weight|')
    plt.xlabel("Expanded State Index")
    plt.ylabel("Next State Index")
    plt.xticks(range(len(data['expand_names'])), expand_names)
    plt.yticks(range(len(data['expand_names'])), expand_names)
    plt.show()


if __name__ == "__main__":
    expansion = ""
    model = "ml_eigen_dmd"
    system = ["saddle_point", "degenerate_node", "inward_spiral", "harmonic_oscillator"][0]

    # model_name = "manual_expansion_ml_dmd_saddle_point"
    model_name = f"{expansion}{model}_{system}"

    # plot_training_losses(model_name, ignore_first_epochs=0)
    # plot_transition_matrix(model_name, print_K_xy=True)

    data = np.load(f"data/models/{model_name}_losses.npz", allow_pickle=True)

    # Print data summary
    for key in data.keys():
        if key == "loss_components_val":
            print(key)
            for key2 in data[key].item().keys():
                print(f"\t{key2}:" + ('\t' * (3 - (len(key2)+1)//8)) + f"shape {np.array(data[key].item()[key2]).shape}, dtype {np.array(data[key].item()[key2]).dtype}")
        else:
            print(f"{key}:    \tshape {data[key].shape}, dtype {data[key].dtype}")

    # Plot loss components
    loss_components = data["loss_components_val"].item()
    n_components = len(loss_components)
    n_rows = (n_components + 1) // 2
    fig, ax = plt.subplots(n_rows, 2, figsize=(12, 3*n_rows))
    plt.suptitle(f"Validation Loss Components, model: {model_name.replace('_', ' ')}")
    
    for i, key2 in enumerate(loss_components.keys()):
        loss_data = np.array(loss_components[key2])
        j, k = i//2, i%2
        ax[j,k].plot(loss_data, label=key2)
        ax[j,k].set_title(f"{key2}")
        ax[j,k].set_ylabel("Loss")
        ax[j,k].legend()
        ax[j,k].grid()
        if j == n_rows - 1:
            ax[j,k].set_xlabel("Epoch")
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.1)
    plt.show()