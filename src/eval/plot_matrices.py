import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_transition_matrix(model, model_name, figdir, expand_names=None, threshold=1e-3):

    """
    Visualize learned transition matrices (K/A) and eigen decompositions if present.
    """

    K = None
    Lambda = None
    Phi = None

    # --------------------------------------------------
    # Extract matrices depending on model type
    # --------------------------------------------------

    if model is None:
        return

    # Standard Koopman / linear models
    if hasattr(model, "K"):
        K = model.K.weight.detach().cpu().numpy()

    # Eigen-DMD style models
    elif hasattr(model, "Lambda") and hasattr(model, "Phi") and hasattr(model, "Phi_inv"):

        Lambda = model.Lambda.detach().cpu().numpy()
        Phi = model.Phi.detach().cpu().numpy()
        Phi_inv = model.Phi_inv.detach().cpu().numpy()

        # reconstruct transition matrix
        K = Phi @ Lambda @ Phi_inv

    # --------------------------------------------------
    # Print matrices
    # --------------------------------------------------
    def pretty_print_matrix(name, M, threshold=1e-6, precision=3, max_rows=10, max_cols=10):
        """
        Nicely print matrices with aligned columns and suppressed small values.
        """

        if M is None:
            return

        M = np.array(M)

        # suppress very small numbers
        M[np.abs(M) < threshold] = 0

        print(f"\n{name}  (shape {M.shape})")

        # show only a submatrix if very large
        M_display = M[:max_rows, :max_cols]

        with np.printoptions(
            precision=precision,
            suppress=True,
            linewidth=120
        ):
            print(M_display)

        if M.shape[0] > max_rows or M.shape[1] > max_cols:
            print(f"... showing top-left {max_rows}x{max_cols} block")

    pretty_print_matrix("Transition matrix K/A", K)
    pretty_print_matrix("Lambda (eigenvalues)", Lambda)
    pretty_print_matrix("Phi (eigenvectors)", Phi)

    # --------------------------------------------------
    # Plot heatmap of K
    # --------------------------------------------------

    if K is None:
        return

    plt.figure(figsize=(8, 6))

    plt.title(
        f"Transition Matrix\nModel: {model_name.replace('_',' ')}\n(values > {threshold})"
    )

    plt.imshow(np.abs(K), cmap="viridis", aspect="auto")

    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if abs(K[i, j]) > threshold:
                plt.text(
                    j,
                    i,
                    f"{K[i,j]:.2e}",
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=7,
                )

    plt.colorbar(label="|weight|")

    plt.xlabel("Current state index")
    plt.ylabel("Next state index")

    # Optional expanded state names
    if expand_names is not None:
        expand_names = [f"${e}$" for e in expand_names]
        plt.xticks(range(len(expand_names)), expand_names, rotation=90)
        plt.yticks(range(len(expand_names)), expand_names)

    plt.tight_layout()

    plt.savefig(f"{figdir}/transition_matrix.png")

    plt.close()