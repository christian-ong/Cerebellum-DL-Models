import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def plot_transition_matrix(
    model=None,
    model_name="model",
    figdir=".",
    expand_names=None,
    threshold=1e-3,
    matrix=None,
):
    """
    Visualize learned transition matrices (K/A) and eigen decompositions if present.

    Parameters
    ----------
    model : torch.nn.Module or None
        Model object for NN-based models.
    matrix : np.ndarray or torch.Tensor or None
        Directly supplied matrix to plot (used for non-NN models like manual EDMD).
    """

    K = None
    Lambda = None
    Phi = None

    # --------------------------------------------------
    # Direct matrix input takes priority
    # --------------------------------------------------
    if matrix is not None:
        if isinstance(matrix, torch.Tensor):
            K = matrix.detach().cpu().numpy()
        else:
            K = np.array(matrix)

    # --------------------------------------------------
    # Otherwise extract matrices from model
    # --------------------------------------------------
    elif model is not None:

        # Standard Koopman / linear models
        if hasattr(model, "K"):
            if hasattr(model.K, "weight"):
                K = model.K.weight.detach().cpu().numpy()
            else:
                K = model.K.detach().cpu().numpy()

        # Eigen-DMD style models
        elif hasattr(model, "Lambda") and hasattr(model, "Phi") and hasattr(model, "Phi_inv"):

            Lambda = model.Lambda.detach().cpu().numpy()
            Phi = model.Phi.detach().cpu().numpy()
            Phi_inv = model.Phi_inv.detach().cpu().numpy()

            # If Lambda is stored as a vector, make it diagonal
            if Lambda.ndim == 1:
                Lambda = np.diag(Lambda)

            # reconstruct transition matrix
            K = Phi @ Lambda @ Phi_inv

    else:
        return

    # --------------------------------------------------
    # Print matrices
    # --------------------------------------------------
    def pretty_print_matrix(name, M, threshold=1e-6, precision=3, max_rows=10, max_cols=10):
        if M is None:
            return

        M = np.array(M).copy()

        if np.iscomplexobj(M):
            M.real[np.abs(M.real) < threshold] = 0
            M.imag[np.abs(M.imag) < threshold] = 0
        else:
            M[np.abs(M) < threshold] = 0

        print(f"\n{name} (shape {M.shape})")

        M_display = M[:max_rows, :max_cols]

        with np.printoptions(precision=precision, suppress=True, linewidth=120):
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

    K_plot = np.abs(K) if np.iscomplexobj(K) else K

    plt.figure(figsize=(8, 6))

    plt.title(
        f"Transition Matrix\nModel: {model_name.replace('_',' ')}\n(values > {threshold})"
    )

    plt.imshow(np.abs(K_plot), cmap="viridis", aspect="auto")

    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            value = K[i, j]
            val_abs = abs(value)

            if val_abs >= threshold:
                if np.iscomplexobj(value):
                    label = f"{value.real:.3f}+{value.imag:.3f}j"
                else:
                    label = f"{value:.3f}"

                plt.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=7,
                )

    plt.colorbar(label="|weight|" if np.iscomplexobj(K) else "value")

    plt.xlabel("Current state index")
    plt.ylabel("Next state index")

    if expand_names is not None and len(expand_names) == K.shape[0] == K.shape[1]:
        expand_names = [f"${e}$" for e in expand_names]
        plt.xticks(range(len(expand_names)), expand_names, rotation=90)
        plt.yticks(range(len(expand_names)), expand_names)

    plt.tight_layout()

    os.makedirs(figdir, exist_ok=True)
    plt.savefig(f"{figdir}/transition_matrix.png", dpi=200)
    plt.close()
#  import numpy as np
# import matplotlib.pyplot as plt
# import torch


# def plot_transition_matrix(model, model_name, figdir, expand_names=None, threshold=1e-3):

#     """
#     Visualize learned transition matrices (K/A) and eigen decompositions if present.
#     """

    # K = None
    # Lambda = None
    # Phi = None

    # # --------------------------------------------------
    # # Extract matrices depending on model type
    # # --------------------------------------------------

    # if model is None:
    #     return

    # # Standard Koopman / linear models
    # if hasattr(model, "K"):
    #     K = model.K.weight.detach().cpu().numpy()

    # # Eigen-DMD style models
    # elif hasattr(model, "Lambda") and hasattr(model, "Phi") and hasattr(model, "Phi_inv"):

    #     Lambda = model.Lambda.detach().cpu().numpy()
    #     Phi = model.Phi.detach().cpu().numpy()
    #     Phi_inv = model.Phi_inv.detach().cpu().numpy()

    #     # reconstruct transition matrix
    #     K = Phi @ Lambda @ Phi_inv

    # # --------------------------------------------------
    # # Print matrices
    # # --------------------------------------------------
    # def pretty_print_matrix(name, M, threshold=1e-6, precision=3, max_rows=10, max_cols=10):
    #     """
    #     Nicely print matrices with aligned columns and suppressed small values.
    #     """

    #     if M is None:
    #         return

    #     M = np.array(M)

    #     # suppress very small numbers
    #     M[np.abs(M) < threshold] = 0

    #     print(f"\n{name}  (shape {M.shape})")

    #     # show only a submatrix if very large
    #     M_display = M[:max_rows, :max_cols]

    #     with np.printoptions(
    #         precision=precision,
    #         suppress=True,
    #         linewidth=120
    #     ):
    #         print(M_display)

    #     if M.shape[0] > max_rows or M.shape[1] > max_cols:
    #         print(f"... showing top-left {max_rows}x{max_cols} block")

    # pretty_print_matrix("Transition matrix K/A", K)
    # pretty_print_matrix("Lambda (eigenvalues)", Lambda)
    # pretty_print_matrix("Phi (eigenvectors)", Phi)

    # # --------------------------------------------------
    # # Plot heatmap of K
    # # --------------------------------------------------

    # if K is None:
    #     return

    # plt.figure(figsize=(8, 6))

    # plt.title(
    #     f"Transition Matrix\nModel: {model_name.replace('_',' ')}\n(values > {threshold})"
    # )

    # plt.imshow(np.abs(K), cmap="viridis", aspect="auto")

    # for i in range(K.shape[0]):
    #     for j in range(K.shape[1]):
    #         if abs(K[i, j]) > threshold:
    #             plt.text(
    #                 j,
    #                 i,
    #                 f"{K[i,j]:.2e}",
    #                 ha="center",
    #                 va="center",
    #                 color="red",
    #                 fontsize=7,
    #             )

    # plt.colorbar(label="|weight|")

    # plt.xlabel("Current state index")
    # plt.ylabel("Next state index")

    # # Optional expanded state names
    # if expand_names is not None:
    #     expand_names = [f"${e}$" for e in expand_names]
    #     plt.xticks(range(len(expand_names)), expand_names, rotation=90)
    #     plt.yticks(range(len(expand_names)), expand_names)

    # plt.tight_layout()

    # plt.savefig(f"{figdir}/transition_matrix.png")

    # plt.close()