import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def plot_transition_matrix(
    model=None,
    model_name="model",
    figdir=".",
    expand_names=None,
    threshold=1e-2,
    matrix=None,
):
    """
    Visualize learned transition matrices (K/A) and eigen decompositions if present.

    Parameters
    ----------
    model : torch.nn.Module or None
        Model object for NN-based models.
    matrix : np.ndarray or torch.Tensor or None
        Directly supplied matrix to plot (used for non-NN models like regression DMD).
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
    # Otherwise extract matrices from model helpers
    # --------------------------------------------------
    elif model is not None:

        if hasattr(model, "get_K_true"):
            K_obj = model.get_K_true()
            K = K_obj.detach().cpu().numpy() if torch.is_tensor(K_obj) else np.array(K_obj)

        if hasattr(model, "get_Lambda"):
            Lambda_obj = model.get_Lambda()
            Lambda = (
                Lambda_obj.detach().cpu().numpy()
                if torch.is_tensor(Lambda_obj)
                else np.array(Lambda_obj)
            )

        if hasattr(model, "get_Phi_true"):
            Phi_obj = model.get_Phi_true()
            Phi = (
                Phi_obj.detach().cpu().numpy()
                if torch.is_tensor(Phi_obj)
                else np.array(Phi_obj)
            )

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
    pretty_print_matrix("Lambda", Lambda)
    pretty_print_matrix("Phi", Phi)

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

    plt.imshow(K_plot, cmap="viridis", aspect="auto")

    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            value = K[i, j]
            val_abs = abs(value)

            if val_abs >= threshold:
                if np.iscomplexobj(value):
                    if abs(value.imag) < 1e-10:
                        label = f"{value.real:.3f}"
                    else:
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

    plt.colorbar(label="|value|" if np.iscomplexobj(K) else "value")

    plt.xlabel("Current feature")
    plt.ylabel("Next feature")

    if expand_names is not None and len(expand_names) == K.shape[0] == K.shape[1]:
        expand_names_fmt = [f"${e}$" for e in expand_names]
        plt.xticks(range(len(expand_names_fmt)), expand_names_fmt, rotation=90)
        plt.yticks(range(len(expand_names_fmt)), expand_names_fmt)

    plt.tight_layout()

    os.makedirs(figdir, exist_ok=True)
    plt.savefig(f"{figdir}/transition_matrix.png", dpi=200)
    plt.close()