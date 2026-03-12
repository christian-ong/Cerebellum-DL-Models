import numpy as np
import matplotlib.pyplot as plt


def plot_eigenvalues(eigvals, figdir):

    if eigvals is None:
        return

    plt.figure(figsize=(6, 6))

    plt.scatter(eigvals.real, eigvals.imag)

    circle = plt.Circle((0, 0), 1, color="gray", fill=False)
    plt.gca().add_artist(circle)

    plt.xlabel("Real")
    plt.ylabel("Imag")

    plt.title("Eigenvalue spectrum")

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

    plt.tight_layout()

    plt.savefig(f"{figdir}/eigenvalues.png")

    plt.close()