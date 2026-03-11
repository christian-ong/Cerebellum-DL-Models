import numpy as np
from scipy.linalg import expm


def compute_system(Ac, params, dt=0.01):

    Ad = expm(Ac * dt)

    eigvals, eigvecs = np.linalg.eig(Ad)

    return {
        "Ac": Ac,
        "Ad": Ad,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "params": params,
        "dt": dt
    }


def print_system(name, result):

    np.set_printoptions(precision=10, suppress=True)

    print("\n==============================")
    print(name)
    print("==============================")

    print("\nA_d:")
    print(np.round(result["Ad"], 10))

    print("\nEigenvalues:")
    print(np.round(result["eigvals"], 10))

    print("\nEigenvectors (Phi):")
    print(np.round(result["eigvecs"], 10))

    print("\nParameters:")
    print(result["params"])

    print("\ndt =", result["dt"])


if __name__ == "__main__":

    dt = 0.01

    systems = {
        "Saddle Point": {
            "Ac": np.array([
                [0.2, 0],
                [0, -0.2]
            ]),
            "params": {}
        },

        "Degenerate Node": {
            "Ac": np.array([
                [-0.7, 0.7],
                [0, -0.7]
            ]),
            "params": {}
        },

        "Inward Spiral": {
            "Ac": np.array([
                [-0.5, -2],
                [2, -0.5]
            ]),
            "params": {}
        },

        "Harmonic Oscillator": {
            "Ac": np.array([
                [0, 1.3],
                [-1.3, 0]
            ]),
            "params": {}
        },

        "Van der Pol": {
            "Ac": np.array([
                [0, 1, 0],
                [-1, 1.5, -1.5],
                [0, 0, 0]
            ]),
            "params": {"mu": 1.5}
        },

        "Lotka Volterra": {
            "Ac": np.array([
                [1.1, 0, -0.4],
                [0, -0.4, 0.1],
                [0, 0, 0]
            ]),
            "params": {"alpha": 1.1, "beta": 0.4, "gamma": 0.4, "delta": 0.1}
        },

        "Pendulum": {
            "Ac": np.array([
                [0, 1, 0],
                [0, 0, -9.81],
                [0, 0, 0]
            ]),
            "params": {"g": 9.81, "L": 1.0}
        },

        "Duffing": {
            "Ac": np.array([
                [0, 1, 0],
                [1.0, -0.2, -1.0],
                [0, 0, 0]
            ]),
            "params": {"alpha": -1.0, "beta": 1.0, "delta": 0.2}
        },

        "Lorenz": {
            "Ac": np.array([
                [-10, 10, 0, 0, 0],
                [28, -1, 0, -1, 0],
                [0, 0, -8/3, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]),
            "params": {"sigma": 10, "rho": 28, "beta": 8/3}
        }

    }

    for name, system in systems.items():

        result = compute_system(system["Ac"], system["params"], dt)

        print_system(name, result)