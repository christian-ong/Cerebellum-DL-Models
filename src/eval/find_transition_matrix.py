import numpy as np
from scipy.linalg import expm




def find_transition_matrix(
    system="linear", 
    A_continuous=np.array([[0.9, 0.1], [0.2, 0.8]]), 
    dt=0.01
    ):

    # Compute the discrete-time transition matrix using the matrix exponential
    if system == "linear":
        A_discrete = expm(A_continuous * dt)
        return A_discrete
    
    elif system == "nonlinear":
        return NotImplementedError("Finding transition matrix for nonlinear systems is not implemented yet.")


if __name__ == "__main__":
    A_continuous = [
        np.array([[-0.5, -2], [ 2,  -0.5]]), # inward spiral
        np.array([[0, 1.3], [-1.3, 0]]), # harmonic oscillator
        np.array([[0.2, 0], [0, -0.2]]), # saddle point
        np.array([[-0.7, 0.7], [0, -0.7]]) # degenerate node
        ][3]
    A_discrete = find_transition_matrix(A_continuous=A_continuous)

    print("Continuous-time matrix A:")
    print(A_continuous)
    print("\nDiscrete-time transition matrix A_discrete:")
    print(A_discrete)