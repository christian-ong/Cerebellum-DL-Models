import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# ==========================================================
# Construct the Continuous Matrix A_c

def get_system_matrices(system="saddle_point", print_matrices=False, plot_phi=False):
    
    # system values
    vp_mu = 1.5 # vanderpol
    lv_al = 1.1 # lotka-volterra
    lv_be = 0.4 
    lv_ga = 0.4 
    lv_de = 0.1
    pe_g = 9.81 # pendulum
    pe_l = 1.0
    du_al = -1.0 # duffing
    du_be = 1.0
    du_de = 0.2
    lo_sigma = 10.0 # lorenz
    lo_rho = 28.0
    lo_beta = 8.0 / 3.0
    cs_mu = 0.1 # closed_small
    cs_al = -1.0
    cl_mu = 0.1 # closed_large
    cl_al = -1.0 
    cl_be = 0.8
    cl_ga = -0.4
    cl_de = 0.2
    ct_om = 1.0 # closed_trig
    ct_alpha = -0.8
    ct_bs1 = 0.7
    ct_bc1 = -0.5
    ct_bs2 = 0.4
    ct_bc2 = 0.2
    ct_bs3 = -0.25
    ct_bc3 = 0.15
    ct_bx = 0.3
    ct_bx2 = -0.08



    A_cs = {
        "saddle_point": np.array([
            [0.2, 0], 
            [0, -0.2]]),
        "degenerate_node": np.array([
            [-0.7, 0.7], 
            [0, -0.7]]),
        "inward_spiral": np.array([
            [-0.5, -2], 
            [2, -0.5]]),
        "harmonic_oscillator": np.array([
            [0, 1.3], 
            [-1.3, 0]]),

        "vanderpol": np.array([
            [0,1,0],
            [-1,vp_mu, -vp_mu],
            [0,0,0]]),
        "lotka_volterra": np.array([
            [lv_al,0,-lv_be],
            [0, -lv_ga, lv_de],
            [0,0,0]]),
        "pendulum": np.array([
            [0,1,0],
            [0,0,-pe_g/pe_l],
            [0,0,0]]),
        "duffing": np.array([
            [0,1,0],
            [-du_al, -du_de, -du_be],
            [0,0,0]]),
        "lorenz": np.array([
            [-lo_sigma, lo_sigma, 0,0,0],
            [lo_rho, -1, 0,-1,0],
            [0, 0, -lo_beta,0,1],
            [0,0,0,0,0],
            [0,0,0,0,0]]),

        "closed_small": np.array([
            [cs_mu, 0, 0],
            [0, cs_al, -cs_al],
            [0, 0, 2*cs_mu]]),
        "closed_large": np.array([
            [cl_mu, 0, 0, 0, 0],
            [0, cl_al, cl_be, cl_ga, cl_de],
            [0, 0, 2*cl_mu, 0, 0],
            [0, 0, 0, 3*cl_mu, 0],
            [0, 0, 0, 0, 4*cl_mu]]),
        "closed_trig_small": np.array([
            [0, 0, 0, 0, 0, 0], 
            [ct_om, 0, 0, 0, 0,0],
            [0, ct_bx, ct_alpha, ct_bx2, ct_bs1, ct_bc1],
            [0, 2*ct_om, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, ct_om], 
            [0, 0, 0, 0, -ct_om, 0]]),
        "closed_trig_medium": np.array([
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [ct_om, 0, 0, 0, 0, 0, 0, 0],
            [0, ct_bx, ct_alpha, ct_bx2, ct_bs1, ct_bc1, ct_bs2, ct_bc2],
            [0, 2*ct_om, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, ct_om, 0, 0], 
            [0, 0, 0, 0, -ct_om, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 2*ct_om], 
            [0, 0, 0, 0, 0, 0, -2*ct_om, 0]]),
        "closed_trig_large": np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [ct_om, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, ct_bx, ct_alpha, ct_bx2, ct_bs1, ct_bc1, ct_bs2, ct_bc2, ct_bs3, ct_bc3],
            [0, 2*ct_om, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, ct_om, 0, 0, 0, 0], 
            [0, 0, 0, 0, -ct_om, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 2*ct_om, 0, 0], 
            [0, 0, 0, 0, 0, 0, -2*ct_om, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3*ct_om], 
            [0, 0, 0, 0, 0, 0, 0, 0, -3*ct_om, 0]])
    }

    if system not in A_cs:
        raise ValueError(f"System '{system}' not found. Available systems: {list(A_cs.keys())}")

    # ==========================================================

    A_c = A_cs[system]

    # Compute A_d
    dt = 0.01
    A_d = expm(A_c * dt)
    eigvals, eigvecs = np.linalg.eig(A_d)

    if print_matrices:
        # compute A_d
        np.set_printoptions(precision=4, suppress=True) # print A_d with 4 decimal places
        print('='*80)
        print(f"\nSystem: {system}\n")
        print("A_d:")
        print(A_d)

        # eigen decomp on A_d
        np.set_printoptions(precision=4, suppress=True) # print A_d with 4 decimal places
        print("Lambda")
        print(eigvals)
        # print("Phi")
        # print(eigvecs)

        np.set_printoptions(precision=3, suppress=True)
        print('phi, real')
        print(np.real(eigvecs))
        print('phi, imag')
        print(np.imag(eigvecs))

    # Plot matrix
    if plot_phi:
        plt.title(f"Theoretical Phi for {system}")
        plt.imshow(abs(eigvecs), cmap='viridis')
        plt.colorbar()
        # write value in cells
        for i in range(eigvecs.shape[0]):
            for j in range(eigvecs.shape[1]):
                if abs(eigvecs[i,j]) > 1e-4: # only write values above a certain threshold for readability
                    if abs(np.imag(eigvecs[i,j])) > 1e-4 and abs(np.real(eigvecs[i,j])) > 1e-4:
                        plt.text(j, i, f"{np.real(eigvecs[i,j]):.3f}\n{np.imag(eigvecs[i,j]):.3f}j", ha='center', va='center', color='red')
                    elif abs(np.imag(eigvecs[i,j])) > 1e-4:
                        plt.text(j, i, f"\n{np.imag(eigvecs[i,j]):.3f}j", ha='center', va='center', color='red')
                    elif abs(np.real(eigvecs[i,j])) > 1e-4:
                        plt.text(j, i, f"{np.real(eigvecs[i,j]):.4f}\n", ha='center', va='center', color='red')
        plt.xlabel("Eigenvector Index")
        plt.ylabel("State Dimension")
        plt.show()

    return A_c, A_d, eigvals, eigvecs


if __name__ == "__main__":

    system = "closed_large"
    A_c, A_d, eigvals, eigvecs = get_system_matrices(system=system, print_matrices=False, plot_phi=True)