import numpy as np
from scipy.linalg import expm

def get_discrete_matrix(A_c, dt=0.01):
    """
    Converts a continuous-time system matrix A_c to a 
    discrete-time transition matrix A_d for a step size dt.
    """
    A_d = expm(A_c * dt)
    return A_d

# ==========================================================
# Construct the Continuous Matrix A_c

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
    "closed_trig": np.array([
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
# ==========================================================


# --- Compute A_d ---
for system, A_c in A_cs.items():
    # if system in ["saddle_point", "degenerate_node", "inward_spiral", "harmonic_oscillator"]:
    #     continue
    
    # compute A_d
    np.set_printoptions(precision=4, suppress=True) # print A_d with 4 decimal places
    A_d = get_discrete_matrix(A_c)
    print(f"System: {system}")
    print(A_d)

    # eigen decomp on A_d
    np.set_printoptions(precision=4, suppress=True) # print A_d with 4 decimal places
    eigvals, eigvecs = np.linalg.eig(A_d)
    print(eigvals)
    print(eigvecs)

    # np.set_printoptions(precision=3, suppress=True)
    # print('real')
    # print(np.real(eigvecs))
    # print('imag')
    # print(np.imag(eigvecs))

    print()
    print('-'*20)

