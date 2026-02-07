import numpy as np
import matplotlib.pyplot as plt
from src.data_generation.data_simulation import simulate, linear_system
from src.data_generation.data_simulation import simulate, vanderpol_system, lotka_volterra_system, pendulum_system, lorenz_system
import os

# ==================== PARAMETERS =======================
simulate_linear = False

# Nonlinear
simulate_vanderpol = False
vdp_params = {
    "x0": [0.5, 0.0],
    "dt": 0.01,
    "T": 20.0,
    "mu": 1.5}

simulate_lotka_volterra = False
lv_params = {
    "x0": [10, 5],
    "dt": 0.01,
    "T": 100.0,
    "alpha": 1.1, 
    "beta": 0.4, 
    "delta": 0.1, 
    "gamma": 0.4}

simulate_pendulum = False
pendulum_params = {
    "x0": [np.pi/2, 0],
    "dt": 0.01,
    "T": 10.0,
    "g": 9.81, 
    "L": 1.0}

simulate_lorenz = False
lorenz_params = {
    "x0": [1.0, 1.0, 1.0],
    "dt": 0.01,
    "T": 20.0,
    "sigma": 10.0, 
    "rho": 28.0, 
    "beta": 8/3}
# ======================================================

os.makedirs("data/trajectories", exist_ok=True)

print("hey")

# Parameters
dt = 1e-3
T = 20.0
A = np.array([[-0.3, -6],
                [6, -0.3]])
x0 = np.array([1.0, 0.0])

# Simulate trajectories
n_points = int(T / dt) + 1
t, X = simulate(linear_system(A), x0=x0, dt=dt, T=T, method="rk4")

# Save trajectories
os.makedirs("data/trajectories", exist_ok=True)
np.savez("data/trajectories/simulated_trajectories2.npz", t=t, X=X)

# TODO: save labels (true system dynamics)

# How to use:
# data = np.load("data/trajectories/simulated_trajectories.npz")
# t = data["t"] # (20001, 2)
# X = data["X"] # (20001,)
# plt.plot(X[:, 0], X[:, 1])


# -----------------------
# Nonlinear Systems
# -----------------------

# Van der Pol
if simulate_vanderpol:
    t, X_vdp = simulate(
        vanderpol_system(
            mu=vdp_params["mu"]), 
        x0=vdp_params["x0"], 
        dt=vdp_params["dt"], 
        T=vdp_params["T"])
    np.savez(
        "data/trajectories/vanderpol_trajectory.npz", 
        t=t, 
        X=X_vdp, 
        mu=vdp_params["mu"], 
        x0=vdp_params["x0"], 
        dt=vdp_params["dt"], 
        T=vdp_params["T"])

# Predator-Prey
if simulate_lotka_volterra:
    t, X_lv = simulate(
        lotka_volterra_system(
            alpha=lv_params["alpha"], 
            beta=lv_params["beta"], 
            delta=lv_params["delta"], 
            gamma=lv_params["gamma"]), 
        x0=lv_params["x0"], 
        dt=lv_params["dt"], 
        T=lv_params["T"])
    np.savez(
        "data/trajectories/lotka_volterra_trajectory.npz", 
        t=t, 
        X=X_lv, 
        alpha=lv_params["alpha"], 
        beta=lv_params["beta"], 
        delta=lv_params["delta"], 
        gamma=lv_params["gamma"], 
        x0=lv_params["x0"], 
        dt=lv_params["dt"], 
        T=lv_params["T"])

# Pendulum
if simulate_pendulum:
    t, X_pen = simulate(
        pendulum_system(
            g=pendulum_params["g"], 
            L=pendulum_params["L"]), 
        x0=pendulum_params["x0"], 
        dt=pendulum_params["dt"], 
        T=pendulum_params["T"])
    np.savez(
        "data/trajectories/pendulum_trajectory.npz", 
        t=t, 
        X=X_pen, 
        g=pendulum_params["g"], 
        L=pendulum_params["L"], 
        x0=pendulum_params["x0"], 
        dt=pendulum_params["dt"], 
        T=pendulum_params["T"])

# Lorenz (3D State)
if simulate_lorenz:
    t, X_lor = simulate(
        lorenz_system(
            sigma=lorenz_params["sigma"], 
            rho=lorenz_params["rho"], 
            beta=lorenz_params["beta"]), 
        x0=lorenz_params["x0"], 
        dt=lorenz_params["dt"], 
        T=lorenz_params["T"])
    np.savez(
        "data/trajectories/lorenz_trajectory.npz",
        t=t,
        X=X_lor,
        sigma=lorenz_params["sigma"],
        rho=lorenz_params["rho"],
        beta=lorenz_params["beta"],
        x0=lorenz_params["x0"],
        dt=lorenz_params["dt"],
        T=lorenz_params["T"])
    