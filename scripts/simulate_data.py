import numpy as np
import matplotlib.pyplot as plt
from src.data_generation.data_simulation import simulate, linear_system
import os

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