import numpy as np
import matplotlib.pyplot as plt
from src.data_generation.data_simulation import simulate, vanderpol_system, lotka_volterra_system, pendulum_system, lorenz_system


# --- Simulation and Visualization ---
plt.figure(figsize=(12, 8))

# Simulate Van der Pol
t, X_vdp = simulate(vanderpol_system(), x0=[0.5, 0.0], dt=0.01, T=20)
t, X_vdp_2 = simulate(vanderpol_system(), x0=[0.6, 0.1], dt=0.01, T=20)  # Different initial condition
plt.subplot(2, 2, 1)
plt.plot(X_vdp[:, 0], X_vdp[:, 1], color='blue', lw=0.5, label='Initial Condition 1')
plt.plot(X_vdp_2[:, 0], X_vdp_2[:, 1], color='red', lw=0.5, label='Initial Condition 2')
# plot starting points
plt.scatter(X_vdp[0, 0], X_vdp[0, 1], color='blue', s=20, label='Start IC1')
plt.scatter(X_vdp_2[0, 0], X_vdp_2[0, 1], color='red', s=20, label='Start IC2')
plt.title("Van der Pol Oscillator (Phase Portrait)")
plt.xlabel("x1"); plt.ylabel("x2")
plt.legend()

# Simulate Predator-Prey
t, X_lv = simulate(lotka_volterra_system(), x0=[10, 5], dt=0.01, T=100)
t, X_lv_2 = simulate(lotka_volterra_system(), x0=[11, 5], dt=0.01, T=100)  # Different initial condition
plt.subplot(2, 2, 2)
plt.plot(t, X_lv[:, 0], lw=0.5, color='red', label='Prey (IC1)')
plt.plot(t, X_lv[:, 1], lw=0.5, color='tomato', label='Predator (IC1)')
plt.plot(t, X_lv_2[:, 0], lw=0.5, color='navy', label='Prey (IC2)')
plt.plot(t, X_lv_2[:, 1], lw=0.5, color='midnightblue', label='Predator (IC2)')
plt.title("Lotka-Volterra (Population vs Time)")
plt.legend()

# Simulate Pendulum
t, X_pen = simulate(pendulum_system(), x0=[np.pi/2, 0], dt=0.01, T=10)
t, X_pen_2 = simulate(pendulum_system(), x0=[np.pi/2 + 0.1, 0], dt=0.01, T=10)  # Slightly different initial angle
plt.subplot(2, 2, 3)
plt.plot(t, X_pen[:, 0], lw=0.5, label='Initial Condition 1')
plt.plot(t, X_pen_2[:, 0], lw=0.5, label='Initial Condition 2')
plt.title("Nonlinear Pendulum (Angle vs Time)")
plt.ylabel("Theta (rad)")
plt.xlabel("Time (s)")
plt.legend()

# Simulate Lorenz (3D State)
t, X_lor = simulate(lorenz_system(), x0=[1.0, 1.0, 1.0], dt=0.01, T=20)
t, X_lor_2 = simulate(lorenz_system(), x0=[1.0, 1.0, 1.1], dt=0.01, T=20)  # Slightly different initial condition
plt.subplot(2, 2, 4)
plt.plot(X_lor[:, 0], X_lor[:, 2], color='red', lw=0.5, label='Initial Condition 1')
plt.plot(X_lor_2[:, 0], X_lor_2[:, 2], color='blue', lw=0.5, label='Initial Condition 2')
plt.scatter(X_lor[0, 0], X_lor[0, 2], color='red', s=20, marker='^', label='Start IC1')
plt.scatter(X_lor_2[0, 0], X_lor_2[0, 2], color='blue', s=20, marker='v', label='Start IC2')
plt.xlabel("X"); plt.ylabel("Z")
plt.title("Lorenz Attractor (X-Z Projection)")
plt.legend()

plt.tight_layout()
plt.show()