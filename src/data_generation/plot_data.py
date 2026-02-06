import numpy as np
import matplotlib.pyplot as plt
from data_simulation import simulate, linear_system
import os

plot_4_systems = True
plot_euler_vs_rk = False
plot_different_dts = False

if __name__ == "__main__" and plot_4_systems:

    # Setup directories
    os.makedirs("data/phase_portraits", exist_ok=True)

    # Parameters
    dt = 1e-3
    T  = 5.0
    n_points = int(np.round(T/dt + 1))

    # 1) Harmonic oscillator
    w = 1
    A = np.array([[0, 1],
                  [-w**2, 0]])
    ts_harmonic, Xs_harmonic = [], []
    x0s_harmonic = [
        [1.0, 0.0], 
        [2.0, 0.0]
    ]
    for x0 in x0s_harmonic:
        t, X = simulate(linear_system(A), x0=x0, dt=dt, T=T, method="rk4")
        ts_harmonic.append(t)
        Xs_harmonic.append(X)

    # # 2) Inward spiral (stable focus)
    # a, w = 0.3, 2*np.pi
    # A = np.array([[-a, -w],
    #               [w, -a]])
    # ts_inward, Xs_inward = [], []
    # x0s_inward = [
    #     [1.0, 0.0],
    #     [-1.0, 0.0]
    # ]
    # for x0 in x0s_inward:
    #     t, X = simulate(linear_system(A), x0=x0, dt=dt, T=T, method="rk4")
    #     ts_inward.append(t)
    #     Xs_inward.append(X)

    # # 3) Hyperbolic saddle point
    # A = np.array([[1, 0],
    #               [0, -1]])
    # ts_saddle, Xs_saddle = [], []
    # x0s_saddle = [
    #     [1.0, 0.0], 
    #     [1.0, 1.0],
    #     [0.0, 1.0],
    #     [-1.0, 1.0], 
    #     [-1.0, 0.0], 
    #     [-1.0, -1.0], 
    #     [0.0, -1.0], 
    #     [1.0, -1.0], 
    # ]
    # for x0 in x0s_saddle:
    #     t, X = simulate(linear_system(A), x0=x0, dt=dt, T=T, method="rk4")
    #     ts_saddle.append(t)
    #     Xs_saddle.append(X)

    # # 4) Degenerate node
    # A = np.array([[-1, 1],
    #               [0, -1]])
    # ts_degenerate, Xs_degenerate = [], []
    # x0s_degenerate = [
    #     [1.0, 0.0], 
    #     [1.0, 1.0],
    #     [0.0, 1.0],
    #     [-1.0, 1.0], 
    #     [-1.0, 0.0], 
    #     [-1.0, -1.0], 
    #     [0.0, -1.0], 
    #     [1.0, -1.0], 
    # ]
    # for x0 in x0s_degenerate:
    #     t, X = simulate(linear_system(A), x0=x0, dt=dt, T=T, method="rk4")
    #     ts_degenerate.append(t)
    #     Xs_degenerate.append(X)

    
    # Phase portraits
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    
    # colors
    color_reds = plt.cm.Reds(np.linspace(0.2, 1, n_points))
    color_blues = plt.cm.Blues(np.linspace(0.2, 1, n_points))

    # Harmonic oscillator
    for i in range(n_points - 1):
        for j in range(len(Xs_harmonic)):
            ax[0,0].plot(Xs_harmonic[j][i:i+2, 0], Xs_harmonic[j][i:i+2, 1], color=plt.cm.Greys((n_points-i)/n_points), linewidth=1, alpha=0.3)
    for i in range(len(Xs_harmonic)):
        ax[0,0].scatter(Xs_harmonic[i][0, 0], Xs_harmonic[i][0, 1], color=plt.cm.tab10(i), s=50, marker='o', label=f'start {i+1}', zorder=5)
    ax[0,0].set_title("Harmonic oscillator (phase space)")
    ax[0,0].set_xlabel("q"); ax[0,0].set_ylabel("p")
    ax[0,0].axis("equal")
    ax[0,0].legend()
    
    # # Inward spiral
    # for i in range(n_points - 1):
    #     for j in range(len(Xs_inward)):
    #         ax[0,1].plot(Xs_inward[j][i:i+2, 0], Xs_inward[j][i:i+2, 1], color=plt.cm.Greys((n_points-i)/n_points), linewidth=1, alpha=0.3)
    
    # for i in range(len(Xs_inward)):
    #     ax[0,1].scatter(Xs_inward[i][0, 0], Xs_inward[i][0, 1], color=plt.cm.tab10(i), s=50, marker='o', label=f'start {i+1}', zorder=5)
    # # ax[0,1].scatter(Xs_inward[1][0, 0], Xs_inward[1][0, 1], color='blue', s=50, marker='o', label='start 2', zorder=5)
    # # ax[0,1].scatter(Y_e[-1, 0], Y_e[-1, 1], color='blue', s=50, marker='o', label='end (Euler)', zorder=5)
    # ax[0,1].set_title("Inward spiral / stable focus")
    # ax[0,1].set_xlabel("x"); ax[0,1].set_ylabel("y")
    # ax[0,1].axis("equal")
    # ax[0,1].legend()

    # # Hyperbolic saddle point
    # A = np.array([[1, 0],
    #               [0, -1]])
    # t3, Z = simulate(linear_system(A), x0=[1.0, 0.5], dt=dt, T=T, method="rk4")
    # # t3_e, Z_e = simulate(linear_system(A), x0=x0, dt=dt, T=T, method="euler")
    # for i in range(n_points - 1):
    #     for j in range(len(Xs_saddle)):
    #         ax[1,0].plot(Xs_saddle[j][i:i+2, 0], Xs_saddle[j][i:i+2, 1], color=plt.cm.Greys((n_points-i)/n_points), linewidth=1, alpha=0.3)
    
    # for i in range(len(Xs_saddle)):
    #     ax[1,0].scatter(Xs_saddle[i][0, 0], Xs_saddle[i][0, 1], color=plt.cm.tab10(i), s=50, marker='o', label=f'start {i+1}', zorder=5)
    # # ax[1,0].scatter(Z_2[0, 0], Z_2[0, 1], color='blue', s=50, marker='o', label='start 2', zorder=5)
    # # ax[1,0].scatter(Z_e[-1, 0], Z_e[-1, 1], color='blue', s=50, marker='o', label='end (Euler)', zorder=5)
    # ax[1,0].set_title("Hyperbolic saddle point")
    # ax[1,0].set_xlabel("x"); ax[1,0].set_ylabel("y")
    # ax[1,0].axis("equal")
    # ax[1,0].set_xlim(-2, 2); ax[1,0].set_ylim(-2, 2)
    # ax[1,0].legend()

    # # Degenerate node
    # A = np.array([[-1, 1],
    #               [0, -1]])
    # t4, W = simulate(linear_system(A), x0=[1.0, 0.0], dt=dt, T=T, method="rk4")
    # # t4_e, W_e = simulate(linear_system(A), x0=x0, dt=dt, T=T, method="euler")
    # for i in range(n_points - 1):
    #     for j in range(len(Xs_degenerate)):
    #         ax[1,1].plot(Xs_degenerate[j][i:i+2, 0], Xs_degenerate[j][i:i+2, 1], color=plt.cm.Greys((n_points-i)/n_points), linewidth=1., alpha=0.3)
    # for i in range(len(Xs_degenerate)):
    #     ax[1,1].scatter(Xs_degenerate[i][0, 0], Xs_degenerate[i][0, 1], color=plt.cm.tab10(i), s=50., marker='o', label=f'start {i+1}', zorder=5)
    # # ax[1,1].scatter(Xs_degenerate[1][0, 0], Xs_degenerate[1][0, 1], color='blue', s=50., marker='o', label='start 2', zorder=5)
    # # ax[1,1].scatter(W_e[-1 , 0 ], W_e[-1 , 1 ], color='blue', s=50., marker='o', label='end (Euler)', zorder=5)
    # ax[1,1].set_title("Degenerate node")
    # ax[1,1].set_xlabel("x"); ax[1,1].set_ylabel("y")
    # ax[1,1].axis("equal")
    # ax[1,1].legend()


    plt.tight_layout()

    # save plot
    plt.savefig("data/phase_portraits/portrait_4_systems_test.png")


if __name__ == "__main__" and plot_euler_vs_rk:
    dt = 1e-3
    T  = 5.0

    # 2) Inward spiral (stable focus)
    a, w = 0.3, 2*np.pi
    A = np.array([[-a, -w],
                  [w, -a]])
    x0=[1.0, 0.0]
    t, X = simulate(linear_system(A), x0=x0, dt=dt, T=T, method="rk4")
    t_e, X_e = simulate(linear_system(A), x0=x0, dt=dt, T=T, method="euler")

    # Plot trajectories over time
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    n_points = len(X)
    color_reds = plt.cm.Reds(np.linspace(0.2, 1, n_points))
    color_blues = plt.cm.Blues(np.linspace(0.2, 1, n_points))
    for i in range(n_points - 1):
        ax.plot(X[i:i+2, 0], X[i:i+2, 1], color=color_reds[-i], linewidth=1, alpha=0.3)
        ax.plot(X_e[i:i+2, 0], X_e[i:i+2, 1], color=color_blues[-i], linewidth=1, alpha=0.3)
    ax.scatter(X[0, 0], X[0, 1], color='black', s=50, marker='o', label='start (both)', zorder=5)
    ax.scatter(X[-1, 0], X[-1, 1], color='red', s=50, marker='o', label='end (RK4)', zorder=5)
    ax.scatter(X_e[-1, 0], X_e[-1, 1], color='blue', s=50, marker='o', label='end (Euler)', zorder=5)
    ax.set_title("Inward spiral / stable focus")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    
    plt.tight_layout()
    
    # save plot
    plt.savefig("data/phase_portraits/euler_vs_rk4.png")


if __name__ == "__main__" and plot_different_dts:
    dt_values = [1e-2, 1e-3, 1e-4]
    T = 5.0

    # Harmonic oscillator
    w = 2*np.pi
    A = np.array([[0, 1],
                  [-w**2, 0]])
    x0=[1.0, 0.0]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    Xs = []
    for i, dt in enumerate(dt_values):
        t, X = simulate(linear_system(A), x0=x0, dt=dt, T=T, method="rk4")
        # ax[i//2, i%2].plot(X[:, 0], X[:, 1], color='red', linewidth=1)
        # ax[i//2, i%2].scatter(X[0, 0], X[0, 1], color='black', s=50, marker='o', label='start', zorder=5)
        # ax[i//2, i%2].scatter(X[-1, 0], X[-1, 1], color='red', s=50, marker='o', label='end', zorder=5)
        # ax[i//2, i%2].set_title(f"dt = {dt:.0e}")
        # ax[i//2, i%2].set_xlabel("q"); ax[i//2, i%2].set_ylabel("p")
        # ax[i//2, i%2].axis("equal")
        # ax[i//2, i%2].legend()
        Xs.append(X)

    # Plot all on the same axes
    colors = plt.cm.viridis(np.linspace(0, 1, len(dt_values)))
    for i, (dt, X) in enumerate(zip(dt_values, Xs)):
        ax.plot(X[:, 0], X[:, 1], color=colors[i], linewidth=1, label=f'dt={dt:.0e}')
    ax.scatter(Xs[0][0, 0], Xs[0][0, 1], color='black', s=50, marker='o', label='start', zorder=5)
    ax.set_title("Harmonic oscillator (phase space)")
    ax.set_xlabel("q"); ax.set_ylabel("p")
    ax.axis("equal")
    ax.legend()
    
    plt.tight_layout()
    
    # save plot
    plt.savefig("data/phase_portraits/different_dts.png")