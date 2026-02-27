import numpy as np
import matplotlib.pyplot as plt
from src.data_generation.data_simulation import simulate, rk4_step
import os

# =========== PARAMETERS ==============
plot_4_systems = True
plot_euler_vs_rk = False
plot_different_dts = False
# ====================================

def plot_init_conditions(x0s, corner_points, corner_trajs, system_name="system"):
    colours = ['red', 'blue', 'green', 'orange',
               'purple', 'cyan', 'magenta', 'yellow']
    dir = "data/phase_portraits/initial_condition_trajs_4"
    dim_names = ["x", "y", "z"]

    point_size = 20

    plt.figure(figsize=(6, 6))    

    # Plot all 3 dimensions for Lorenz
    if system_name == "lorenz":
        for dim1, dim2 in [(0, 1), (0, 2), (1, 2)]:
            # dummy points for legend
            plt.scatter(x0s[0, dim1], x0s[0, dim2], color='black', s=point_size, label=f'Initial conditions (100 samples)')
            plt.plot([], [], color='black', lw=1, label=f'Trajectories (8)')

            plt.scatter(x0s[:, dim1], x0s[:, dim2], color='gray', s=point_size)#, label=f'Initial conditions')
            for i in range(len(corner_points)):
                plt.scatter(corner_points[i][dim1], corner_points[i][dim2], color=colours[i], s=20)#, label=f'Corner {i+1}')
                plt.plot(corner_trajs[:, i, dim1], corner_trajs[:, i, dim2], color=colours[i], lw=1, alpha=0.7)#, label=f'Trajectory {i+1}')
            plt.xlabel(f"{dim_names[dim1]}"); plt.ylabel(f"{dim_names[dim2]}")
            # plt.title(f"Initial conditions and simulated trajectories, \n{system_name} system ({dim_names[dim1]}, {dim_names[dim2]})")
            plt.title(f"{system_name.replace('_', ' ')} system ({dim_names[dim1]} vs {dim_names[dim2]})", fontsize=15)
            plt.legend(loc='lower right')
            plt.grid()
            plt.axis("equal")
            plt.tight_layout()
            os.makedirs(dir, exist_ok=True)
            plt.savefig(f"{dir}/{system_name}({dim1}_{dim2}).png")
            plt.close()

        return

    
    # dummy points for legend
    plt.scatter(x0s[0, 0], x0s[0, 1], color='black', s=point_size, label=f'Initial conditions (100 samples)')
    plt.plot([], [], color='black', lw=1, label=f'Trajectories (8)')
    plt.scatter(x0s[:, 0], x0s[:, 1], color='gray', s=point_size)#, label=f'Initial conditions')
    for i in range(len(corner_points)):
        plt.scatter(corner_points[i][0], corner_points[i][1], color=colours[i], s=20)#, label=f'Corner {i+1}')
        plt.plot(corner_trajs[:, i, 0], corner_trajs[:, i, 1], color=colours[i], lw=1, alpha=0.7)#, label=f'Trajectory {i+1}')
    
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(f"{system_name.replace('_', ' ')} system (x vs y)", fontsize=15)

    plt.legend(loc='lower right', fontsize=12)
    plt.grid()
    plt.axis("equal")
    plt.tight_layout()
    os.makedirs(dir, exist_ok=True)
    plt.savefig(f"{dir}/{system_name}.png")
    plt.close()


def plot_trajectories_only(f, x0s, dt, T,
                           system_name="system",
                           max_trajs_to_plot=100,
                           outdir="data/phase_portraits/clean"):

    os.makedirs(outdir, exist_ok=True)

    t, X = simulate(f, x0=x0s, dt=dt, T=T, method="rk4")

    if X.shape[1] > max_trajs_to_plot:
        idx = np.linspace(0, X.shape[1]-1,
                          max_trajs_to_plot, dtype=int)
        X = X[:, idx, :]
        x0s = x0s[idx]

    plt.figure(figsize=(7,7))

    # Use matplotlib default color cycle
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i in range(X.shape[1]):
        color = cmap[i % len(cmap)]

        # trajectory
        plt.plot(X[:, i, 0],
                 X[:, i, 1],
                 lw=1.5,
                 color=color)

        # matching initial condition
        plt.scatter(x0s[i, 0],
                    x0s[i, 1],
                    color=color,
                    s=35,
                    edgecolor='black',
                    linewidth=0.5,
                    zorder=3)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{system_name.replace('_',' ')} system — trajectories")
    # plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{outdir}/{system_name}_trajectories.png")
    plt.close()

def plot_flow_map_displacement(f,
                               grid_lim=1.6,
                               grid_n=25,
                               tau=0.12,
                               system_name="system",
                               outdir="data/phase_portraits/clean"):

    os.makedirs(outdir, exist_ok=True)

    x1 = np.linspace(-grid_lim, grid_lim, grid_n)
    x2 = np.linspace(-grid_lim, grid_lim, grid_n)
    X1, X2 = np.meshgrid(x1, x2)

    pts = np.column_stack([X1.ravel(), X2.ravel()])

    Phi = np.array([rk4_step(f, p, tau) for p in pts])
    D = Phi - pts

    DX = D[:, 0].reshape(X1.shape)
    DY = D[:, 1].reshape(X2.shape)

    # ---- AUTO SCALE ----
    magnitudes = np.sqrt(DX**2 + DY**2)
    max_mag = np.max(magnitudes)

    # Avoid divide by zero
    if max_mag > 0:
        scale_factor = max_mag
    else:
        scale_factor = 1.0

    DX = DX / scale_factor
    DY = DY / scale_factor
    # --------------------

    plt.figure(figsize=(7,7))

    plt.quiver(X1, X2, DX, DY,
               angles='xy',
               scale_units='xy',
               scale=2,
               alpha=0.9,
               color='tab:blue',
               pivot='mid')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{system_name.replace('_',' ')} system — flow map displacement")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{outdir}/{system_name}_flowmap.png")
    plt.close()