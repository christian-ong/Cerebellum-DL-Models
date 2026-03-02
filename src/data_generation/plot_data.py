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


def plot_trajectories_only(
    f,
    x0s,
    dt,
    T,
    system_name="system",
    max_trajs_to_plot=100,
    outdir="data/figures/trajectories/",
):

    os.makedirs(outdir, exist_ok=True)

    t, X = simulate(f, x0=x0s, dt=dt, T=T, method="rk4")

    if X.ndim == 2:
        X = X[:, None, :]
        x0s = np.asarray(x0s)[None, :]
    else:
        x0s = np.asarray(x0s)

    n_traj = X.shape[1]
    if n_traj > max_trajs_to_plot:
        idx = np.linspace(0, n_traj - 1, max_trajs_to_plot, dtype=int)
        X = X[:, idx, :]
        x0s = x0s[idx]
        n_traj = max_trajs_to_plot

    state_dim = X.shape[2]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # ============================================================
    # 2D SYSTEM
    # ============================================================
    if state_dim == 2:

        plt.figure(figsize=(7, 7))

        for i in range(n_traj):
            color = colors[i % len(colors)]
            plt.plot(X[:, i, 0], X[:, i, 1], lw=1.2, color=color)
            plt.scatter(
                x0s[i, 0], x0s[i, 1],
                color=color,
                s=35,
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{system_name.replace('_',' ').title()} — RK4 simulated trajectories", fontsize=15)
        # plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/{system_name}_trajectories.png", dpi=300)
        plt.close()

    # ============================================================
    # 3D SYSTEM → 3 PROJECTIONS
    # ============================================================
    elif state_dim == 3:

        projections = [
            (0, 1, "x", "y"),
            (0, 2, "x", "z"),
            (1, 2, "y", "z"),
        ]

        for dim1, dim2, label1, label2 in projections:

            plt.figure(figsize=(7, 7))

            for i in range(n_traj):
                color = colors[i % len(colors)]

                plt.plot(
                    X[:, i, dim1],
                    X[:, i, dim2],
                    lw=1.0,
                    color=color,
                )

                plt.scatter(
                    x0s[i, dim1],
                    x0s[i, dim2],
                    color=color,
                    s=35,
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                )

            plt.xlabel(label1)
            plt.ylabel(label2)
            plt.title(
                f"{system_name.replace('_',' ').title()} — RK4 simulated trajectories {label1}-{label2}", fontsize=15
            )

            # plt.axis("equal")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                f"{outdir}/{system_name}_{label1}{label2}.png",
                dpi=300,
            )
            plt.close()


def plot_flow_map_displacement(
    f,
    state_dim,
    grid_lim=20,
    grid_n=25,
    tau=0.01,
    xlim=None,
    ylim=None,
    zlim=None,
    system_name="system",
    outdir="data/figures/flowmaps/",
):

    os.makedirs(outdir, exist_ok=True)

    # --- Grid limits ---
    if xlim is None:
        xlim = (-grid_lim, grid_lim)
    if ylim is None:
        ylim = (-grid_lim, grid_lim)

    x1 = np.linspace(xlim[0], xlim[1], grid_n)
    x2 = np.linspace(ylim[0], ylim[1], grid_n)
    X1, X2 = np.meshgrid(x1, x2)

    pts2d = np.column_stack([X1.ravel(), X2.ravel()])

    # ============================================================
    # 2D SYSTEM
    # ============================================================
    if state_dim == 2:

        Phi = np.array([rk4_step(f, p, tau) for p in pts2d])
        D = Phi - pts2d

        DX = D[:, 0].reshape(X1.shape)
        DY = D[:, 1].reshape(X2.shape)

        speed = np.sqrt(DX**2 + DY**2)

        plt.figure(figsize=(8, 7))
        plt.streamplot(X1, X2, DX, DY, density=1.1, color=speed, cmap="viridis")
        plt.colorbar(label="$|\dot{x}|$")
        plt.xlabel("x")
        plt.ylabel("y")
        # plt.axis("equal")
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        plt.grid(True, alpha=0.3)
        plt.title(f"{system_name.replace('_',' ').title()} — Phase Portrait", fontsize=15)
        plt.tight_layout()
        plt.savefig(f"{outdir}/{system_name}_flowmap.png", dpi=300)
        plt.close()

    # ============================================================
    # 3D SYSTEM → 3 PROJECTION SLICES
    # ============================================================
    elif state_dim == 3:
            # Reference centers for Lorenz "Wings"
            z_ref, y_ref, x_ref = 27.0, 0.0, 0.0
            
            projections = [
                (0, 1, 2, "x", "y", xlim, ylim, z_ref),
                (0, 2, 1, "x", "z", xlim, zlim, y_ref),
                (1, 2, 0, "y", "z", ylim, zlim, x_ref),
            ]

            # --- Part A: The 2D Projections ---
            for dim1, dim2, fixed_dim, label1, label2, lim1, lim2, ref_val in projections:
                x1 = np.linspace(lim1[0], lim1[1], grid_n)
                x2 = np.linspace(lim2[0], lim2[1], grid_n)
                X1, X2 = np.meshgrid(x1, x2)

                pts3d = np.zeros((X1.size, 3))
                pts3d[:, dim1] = X1.ravel()
                pts3d[:, dim2] = X2.ravel()
                pts3d[:, fixed_dim] = ref_val 

                F = np.array([f(0, p) for p in pts3d])
                DX = F[:, dim1].reshape(X1.shape)
                DY = F[:, dim2].reshape(X2.shape)
                speed = np.sqrt(DX**2 + DY**2)

                plt.figure(figsize=(8, 7))
                strm = plt.streamplot(X1, X2, DX, DY, density=1.0, color=speed, cmap="magma")
                plt.colorbar(label="|F(x)|")
                plt.title(f"{system_name.replace('_',' ').title()}: {label1}-{label2} slice at {ref_val}")
                plt.xlabel(label1); plt.ylabel(label2)
                plt.tight_layout()
                plt.savefig(f"{outdir}/{system_name}_slice_{label1}{label2}.png", dpi=300)
                plt.close()

            # --- Part B: The Hero 3D Plot ---
            grid_3d = 8 
            x_3d = np.linspace(xlim[0], xlim[1], grid_3d)
            y_3d = np.linspace(ylim[0], ylim[1], grid_3d)
            z_3d = np.linspace(zlim[0], zlim[1], grid_3d)
            X, Y, Z = np.meshgrid(x_3d, y_3d, z_3d)

            u, v, w = np.zeros(X.shape), np.zeros(Y.shape), np.zeros(Z.shape)
            for i in range(grid_3d):
                for j in range(grid_3d):
                    for k in range(grid_3d):
                        vel = f(0, np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]]))
                        u[i,j,k], v[i,j,k], w[i,j,k] = vel

            speed_3d = np.sqrt(u**2 + v**2 + w**2 + 1e-6)
            un, vn, wn = u/speed_3d, v/speed_3d, w/speed_3d
            
            # Prepare color data
            C = speed_3d.flatten()

            # Adjust figsize to (8, 7) to match 2D plots more closely, 
            # though 3D usually needs a bit more width for the colorbar
            fig = plt.figure(figsize=(8, 7))
            # Use a slightly smaller shrink and pad for the colorbar
            ax = fig.add_subplot(111, projection='3d')

            q = ax.quiver(X, Y, Z, un, vn, wn, 
                            length=3.0, 
                            cmap='viridis', 
                            array=C, 
                            alpha=0.4, 
                            linewidth=1.5)

            # Use 'fraction' and 'pad' to keep the colorbar tight to the plot
            # shrink=0.5 helps it not look "longer" than the 3D box height
            fig.colorbar(q, ax=ax, label="|F(x)|", shrink=0.5, pad=0.05, fraction=0.046)

            # set labels
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

            # Manually adjust the subplots to remove the excess 3D padding
            plt.subplots_adjust(left=0, right=0.85, top=0.9, bottom=0)

            ax.set_title(f"{system_name.replace('_',' ').title()} — 3D Phase Portrait", fontsize=15)
            ax.view_init(elev=20, azim=45)
            ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False 
            
            plt.savefig(f"{outdir}/{system_name}_3D_Hero.png", dpi=300, bbox_inches='tight')
            plt.close()