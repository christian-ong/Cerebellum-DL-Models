import argparse
import os
import numpy as np

from src.data_generation.data_simulation import (
    simulate,
    linear_system,
    vanderpol_system,
    lotka_volterra_system,
    pendulum_system,
    lorenz_system,
    duffing_system
)
from src.data_generation.plot_data import plot_init_conditions, plot_trajectories_only, plot_flow_map_displacement

"""
Defaults parameters:
    --system (inward_spiral | harmonic_oscillator | saddle_point | degenerate_node | vanderpol | lotka_volterra | pendulum | duffing | lorenz)
    --name (optional suffix for filename)
    --dt 0.01
    --T 20.0
    --method rk4
    --n_traj 1
    --val_frac 0.2
    --seed 0

System-specific parameters:
    Van der Pol:
        --mu 1.5
    Lotka--Volterra:
        --alpha_LV 1.1
        --beta_LV 0.4
        --delta_LV 0.1
        --gamma_LV 0.4
    Pendulum:
        --g 9.81
        --L 1.0
    Lorenz:
        --sigma 10.0
        --rho 28.0
        --beta_LZ 8/3
    Duffing:
        --alpha_DUF 1.0
        --beta_DUF 5.0
        --delta_DUF 0.2
        --gamma_DUF 1.0
        --omega_DUF 1.0

--------------------------------------------------
Linear system  x' = A x
--------------------------------------------------

python -m scripts.simulate_data --system inward_spiral --n_traj 100 --T 5
python -m scripts.simulate_data --system harmonic_oscillator --n_traj 100 --T 5
python -m scripts.simulate_data --system saddle_point --n_traj 100 --T 5
python -m scripts.simulate_data --system degenerate_node --n_traj 100 --T 5

--------------------------------------------------
Nonlinear systems
--------------------------------------------------

python -m scripts.simulate_data --system vanderpol --n_traj 100 --T 15
python -m scripts.simulate_data --system lotka_volterra --n_traj 100 --T 15
python -m scripts.simulate_data --system pendulum --n_traj 100 --T 15
python -m scripts.simulate_data --system duffing --n_traj 100 --T 15
python -m scripts.simulate_data --system lorenz --n_traj 100 --T 15

--------------------------------------------------
Output
--------------------------------------------------

Saved file:
    data/trajectories/{system}_trajectory[_<name>].npz

Contents:
    t         : (T_steps+1,)
    X         : (T_steps+1, state_dim) or (T_steps+1, n_traj, state_dim)
    train_idx : indices of training trajectories
    val_idx   : indices of validation trajectories
    dt        : time step used in simulation
    T         : total simulation time
    system     : name of the system
    n_traj     : number of trajectories
    seed       : random seed used in initial condition sampling
    ...        : system-specific parameters (e.g. mu for Van der Pol)
"""

# --------------------------------------------------
# Initial condition samplers
# --------------------------------------------------

# def sample_linear_ic(n_traj, rng):
#     theta = rng.uniform(0, 2 * np.pi, size=n_traj)
#     r = rng.uniform(0.5, 1.5, size=n_traj)
#     return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


# def sample_generic_ic(x0, n_traj, rng, noise_scale=0.1):
#     if n_traj == 1:
#         return x0
#     d = x0.shape[0]
#     noise = noise_scale * rng.standard_normal(size=(n_traj, d))
#     return x0[None, :] + noise


# def sample_uniform_ic(n_traj, rng, lows=np.array([-1.5, -1.5]), highs=np.array([1.5, 1.5])):
#     d = lows.shape[0]
#     x0s = np.zeros((n_traj, d), dtype=float)
#     for i in range(d):
#         x0s[:, i] = rng.uniform(lows[i], highs[i], size=n_traj)
#     return x0s

def sample_annulus_ic(n_traj, rng, r_min=0.2, r_max=1.5):
    """
    Uniform in annulus (area-uniform).
    Good for centered 2D systems.
    """
    theta = rng.uniform(0.0, 2*np.pi, size=n_traj)
    r2 = rng.uniform(r_min**2, r_max**2, size=n_traj)
    r = np.sqrt(r2)
    return np.stack([r*np.cos(theta), r*np.sin(theta)], axis=1)


def sample_box_ic(n_traj, rng, lows, highs):
    """
    Uniform in axis-aligned box.
    """
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float)

    d = lows.shape[0]
    x0s = np.zeros((n_traj, d), dtype=float)
    for i in range(d):
        x0s[:, i] = rng.uniform(lows[i], highs[i], size=n_traj)
    return x0s

def sample_elliptic_annulus_ic(n_traj, rng, *, a=1.0, b=1.0, r_min=0.2, r_max=1.0):
    """
    Elliptical annulus centered at origin.
    Points satisfy: (x/a)^2 + (y/b)^2 in [r_min^2, r_max^2]
    Area-uniform in the ellipse coordinates.
    """
    theta = rng.uniform(0.0, 2*np.pi, size=n_traj)
    r2 = rng.uniform(r_min**2, r_max**2, size=n_traj)
    r = np.sqrt(r2)

    x = a * r * np.cos(theta)
    y = b * r * np.sin(theta)
    return np.stack([x, y], axis=1)

def sample_ic(args, rng, *, kind, x0_single,
              lows=None, highs=None,
              r_min=None, r_max=None,
              a=None, b=None):

    if args.n_traj == 1:
        return np.asarray(x0_single, dtype=float)

    if kind == "annulus":
        return sample_annulus_ic(
            args.n_traj, rng,
            r_min=r_min, r_max=r_max
        )

    if kind == "elliptic_annulus":
        return sample_elliptic_annulus_ic(
            args.n_traj, rng,
            a=a, b=b,
            r_min=r_min, r_max=r_max
        )

    if kind == "box":
        return sample_box_ic(
            args.n_traj, rng,
            lows=lows, highs=highs
        )

    raise ValueError(f"Unknown sampling kind: {kind}")

def sample_lorenz_ic(n_traj, rng, rho, beta,
                     std=(2.0, 2.0, 2.0)):

    z0 = rho - 1.0
    r  = np.sqrt(beta * (rho - 1.0))

    c1 = np.array([ r,  r, z0])
    c2 = np.array([-r, -r, z0])

    n1 = n_traj // 2
    n2 = n_traj - n1

    std = np.asarray(std, dtype=float)

    X = np.zeros((n_traj, 3), dtype=float)
    X[:n1] = c1 + rng.standard_normal((n1, 3)) * std
    X[n1:] = c2 + rng.standard_normal((n2, 3)) * std

    rng.shuffle(X)
    return X

# --------------------------------------------------
# System builders
# --------------------------------------------------

def build_inward_spiral(args, rng):
    A = np.array([[-0.5, -2],
                  [ 2,  -0.5]], dtype=float)
    f = linear_system(A)

    x0 = sample_ic(args, rng,
                   kind="annulus",
                   r_min=0.2, r_max=1.5,
                   x0_single=[1.0, 0.0])

    return f, x0, {"A": A}


def build_harmonic_oscillator(args, rng):
    A = np.array([[0, 1.3],
                  [-1.3, 0]], dtype=float)
    f = linear_system(A)

    x0 = sample_ic(args, rng,
                   kind="annulus",
                   r_min=0.2, r_max=1.5,
                   x0_single=[1.0, 0.0])

    return f, x0, {"A": A}


def build_saddle_point(args, rng):
    A = np.array([[0.2, 0],
                  [0, -0.2]], dtype=float)
    f = linear_system(A)

    if args.n_traj == 1:
        x0 = np.array([1.0, 0.0], dtype=float)
    else:
        x0 = sample_elliptic_annulus_ic(
            args.n_traj, rng,
            a=0.5,   # compress x
            b=1.5,    # full y range
            r_min=0.1, r_max=1.1
        )

    return f, x0, {"A": A}


def build_degenerate_node(args, rng):
    A = np.array([[-0.7, 0.7],
                  [0, -0.7]], dtype=float)
    f = linear_system(A)

    x0 = sample_ic(args, rng,
                   kind="annulus",
                   r_min=0.2, r_max=1.5,
                   x0_single=[1.0, 0.0])

    return f, x0, {"A": A}


def build_vanderpol(args, rng):
    f = vanderpol_system(mu=args.mu)

    x0 = sample_ic(args, rng,
                   kind="annulus",
                   r_min=0.2, r_max=3.5,
                   x0_single=[1.0, 0.0])

    return f, x0, {"mu": args.mu}


def build_lotka_volterra(args, rng):
    f = lotka_volterra_system(
        alpha=args.alpha_LV,
        beta=args.beta_LV,
        delta=args.delta_LV,
        gamma=args.gamma_LV,
    )

    x_star = args.gamma_LV / args.delta_LV
    y_star = args.alpha_LV / args.beta_LV

    if args.n_traj == 1:
        x0 = np.array([x_star + 1.0, y_star], dtype=float)
    else:
        # sample annulus around origin
        pts = sample_annulus_ic(
            n_traj=args.n_traj,
            rng=rng,
            r_min=0.2,
            r_max=3.0
        )

        # shift to equilibrium
        x0 = pts + np.array([x_star, y_star])

        # ensure positivity
        x0 = np.maximum(x0, 0.3)

    return f, x0, {
        "alpha": args.alpha_LV,
        "beta": args.beta_LV,
        "delta": args.delta_LV,
        "gamma": args.gamma_LV,
    }


def build_pendulum(args, rng):
    f = pendulum_system(g=args.g, L=args.L)

    x0 = sample_ic(
        args, rng,
        kind="elliptic_annulus",
        a=2.8, b=3.5,
        r_min=0.1, r_max=1.0,
        x0_single=[0.5, 0.0]
    )

    return f, x0, {"g": args.g, "L": args.L}

def build_duffing(args, rng):
    f = duffing_system(
        alpha=args.alpha_DUF,
        beta=args.beta_DUF,
        delta=args.delta_DUF,
        gamma=args.gamma_DUF,
        omega=args.omega_DUF
    )

    if args.n_traj == 1:
        x0 = np.array([0.5, 0.0], dtype=float)
    else:
        # double-well centers (requires alpha < 0, beta > 0)
        x_eq = np.sqrt(-args.alpha_DUF / args.beta_DUF)

        n1 = args.n_traj // 2
        n2 = args.n_traj - n1

        blob1 = sample_elliptic_annulus_ic(
            n1, rng,
            a=0.6, b=1.2,
            r_min=0.0, r_max=1.0
        ) + np.array([+x_eq, 0.0])

        blob2 = sample_elliptic_annulus_ic(
            n2, rng,
            a=0.6, b=1.2,
            r_min=0.0, r_max=1.0
        ) + np.array([-x_eq, 0.0])

        x0 = np.vstack([blob1, blob2])
        rng.shuffle(x0)

    meta = {
        "alpha": args.alpha_DUF,
        "beta": args.beta_DUF,
        "delta": args.delta_DUF,
        "gamma": args.gamma_DUF,
        "omega": args.omega_DUF
    }
    return f, x0, meta


def build_lorenz(args, rng):
    f = lorenz_system(sigma=args.sigma, rho=args.rho, beta=args.beta_LZ)

    if args.n_traj == 1:
        x0 = np.array([1.0, 1.0, 25.0], dtype=float)
    else:
        x0 = sample_lorenz_ic(
            args.n_traj,
            rng,
            rho=args.rho,
            beta=args.beta_LZ,
            std=(5.0, 5.0, 5.0),
        )

    meta = {"sigma": args.sigma, "rho": args.rho, "beta": args.beta_LZ}
    return f, x0, meta


SYSTEMS = {
    # linear
    "linear": build_inward_spiral, # inward spiral by default
    "inward_spiral": build_inward_spiral,
    "harmonic_oscillator": build_harmonic_oscillator,
    "saddle_point": build_saddle_point,
    "degenerate_node": build_degenerate_node,
    
    # nonlinear
    "vanderpol": build_vanderpol,
    "lotka_volterra": build_lotka_volterra,
    "pendulum": build_pendulum,
    "lorenz": build_lorenz,
    "duffing": build_duffing
}

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Simulate dynamical systems")
    parser.add_argument("--debug", type=str, choices=["init_conditions", "phase_portrait"])

    parser.add_argument("--system", type=str, required=True, choices=SYSTEMS.keys())
    parser.add_argument("--name", type=str, default=None, help="Optional suffix added to the dataset filename")

    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--T", type=float, default=20.0)
    parser.add_argument("--method", type=str, default="rk4")
    parser.add_argument("--n_traj", type=int, default=1)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)

    # System-specific params
    # Van der Pol
    parser.add_argument("--mu", type=float, default=1.5)

    # Lokta--Volterra
    parser.add_argument("--alpha_LV", type=float, default=1.1)
    parser.add_argument("--beta_LV", type=float, default=0.4)
    parser.add_argument("--delta_LV", type=float, default=0.1)
    parser.add_argument("--gamma_LV", type=float, default=0.4)

    # Pendulum
    parser.add_argument("--g", type=float, default=9.81)
    parser.add_argument("--L", type=float, default=1.0)

    # Lorenz
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=28.0)
    parser.add_argument("--beta_LZ", type=float, default=2.67)

    # Duffing
    parser.add_argument("--alpha_DUF", type=float, default=-1.0)
    parser.add_argument("--beta_DUF", type=float, default=1.0)
    parser.add_argument("--delta_DUF", type=float, default=0.2)
    parser.add_argument("--gamma_DUF", type=float, default=0.3)
    parser.add_argument("--omega_DUF", type=float, default=1.2)

    parser.add_argument("--outdir", type=str, default="data/trajectories")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Build system
    f, x0, meta = SYSTEMS[args.system](args, rng)

    # Plot initial conditions and simulate trajectories for 4 corners
    if args.debug == "init_conditions":
        # 2D systems
        p1 = x0[np.argmax(+x0[:, 0] + x0[:, 1])]
        p2 = x0[np.argmax(+x0[:, 0] - x0[:, 1])]
        p3 = x0[np.argmax(-x0[:, 0] + x0[:, 1])]
        p4 = x0[np.argmax(-x0[:, 0] - x0[:, 1])]
        ps = np.array([p1, p2, p3, p4])

        # 3D systems (e.g. Lorenz)
        if x0.shape[1] == 3:
            p1 = x0[np.argmax(+ x0[:, 0] + x0[:, 1] + x0[:, 2])]
            p2 = x0[np.argmax(+ x0[:, 0] + x0[:, 1] - x0[:, 2])]
            p3 = x0[np.argmax(+ x0[:, 0] - x0[:, 1] + x0[:, 2])]
            p4 = x0[np.argmax(+ x0[:, 0] - x0[:, 1] - x0[:, 2])]
            p5 = x0[np.argmax(- x0[:, 0] + x0[:, 1] + x0[:, 2])]
            p6 = x0[np.argmax(- x0[:, 0] + x0[:, 1] - x0[:, 2])]
            p7 = x0[np.argmax(- x0[:, 0] - x0[:, 1] + x0[:, 2])]
            p8 = x0[np.argmax(- x0[:, 0] - x0[:, 1] - x0[:, 2])]
            ps = np.array([p1, p2, p3, p4, p5, p6, p7, p8])
        
        t, X = simulate(f, x0=ps, dt=args.dt, T=args.T, method=args.method)
        plot_init_conditions(x0s=x0, corner_points=ps, corner_trajs=X, system_name=args.system)
        return
    
    if args.debug == "phase_portrait":

        if x0.ndim == 1:
            x0s = x0[None, :]
        else:
            x0s = x0

        # Get state dimension
        state_dim = x0s.shape[1]

        plot_trajectories_only(
            f=f,
            x0s=x0s,
            dt=args.dt,
            T=args.T,
            system_name=args.system
        )
        if args.system in ["inward_spiral", "harmonic_oscillator", "saddle_point", "degenerate_node"]:

            plot_flow_map_displacement(
                f=f,
                state_dim=state_dim,
                system_name=args.system,
                grid_lim=1.5,
            )

        if args.system in ["vanderpol"]:
            plot_flow_map_displacement(
                f=f,
                state_dim=state_dim,
                system_name=args.system,
                grid_lim=5.0,
            )

        if args.system in ["lotka_volterra"]:
            plot_flow_map_displacement(
                f=f,
                state_dim=state_dim,
                system_name=args.system,
                xlim=(0, 27),
                ylim=(0, 11)
            )

        if args.system in ["pendulum"]:
            plot_flow_map_displacement(
                f=f,
                state_dim=state_dim,
                system_name=args.system,
                xlim=(-3, 3),
                ylim=(-6, 6)
            )
        
        if args.system in ["duffing"]:
            plot_flow_map_displacement(
                f=f,
                state_dim=state_dim,
                system_name=args.system,
                grid_lim=2.0,
            )
        
        if args.system in ["lorenz"]:
            plot_flow_map_displacement(
                f=f,
                state_dim=state_dim,
                system_name=args.system,
                xlim=(-25, 25),
                ylim=(-30, 30),
                zlim=(0, 50)
            )

        return
    
    t, X = simulate(f, x0=x0, dt=args.dt, T=args.T, method=args.method)

    # Train / validation split by trajectory
    if args.n_traj == 1:
        train_idx = np.array([0], dtype=int)
        val_idx   = np.array([], dtype=int)
    else:
        indices = np.arange(args.n_traj)
        rng.shuffle(indices)

        n_val = int(args.val_frac * args.n_traj)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

    # Save
    base = f"{args.system}_trajectory"
    if args.name is not None:
        filename = f"{base}_{args.name}.npz"
    else:
        filename = f"{base}.npz"
    outpath = os.path.join(args.outdir, filename)
    np.savez(
        outpath,
        t=t,
        X=X,
        x0=x0,
        train_idx=train_idx,
        val_idx=val_idx,
        dt=args.dt,
        T=args.T,
        system=args.system,
        n_traj=args.n_traj,
        seed=args.seed,
        **meta,
    )

    print(f"Saved trajectory to {outpath}")


if __name__ == "__main__":
    main()
