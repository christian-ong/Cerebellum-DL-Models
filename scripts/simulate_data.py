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
from src.data_generation.plot_data import plot_init_conditions

"""
Defaults parameters:
    --system (linear | vanderpol | lotka_volterra | pendulum | lorenz)
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
        --alpha 1.1
        --beta 0.4
        --delta 0.1
        --gamma 0.4
    Pendulum:
        --g 9.81
        --L 1.0
    Lorenz:
        --sigma 10.0
        --rho 28.0
        --beta 8/3
    Duffing:
        --alpha 1.0
        --beta 5.0
        --delta 0.02
        --gamma 8.0
        --omega 0.5

--------------------------------------------------
Linear system  x' = A x
--------------------------------------------------

Single trajectory:
python -m scripts.simulate_data --system linear

Multiple trajectories:
python -m scripts.simulate_data --system linear --n_traj 100

--------------------------------------------------
Van der Pol oscillator
--------------------------------------------------

python -m scripts.simulate_data --system vanderpol --n_traj 100

--------------------------------------------------
Lotka--Volterra predator--prey
--------------------------------------------------

python -m scripts.simulate_data --system lotka_volterra --n_traj 100

--------------------------------------------------
Pendulum
--------------------------------------------------

python -m scripts.simulate_data --system pendulum --n_traj 100

--------------------------------------------------
Lorenz system
--------------------------------------------------

python -m scripts.simulate_data --system lorenz --n_traj 100 --beta 2.666666666667

--------------------------------------------------
Duffing oscillator
--------------------------------------------------

python -m scripts.simulate_data --system duffing --n_traj 100 --alpha 1.0 --beta 5.0 --delta 0.02 --gamma 8.0 --omega 0.5

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

def sample_linear_ic(n_traj, rng):
    theta = rng.uniform(0, 2 * np.pi, size=n_traj)
    r = rng.uniform(0.5, 1.5, size=n_traj)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def sample_generic_ic(x0, n_traj, rng, noise_scale=0.1):
    if n_traj == 1:
        return x0
    d = x0.shape[0]
    noise = noise_scale * rng.standard_normal(size=(n_traj, d))
    return x0[None, :] + noise


def sample_uniform_ic(lows, highs, n_traj, rng):
    d = lows.shape[0]
    x0s = np.zeros((n_traj, d), dtype=float)
    for i in range(d):
        x0s[:, i] = rng.uniform(lows[i], highs[i], size=n_traj)
    return x0s


# --------------------------------------------------
# System builders
# --------------------------------------------------

def build_inward_spiral(args, rng):
    A = np.array([[-0.3, -6],
                  [ 6,  -0.3]], dtype=float)
    f = linear_system(A)
    if args.n_traj > 1:
        x0 = sample_uniform_ic(
            lows=np.array([-1.5, -1.5]), 
            highs=np.array([1.5, 1.5]), 
            n_traj=args.n_traj, 
            rng=rng)
    else:
        x0 = np.array([1.0, 0.0], dtype=float)
    meta = {"A": A}
    return f, x0, meta


def build_harmonic_oscillator(args, rng):
    A = np.array([[0, 1],
                  [-1,  0]], dtype=float)
    f = linear_system(A)
    if args.n_traj > 1:
        x0 = sample_uniform_ic(
            lows=np.array([-1.5, -1.5]),
            highs=np.array([1.5, 1.5]),
            n_traj=args.n_traj,
            rng=rng)
    else:
        x0 = np.array([1.0, 0.0], dtype=float)
    meta = {"A": A}
    return f, x0, meta


def build_saddle_point(args, rng):
    A = np.array([[0.1, 0],
                  [0, -0.1]], dtype=float)
    f = linear_system(A)
    if args.n_traj > 1:
        x0 = sample_uniform_ic(
            lows=np.array([-2.5, -7.5]),
            highs=np.array([2.5, 7.5]),
            n_traj=args.n_traj,
            rng=rng)
    else:
        x0 = np.array([1.0, 0.0], dtype=float)
    meta = {"A": A}
    return f, x0, meta


def build_degenerate_node(args, rng):
    A = np.array([[-1, 1],
                  [0, -1]], dtype=float)
    f = linear_system(A)
    if args.n_traj > 1:
        x0 = sample_uniform_ic(
            lows=np.array([-1.5, -1.5]),
            highs=np.array([1.5, 1.5]),
            n_traj=args.n_traj,
            rng=rng)
    else:
        x0 = np.array([1.0, 0.0], dtype=float)
    meta = {"A": A}
    return f, x0, meta


def build_vanderpol(args, rng):
    f = vanderpol_system(mu=args.mu)
    x0 = sample_uniform_ic(
        lows=np.array([-4.0, -4.0]),
        highs=np.array([4.0, 4.0]),
        n_traj=args.n_traj,
        rng=rng)
    meta = {"mu": args.mu}
    return f, x0, meta


def build_lotka_volterra(args, rng):
    f = lotka_volterra_system(
        alpha=args.alpha,
        beta=args.beta,
        delta=args.delta,
        gamma=args.gamma,
    )
    x0 = sample_uniform_ic(
        lows=np.array([0.0, 0.0]),
        highs=np.array([20.0, 20.0]),
        n_traj=args.n_traj,
        rng=rng)
    meta = {
        "alpha": args.alpha,
        "beta": args.beta,
        "delta": args.delta,
        "gamma": args.gamma,
    }
    return f, x0, meta


def build_pendulum(args, rng):
    f = pendulum_system(g=args.g, L=args.L)
    x0 = sample_uniform_ic(
        lows=np.array([-1, -2]),
        highs=np.array([1, 2]),
        n_traj=args.n_traj,
        rng=rng)
    meta = {"g": args.g, "L": args.L}
    return f, x0, meta


def build_lorenz(args, rng):
    f = lorenz_system(
        sigma=args.sigma,
        rho=args.rho,
        beta=args.beta,
    )
    x0 = sample_uniform_ic(
        lows=np.array([-20.0, -20.0, 0.0]),
        highs=np.array([20.0, 20.0, 50.0]),
        n_traj=args.n_traj,
        rng=rng)
    meta = {
        "sigma": args.sigma,
        "rho": args.rho,
        "beta": args.beta,
    }
    return f, x0, meta

def build_duffing(args, rng):
    f = duffing_system(
        alpha=args.alpha,
        beta=args.beta,
        delta=args.delta,
        gamma=args.gamma,
        omega=args.omega
    )
    x0_base = np.array([1.0, 0.0], dtype=float)
    x0 = sample_generic_ic(x0_base, args.n_traj, rng)

    meta = {
        "alpha": args.alpha,
        "beta": args.beta,
        "delta": args.delta,
        "gamma": args.gamma,
        "omega": args.omega
    }
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
    parser.add_argument("--debug", type=str, choices=["init_conditions"])

    parser.add_argument("--system", type=str, required=True, choices=SYSTEMS.keys())
    parser.add_argument("--name", type=str, default=None, help="Optional suffix added to the dataset filename")

    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--T", type=float, default=20.0)
    parser.add_argument("--method", type=str, default="rk4")
    parser.add_argument("--n_traj", type=int, default=1)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)

    # System-specific params
    parser.add_argument("--mu", type=float, default=1.5)

    parser.add_argument("--alpha", type=float, default=1.1)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.4)

    parser.add_argument("--g", type=float, default=9.81)
    parser.add_argument("--L", type=float, default=1.0)

    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=28.0)

    parser.add_argument("--omega", type=float, default=0.5)

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
