import argparse
import math
import os
import numpy as np

DEFAULT_DT = 1e-2
TRAIN_T = 2.0
EVAL_T = 2.0
TEST_T = 5.0
DEFAULT_TARGET_TRAIN_WINDOWS = 80_000
DEFAULT_MAX_ROLLOUT_HORIZON = 100

from src.data_generation.data_simulation import (
    simulate,
    linear_system,
    vanderpol_system,
    lotka_volterra_system,
    pendulum_system,
    lorenz_system,
    duffing_system,
    closed_small_system,
    closed_large_system,
    closed_trig_small_system,
    closed_trig_medium_system,
    closed_trig_large_system,
    
)
from src.data_generation.plot_data import (
    plot_init_conditions,
    plot_flow_map_displacement,
    plot_trajectories_from_array,
)

"""
Defaults parameters:
    --system (inward_spiral | harmonic_oscillator | saddle_point | degenerate_node | vanderpol | lotka_volterra | pendulum | duffing | lorenz)
    --name (optional suffix for filename)
    --dt 0.01
    --T_train 3.0
    --T_val 3.0
    --T_test 3.0
    --method rk4
    --n_traj_train 1
    --n_traj_val 1
    --n_traj_test 1
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

python -m scripts.simulate_data --system inward_spiral --target_train_windows 80000 --max_rollout_horizon 100
python -m scripts.simulate_data --system harmonic_oscillator --target_train_windows 80000 --max_rollout_horizon 100
python -m scripts.simulate_data --system saddle_point --target_train_windows 80000 --max_rollout_horizon 100
python -m scripts.simulate_data --system degenerate_node --target_train_windows 80000 --max_rollout_horizon 100

Nonlinear systems (SHORT TRAJECTORIES)
--------------------------------------------------

python -m scripts.simulate_data --system closed_small --target_train_windows 80000 --max_rollout_horizon 100
python -m scripts.simulate_data --system closed_large --target_train_windows 80000 --max_rollout_horizon 100
python -m scripts.simulate_data --system closed_trig_small --target_train_windows 80000 --max_rollout_horizon 100
python -m scripts.simulate_data --system closed_trig_medium --target_train_windows 80000 --max_rollout_horizon 100
python -m scripts.simulate_data --system closed_trig_large --target_train_windows 80000 --max_rollout_horizon 100

python -m scripts.simulate_data --system vanderpol --target_train_windows 80000 --max_rollout_horizon 100
python -m scripts.simulate_data --system lotka_volterra --target_train_windows 80000 --max_rollout_horizon 100
python -m scripts.simulate_data --system pendulum --target_train_windows 80000 --max_rollout_horizon 100
python -m scripts.simulate_data --system duffing --target_train_windows 80000 --max_rollout_horizon 100
python -m scripts.simulate_data --system lorenz --target_train_windows 80000 --max_rollout_horizon 100

--------------------------------------------------
Output
--------------------------------------------------

Saved files:
    data/trajectories/{linear|nonlinear}/{system}/{split}.npz
    data/trajectories/{linear|nonlinear}/{system}/{name}/{split}.npz   (if --name is provided)

Where split is one of: train, val, test

Each .npz file contains:
    t         : (T_steps+1,)
    X         : (T_steps+1, state_dim) or (T_steps+1, n_traj, state_dim)
    x0        : initial conditions for this split
    dt        : time step used in simulation
    T         : total simulation time for this split
    system     : name of the system
    n_traj     : number of trajectories for this split
    seed       : random seed used in initial condition sampling
    ...        : system-specific parameters (e.g. mu for Van der Pol)
"""

# --------------------------------------------------
# Initial condition samplers
# --------------------------------------------------

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


def resolve_n_traj(args, default=1):
    """Resolve active trajectory count without relying on legacy args.n_traj."""
    if hasattr(args, "n_traj_current"):
        return int(args.n_traj_current)
    return int(default)


def set_simulation_vars(args):
    args.dt = DEFAULT_DT
    data_points = int(TRAIN_T / DEFAULT_DT)
    args.T_train = (data_points + args.max_rollout_horizon) * args.dt

    # Validation always uses the short usable horizon.
    val_data_points = int(EVAL_T / DEFAULT_DT)
    args.T_val = (val_data_points + args.max_rollout_horizon) * args.dt

    # Test uses the same short usable horizon.
    test_data_points = int(TEST_T / DEFAULT_DT)
    args.T_test = (test_data_points + args.max_rollout_horizon) * args.dt

    # Default counts for val/test if not explicitly provided by user
    if args.n_traj_val == 1:
        args.n_traj_val = int(math.ceil(10000.0 / float(val_data_points)))
    if args.n_traj_test == 1:
        args.n_traj_test = int(math.ceil(10000.0 / float(test_data_points)))

    # Compute needed training trajectories based on the target windows
    if args.target_train_windows is not None:
        if args.target_train_windows <= 0:
            raise ValueError("target_train_windows must be positive")

        usable_per_traj = data_points
        args.n_traj_train = int(math.ceil(args.target_train_windows / float(usable_per_traj)))

        if args.n_traj_train < 1:
            args.n_traj_train = 1

def sample_ic(args, rng, *, kind, x0_single,
              lows=None, highs=None,
              r_min=None, r_max=None,
              a=None, b=None):

    n_traj = resolve_n_traj(args)

    if n_traj == 1:
        return np.asarray(x0_single, dtype=float)

    if kind == "annulus":
        return sample_annulus_ic(
            n_traj, rng,
            r_min=r_min, r_max=r_max
        )

    if kind == "elliptic_annulus":
        return sample_elliptic_annulus_ic(
            n_traj, rng,
            a=a, b=b,
            r_min=r_min, r_max=r_max
        )

    if kind == "box":
        return sample_box_ic(
            n_traj, rng,
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
                   r_min=0.0, r_max=1.5,
                   x0_single=[1.0, 0.0])

    return f, x0, {"A": A}


def build_inward_spiral_cw(args, rng):
    """ Inward spiral clockwise direction """
    A = np.array([[-0.5, 2],
                  [-2,  -0.5]], dtype=float)
    f = linear_system(A)

    x0 = sample_ic(args, rng,
                   kind="annulus",
                   r_min=0.0, r_max=1.5,
                   x0_single=[1.0, 0.0])

    return f, x0, {"A": A}


def build_harmonic_oscillator(args, rng):
    A = np.array([[0, 1.3],
                  [-1.3, 0]], dtype=float)
    f = linear_system(A)

    x0 = sample_ic(args, rng,
                   kind="annulus",
                   r_min=0.0, r_max=1.5,
                   x0_single=[1.0, 0.0])

    return f, x0, {"A": A}


def build_saddle_point(args, rng):
    A = np.array([[0.2, 0],
                  [0, -0.2]], dtype=float)
    f = linear_system(A)

    n_traj = resolve_n_traj(args)

    if n_traj == 1:
        x0 = np.array([1.0, 0.0], dtype=float)
    else:
        x0 = sample_elliptic_annulus_ic(
            n_traj, rng,
            a=0.5,   # compress x
            b=1.5,    # full y range
            r_min=0.0, r_max=1.1
        )

    return f, x0, {"A": A}


def build_degenerate_node(args, rng):
    A = np.array([[-0.7, 0.7],
                  [0, -0.7]], dtype=float)
    f = linear_system(A)

    x0 = sample_ic(args, rng,
                   kind="annulus",
                   r_min=0.0, r_max=1.5,
                   x0_single=[1.0, 0.0])

    return f, x0, {"A": A}


def build_vanderpol(args, rng):
    f = vanderpol_system(mu=args.mu)

    x0 = sample_ic(args, rng,
                   kind="annulus",
                   r_min=0.0, r_max=3.5,
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

    n_traj = resolve_n_traj(args)

    if n_traj == 1:
        x0 = np.array([x_star + 1.0, y_star], dtype=float)
    else:
        # sample elliptic annulus around origin
        pts = sample_elliptic_annulus_ic(
                n_traj, rng,
                a=1.5,   # compress x
                b=0.5,    # full y range
                r_min=0.0, r_max=3.5
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
        r_min=0.0, r_max=1.0,
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

    n_traj = resolve_n_traj(args)

    if n_traj == 1:
        x0 = np.array([0.5, 0.0], dtype=float)
    else:
        x_eq = np.sqrt(-args.alpha_DUF / args.beta_DUF)

        n_center = n_traj // 3
        n_side = (n_traj - n_center) // 2
        n_right = n_side
        n_left = n_traj - n_center - n_right

        blob_center = sample_elliptic_annulus_ic(
            n_center, rng,
            a=0.7, b=1.0,
            r_min=0.0, r_max=0.7
        ) + np.array([0.0, 0.0])

        blob_right = sample_elliptic_annulus_ic(
            n_right, rng,
            a=0.5, b=0.95,
            r_min=0.0, r_max=1.0
        ) + np.array([+x_eq, 0.0])

        blob_left = sample_elliptic_annulus_ic(
            n_left, rng,
            a=0.5, b=0.95,
            r_min=0.0, r_max=1.0
        ) + np.array([-x_eq, 0.0])

        x0 = np.vstack([blob_left, blob_right, blob_center])
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

    n_traj = resolve_n_traj(args)

    if n_traj == 1:
        x0 = np.array([1.0, 1.0, 25.0], dtype=float)
    else:
        x0 = sample_lorenz_ic(
            n_traj,
            rng,
            rho=args.rho,
            beta=args.beta_LZ,
            std=(5.0, 5.0, 5.0),
        )

    meta = {"sigma": args.sigma, "rho": args.rho, "beta": args.beta_LZ}
    return f, x0, meta

def build_closed_small(args, rng):
    f = closed_small_system(mu=args.mu_KP, alpha=args.alpha_KP)

    n_traj = resolve_n_traj(args)

    if n_traj == 1:
        x0 = np.array([0.8, 0.4], dtype=float)
    else:
        x_init = rng.uniform(-1.0, 1.0, size=n_traj)
        y_init = rng.uniform(-1.0, 1.5, size=n_traj)
        x0 = np.stack([x_init, y_init], axis=1)

    A_lift = np.array([
        [args.mu_KP, 0.0, 0.0],
        [0.0, args.alpha_KP, -args.alpha_KP],
        [0.0, 0.0, 2.0 * args.mu_KP],
    ], dtype=float)

    return f, x0, {
        "mu": args.mu_KP,
        "alpha": args.alpha_KP,
        "A_lift": A_lift,
    }

def build_closed_large(args, rng):
    f = closed_large_system(
        mu=args.mu_K234,
        alpha=args.alpha_K234,
        beta=args.beta_K234,
        gamma=args.gamma_K234,
        delta=args.delta_K234,
    )

    n_traj = resolve_n_traj(args)

    if n_traj == 1:
        x0 = np.array([0.8, 0.1], dtype=float)
    else:
        # Keep x moderate so x^2, x^3, x^4 stay well-behaved
        x_init = rng.uniform(-1.0, 1.0, size=n_traj)
        y_init = rng.uniform(-1.0, 1.0, size=n_traj)
        x0 = np.stack([x_init, y_init], axis=1)

    A_lift = np.array([
        [args.mu_K234,      0.0,             0.0,              0.0,              0.0],
        [0.0,               args.alpha_K234, args.beta_K234,   args.gamma_K234,  args.delta_K234],
        [0.0,               0.0,             2.0*args.mu_K234, 0.0,              0.0],
        [0.0,               0.0,             0.0,              3.0*args.mu_K234, 0.0],
        [0.0,               0.0,             0.0,              0.0,              4.0*args.mu_K234],
    ], dtype=float)

    return f, x0, {
        "mu": args.mu_K234,
        "alpha": args.alpha_K234,
        "beta": args.beta_K234,
        "gamma": args.gamma_K234,
        "delta": args.delta_K234,
        "A_lift": A_lift,
    }

def build_closed_trig_small(args, rng):
    f = closed_trig_small_system(
        omega=args.omega_KPT,
        alpha=args.alpha_KPT,
        beta_s1=args.beta_s1_KPT,
        beta_c1=args.beta_c1_KPT,
        beta_x=args.beta_x_KPT,
        beta_x2=args.beta_x2_KPT,
    )

    n_traj = resolve_n_traj(args)

    if n_traj == 1:
        x0 = np.array([0.8, 0.4], dtype=float)
    else:
        x_init = rng.uniform(-1.0, 1.0, size=n_traj)
        y_init = rng.uniform(-1.0, 1.5, size=n_traj)
        x0 = np.stack([x_init, y_init], axis=1)

    return f, x0, {
        "omega": args.omega_KPT,
        "alpha": args.alpha_KPT,
        "beta_s1": args.beta_s1_KPT,
        "beta_c1": args.beta_c1_KPT,
        "beta_x": args.beta_x_KPT,
        "beta_x2": args.beta_x2_KPT,
    }

def build_closed_trig_medium(args, rng):
    f = closed_trig_medium_system(
        omega=args.omega_KPT,
        alpha=args.alpha_KPT,
        beta_s1=args.beta_s1_KPT,
        beta_c1=args.beta_c1_KPT,
        beta_s2=args.beta_s2_KPT,
        beta_c2=args.beta_c2_KPT,
        beta_x=args.beta_x_KPT,
        beta_x2=args.beta_x2_KPT,
    )

    n_traj = resolve_n_traj(args)

    if n_traj == 1:
        x0 = np.array([0.8, 0.4], dtype=float)
    else:
        x_init = rng.uniform(-1.0, 1.0, size=n_traj)
        y_init = rng.uniform(-1.0, 1.5, size=n_traj)
        x0 = np.stack([x_init, y_init], axis=1)

    return f, x0, {
        "omega": args.omega_KPT,
        "alpha": args.alpha_KPT,
        "beta_s1": args.beta_s1_KPT,
        "beta_c1": args.beta_c1_KPT,
        "beta_s2": args.beta_s2_KPT,
        "beta_c2": args.beta_c2_KPT,
        "beta_x": args.beta_x_KPT,
        "beta_x2": args.beta_x2_KPT,
    }


def build_closed_trig_large(args, rng):
    f = closed_trig_large_system(
        omega=args.omega_KPT,
        alpha=args.alpha_KPT,
        beta_s1=args.beta_s1_KPT,
        beta_c1=args.beta_c1_KPT,
        beta_s2=args.beta_s2_KPT,
        beta_c2=args.beta_c2_KPT,
        beta_s3=args.beta_s3_KPT,
        beta_c3=args.beta_c3_KPT,
        beta_x=args.beta_x_KPT,
        beta_x2=args.beta_x2_KPT,
    )

    n_traj = resolve_n_traj(args)

    if n_traj == 1:
        x0 = np.array([0.0, 0.0], dtype=float)
    else:
        # Keep x in a moderate interval initially; since x' = omega,
        # x will drift linearly in time.
        x_init = rng.uniform(-2.0, 2.0, size=n_traj)
        y_init = rng.uniform(-1.0, 1.0, size=n_traj)
        x0 = np.stack([x_init, y_init], axis=1)

    return f, x0, {
        "omega": args.omega_KPT,
        "alpha": args.alpha_KPT,
        "beta_s1": args.beta_s1_KPT,
        "beta_c1": args.beta_c1_KPT,
        "beta_s2": args.beta_s2_KPT,
        "beta_c2": args.beta_c2_KPT,
        "beta_s3": args.beta_s3_KPT,
        "beta_c3": args.beta_c3_KPT,
        "beta_x": args.beta_x_KPT,
        "beta_x2": args.beta_x2_KPT,
    }


SYSTEMS = {
    # linear
    "linear": build_inward_spiral, # inward spiral by default
    "inward_spiral": build_inward_spiral,
    "inward_spiral_cw": build_inward_spiral_cw,
    "harmonic_oscillator": build_harmonic_oscillator,
    "saddle_point": build_saddle_point,
    "degenerate_node": build_degenerate_node,
    
    # nonlinear
    "vanderpol": build_vanderpol,
    "lotka_volterra": build_lotka_volterra,
    "pendulum": build_pendulum,
    "lorenz": build_lorenz,
    "duffing": build_duffing,
    "closed_small": build_closed_small,
    "closed_large": build_closed_large,
    "closed_trig_small": build_closed_trig_small,
    "closed_trig_medium": build_closed_trig_medium,
    "closed_trig_large": build_closed_trig_large
}

LINEAR_SYSTEMS = {
    "linear",
    "inward_spiral",
    "inward_spiral_cw",
    "harmonic_oscillator",
    "saddle_point",
    "degenerate_node",
}
# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Simulate dynamical systems")
    parser.add_argument("--debug", type=str, choices=["init_conditions", "phase_portrait"])

    parser.add_argument("--system", type=str, required=True, choices=SYSTEMS.keys())
    parser.add_argument("--name", type=str, default=None, help="Optional suffix added to the dataset filename")

    parser.add_argument("--target_train_windows", type=int, default=None, help="Exact number of training initial conditions/pairs to generate. When set, training time is forced to (max_rollout_horizon + 1) * dt so each initial condition has one next-state pair plus the full future rollout.")
    parser.add_argument("--max_rollout_horizon", type=int, default=DEFAULT_MAX_ROLLOUT_HORIZON, help="Maximum rollout horizon to reserve in the generated data.")

    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--T_train", type=float, default=3.0)
    parser.add_argument("--T_val", type=float, default=5.0)
    parser.add_argument("--T_test", type=float, default=20.0)
    parser.add_argument("--method", type=str, default="rk4")
    parser.add_argument("--n_traj_train", type=int, default=1)
    parser.add_argument("--n_traj_val", type=int, default=1)
    parser.add_argument("--n_traj_test", type=int, default=1)
    parser.add_argument("--n_debug_traj", type=int, default=100)
    parser.add_argument("--plot_splits", action="store_true", help="Save train/val/test trajectory plots under data/figures/trajectories/<system>")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--burn_in",type=float,default=0.0,help="Extra simulation time to discard before saving each split.")

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
    parser.add_argument("--gamma_DUF", type=float, default=0.0)
    parser.add_argument("--omega_DUF", type=float, default=0.0)

    # koopman poly simple nonlinear system (section 2.5.2 overleaf)
    parser.add_argument("--mu_KP", type=float, default=0.1)
    parser.add_argument("--alpha_KP", type=float, default=-1.0)
    
    # Koopman polynomial LARGE (x^2 + x^3 + x^4 test system, written in docs)
    parser.add_argument("--mu_K234", type=float, default=0.1)
    parser.add_argument("--alpha_K234", type=float, default=-1.0)
    parser.add_argument("--beta_K234", type=float, default=0.8)
    parser.add_argument("--gamma_K234", type=float, default=-0.4)
    parser.add_argument("--delta_K234", type=float, default=0.2)

    # Koopman polynomial + trigonometric test system (last level closed form system)
    parser.add_argument("--omega_KPT", type=float, default=1.0)
    parser.add_argument("--alpha_KPT", type=float, default=-0.8)
    parser.add_argument("--beta_s1_KPT", type=float, default=0.7)
    parser.add_argument("--beta_c1_KPT", type=float, default=-0.5)
    parser.add_argument("--beta_s2_KPT", type=float, default=0.4)
    parser.add_argument("--beta_c2_KPT", type=float, default=0.2)
    parser.add_argument("--beta_s3_KPT", type=float, default=-0.25)
    parser.add_argument("--beta_c3_KPT", type=float, default=0.15)
    parser.add_argument("--beta_x_KPT", type=float, default=0.3)
    parser.add_argument("--beta_x2_KPT", type=float, default=-0.08)

    parser.add_argument("--outdir", type=str, default="data/trajectories")

    args = parser.parse_args()

    set_simulation_vars(args)

    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Build system
    f, _, meta = SYSTEMS[args.system](args, rng)

    # Plot initial conditions and simulate trajectories for 4 corners
    if args.debug == "init_conditions":
        args.n_traj_current = args.n_debug_traj
        _, x0, _ = SYSTEMS[args.system](args, rng)
        
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
        
        t, X = simulate(f, x0=ps, dt=args.dt, T=args.T_test, method=args.method)
        plot_init_conditions(x0s=x0, corner_points=ps, corner_trajs=X, system_name=args.system)
        return
    
    if args.debug == "phase_portrait":
        debug_outdir = os.path.join("data", "figures", args.system, "phase_portrait")
        os.makedirs(debug_outdir, exist_ok=True)

        debug_splits = [
            ("train", "train", args.n_traj_train, args.T_train),
            ("val", "eval", args.n_traj_val, args.T_val),
            ("test", "test", args.n_traj_test, args.T_test),
        ]

        state_dim = None
        for split_name, plot_label, n_traj, T_split in debug_splits:
            if n_traj <= 0:
                print(f"Skipping {split_name} debug plot (n_traj={n_traj})")
                continue

            temp_args = argparse.Namespace(**vars(args))
            temp_args.n_traj_current = n_traj
            _, x0_split, _ = SYSTEMS[args.system](temp_args, rng)
            _, X_split = simulate(f, x0=x0_split, dt=args.dt, T=T_split, method=args.method)

            if state_dim is None:
                state_dim = X_split.shape[-1]

            plot_trajectories_from_array(
                X=X_split,
                x0s=x0_split,
                system_name=args.system,
                max_trajs_to_plot=None,
                outdir=debug_outdir,
                split_name=plot_label,
            )
            print(f"Saved {plot_label} trajectories plot to {debug_outdir}")

        if state_dim is None:
            raise ValueError("No debug plots generated. Set at least one of n_traj_train/n_traj_val/n_traj_test > 0.")

        if args.system in ["inward_spiral", "harmonic_oscillator", "saddle_point", "degenerate_node"]:

            plot_flow_map_displacement(
                f=f,
                state_dim=state_dim,
                system_name=args.system,
                grid_lim=1.5,
                outdir=debug_outdir,
            )

        if args.system in ["vanderpol"]:
            plot_flow_map_displacement(
                f=f,
                state_dim=state_dim,
                system_name=args.system,
                grid_lim=5.0,
                outdir=debug_outdir,
            )

        if args.system in ["lotka_volterra"]:
            plot_flow_map_displacement(
                f=f,
                state_dim=state_dim,
                system_name=args.system,
                xlim=(0, 27),
                ylim=(0, 11),
                outdir=debug_outdir,
            )

        if args.system in ["pendulum"]:
            plot_flow_map_displacement(
                f=f,
                state_dim=state_dim,
                system_name=args.system,
                xlim=(-3, 3),
                ylim=(-6, 6),
                outdir=debug_outdir,
            )
        
        if args.system in ["duffing"]:
            plot_flow_map_displacement(
                f=f,
                state_dim=state_dim,
                system_name=args.system,
                grid_lim=2.0,
                outdir=debug_outdir,
            )
        
        if args.system in ["lorenz"]:
            plot_flow_map_displacement(
                f=f,
                state_dim=state_dim,
                system_name=args.system,
                xlim=(-25, 25),
                ylim=(-30, 30),
                zlim=(0, 50),
                outdir=debug_outdir,
            )
        if args.system in ["closed_small", "closed_large"]:
            plot_flow_map_displacement(
                f=f,
                state_dim=state_dim,
                system_name=args.system,
                xlim=(-1.5, 1.5),
                ylim=(-1.5, 1.5),
                outdir=debug_outdir,
            )


        if args.system in ["closed_trig_small", "closed_trig_medium", "closed_trig_large"]:
            plot_flow_map_displacement(
                f=f,
                state_dim=state_dim,
                system_name=args.system,
                xlim=(-2.5, 2.5),
                ylim=(-2.0, 2.0),
                outdir=debug_outdir,
            )    
        print(f"Saved vector field plot(s) to {debug_outdir}")
        return
    
    # Simulate and save train, val, test separately
    splits = [
        ("train", args.n_traj_train, args.T_train),
        ("val", args.n_traj_val, args.T_val),
        ("test", args.n_traj_test, args.T_test),
    ]
    
    for split_name, n_traj, T in splits:
        if n_traj == 0:
            print(f"Skipping {split_name} split (n_traj=0)")
            continue
        
        # Create a temporary args object with the correct n_traj for the system builder
        temp_args = argparse.Namespace(**vars(args))
        temp_args.n_traj_current = n_traj
        
        # Sample initial conditions for this split
        _, x0_split, _ = SYSTEMS[args.system](temp_args, rng)
        
        # Simulate extra time if burn-in is requested, then discard it before saving.
        T_total = T + args.burn_in
        t_full, X_full = simulate(f, x0=x0_split, dt=args.dt, T=T_total, method=args.method)

        burn_steps = int(round(args.burn_in / args.dt))

        if burn_steps > 0:
            if burn_steps >= X_full.shape[0] - 2:
                raise ValueError(
                    f"burn_in={args.burn_in} removes too much data for split={split_name}. "
                    f"burn_steps={burn_steps}, trajectory length={X_full.shape[0]}."
                )

            X_split = X_full[burn_steps:]
            t_split = t_full[burn_steps:] - t_full[burn_steps]
        else:
            X_split = X_full
            t_split = t_full

        x0_saved = np.asarray(X_split[0]).copy()
        T_saved = float(t_split[-1])
        
        # Save using clean dataset directory layout with linear/nonlinear categorization.
        category = "linear" if args.system in LINEAR_SYSTEMS else "nonlinear"
        save_dir = os.path.join(args.outdir, category, args.system)
        if args.name is not None:
            save_dir = os.path.join(save_dir, args.name)
        os.makedirs(save_dir, exist_ok=True)

        outpath = os.path.join(save_dir, f"{split_name}.npz")

        np.savez(
            outpath,
            t=t_split,
            X=X_split,
            x0=x0_saved,
            x0_before_burn_in=x0_split,
            dt=args.dt,
            T=T_saved,
            burn_in=args.burn_in,
            burn_steps=burn_steps,
            system=args.system,
            n_traj=n_traj,
            seed=args.seed,
            **meta,
        )

        if args.plot_splits:
            plot_dir = os.path.join("data", "figures", "trajectories", args.system)
            plot_trajectories_from_array(
                X=X_split,
                x0s=x0_split,
                system_name=args.system,
                max_trajs_to_plot=100,
                outdir=plot_dir,
                split_name=split_name,
            )
            print(f"Saved {split_name} plot to {plot_dir}")

        print(f"Saved {split_name} trajectory to {outpath}")


if __name__ == "__main__":
    main()
