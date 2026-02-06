import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Integrators (batch-friendly)
# x can be shape (2,) or (N,2)
# -----------------------
def euler_step(f, x, dt, t=0.0):
    return x + dt * f(t, x)

def rk4_step(f, x, dt, t=0.0):
    k1 = f(t, x)
    k2 = f(t + 0.5*dt, x + 0.5*dt*k1)
    k3 = f(t + 0.5*dt, x + 0.5*dt*k2)
    k4 = f(t + dt,     x + dt*k3)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def simulate(f, x0, dt, T, method="rk4"):
    """
    Simulate x' = f(t,x) from x0 for T seconds with step dt.
    Returns times t (len steps+1) and states X (shape (steps+1,2) or (steps+1,N,2)).
    """
    step = rk4_step if method.lower() == "rk4" else euler_step
    steps = int(np.round(T / dt))
    t = np.linspace(0.0, steps*dt, steps+1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    x = np.array(x0, dtype=float)
    X = np.empty((steps+1,) + x.shape, dtype=float)
    X[0] = x

    for k in range(steps):
        x = step(f, x, dt, t=t[k])
        X[k+1] = x

    return t, X

# -----------------------
# Systems
# -----------------------
def linear_system(A):
    """
    Takes a 2x2 matrix A and returns a function f(t, x) 
    ready for your RK4 solver.
    """
    def f(t, x):
        # x is the state vector [x1, x2]
        # returns the dot product Ax
        return A @ x 
    return f


if __name__ == "__main__":

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
    np.savez("data/trajectories/simulated_trajectories.npz", t=t, X=X)

    # TODO: save labels (true system dynamics)

    # How to use:
    # data = np.load("data/trajectories/simulated_trajectories.npz")
    # t = data["t"] # (20001, 2)
    # X = data["X"] # (20001,)
    # plt.plot(X[:, 0], X[:, 1])