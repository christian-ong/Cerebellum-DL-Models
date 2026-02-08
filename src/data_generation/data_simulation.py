import numpy as np

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
# Linear Systems
# -----------------------
def linear_system(A):
    """
    Takes a 2x2 matrix A and returns a function f(t, x) 
    ready for your RK4 solver.
    """
    A = np.asarray(A, dtype=float)
    def f(t, x):
        # x is the state vector [x1, x2]
        # returns the dot product Ax
        x = np.asarray(x, dtype=float)
        return x @ A.T
    return f

# -----------------------
# Nonlinear Systems
# -----------------------
def vanderpol_system(mu=1.5):
    def f(t, x):
        x = np.asarray(x, dtype=float)
        x1 = x[..., 0]
        x2 = x[..., 1]
        dx1 = x2
        dx2 = mu * (1 - x1**2) * x2 - x1
        return np.stack([dx1, dx2], axis=-1)
    return f

def lotka_volterra_system(alpha=1.1, beta=0.4, delta=0.1, gamma=0.4):
    def f(t, x):
        x = np.asarray(x, dtype=float)
        prey = x[..., 0]
        pred = x[..., 1]
        d_prey = alpha * prey - beta * prey * pred
        d_pred = -gamma * pred + delta * prey * pred
        return np.stack([d_prey, d_pred], axis=-1)
    return f

def pendulum_system(g=9.81, L=1.0):
    def f(t, x):
        x = np.asarray(x, dtype=float)
        theta = x[..., 0]
        omega = x[..., 1]
        d_theta = omega
        d_omega = -(g / L) * np.sin(theta)
        return np.stack([d_theta, d_omega], axis=-1)
    return f

def lorenz_system(sigma=10.0, rho=28.0, beta=8/3):
    def f(t, x):
        # x is now [x, y, z]
        xs = x[..., 0]; ys = x[..., 1]; zs = x[..., 2]
        dx = sigma * (ys - xs)
        dy = xs * (rho - zs) - ys
        dz = xs * ys - beta * zs
        return np.stack([dx, dy, dz], axis=-1)
    return f