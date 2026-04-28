import numpy as np

# -----------------------
# Integrators (batch-friendly)
# x can be shape (d,) or (N, d)
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
    Returns:
    X : (steps+1, d) or (steps+1, N, d).
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
    Takes a (d x d) matrix A and returns a function f(t, x) 
    f is the vector field for the linear system x' = A x
    """
    A = np.asarray(A, dtype=float)
    def f(t, x):
        # x is the state vector [x1, x2, ..., xd]
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


def duffing_system(alpha=1.0, beta=1.0, delta=0.2,
                          gamma=0.3, omega=1.0):
    """
    Forced Duffing oscillator:
        x' = y
        y' = -delta*y - alpha*x - beta*x^3 + gamma*cos(omega*t)
    """
    def f(t, x):
        x = np.asarray(x, dtype=float)
        q = x[..., 0]
        p = x[..., 1]

        dq = p
        dp = -delta * p - alpha * q - beta * q**3 + gamma * np.cos(omega * t)

        return np.stack([dq, dp], axis=-1)

    return f

def closed_small_system(mu=0.1, alpha=-1.0):
    def f(t, x):
        x = np.asarray(x, dtype=float)
        x1 = x[..., 0]
        x2 = x[..., 1]
        dx1 = mu * x1
        dx2 = alpha * (x2 - x1**2)
        return np.stack([dx1, dx2], axis=-1)
    return f

def closed_large_system(mu=0.1, alpha=-1.0, beta=0.8, gamma=-0.4, delta=0.2):
    """
    2D nonlinear system with exact finite-dimensional polynomial closure:
        x' = mu * x
        y' = alpha * y + beta * x^2 + gamma * x^3 + delta * x^4

    In lifted coordinates z = [x, y, x^2, x^3, x^4], the dynamics are linear.
    """
    def f(t, x):
        x = np.asarray(x, dtype=float)
        x1 = x[..., 0]
        x2 = x[..., 1]

        dx1 = mu * x1
        dx2 = alpha * x2 + beta * x1**2 + gamma * x1**3 + delta * x1**4

        return np.stack([dx1, dx2], axis=-1)

    return f

def closed_trig_small_system(
    omega=1.0,
    alpha=-0.8,
    beta_s1=0.7,
    beta_c1=-0.5,
    beta_x=0.3,
    beta_x2=-0.08,
):
    """
    2D nonlinear system with an exact finite-dimensional closure in a
    custom polynomial + trigonometric observable dictionary.
    Dynamics:
    x' = omega
    y' = alpha*y + beta_s1*sin(x) + beta_c1*cos(x) + beta_x*x + beta_x2*x^2
    A suitable exact lifted dictionary is for example:
    [1, x, y, x^2, sin(x), cos(x)]
    """
    def f(t, x):
        x = np.asarray(x, dtype=float)
        x1 = x[..., 0]
        x2 = x[..., 1]

        dx1 = np.full_like(x1, omega, dtype=float)
        dx2 = alpha * x2 + beta_s1 * np.sin(x1) + beta_c1 * np.cos(x1) + beta_x * x1 + beta_x2 * x1**2

        return np.stack([dx1, dx2], axis=-1)

    return f

def closed_trig_medium_system(
    omega=1.0,
    alpha=-0.8,
    beta_s1=0.7,
    beta_c1=-0.5,
    beta_s2=0.4,
    beta_c2=0.2,
    beta_x=0.3,
    beta_x2=-0.08,
):
    """
    2D nonlinear system with an exact finite-dimensional closure in a
    custom polynomial + trigonometric observable dictionary.
    Dynamics:
    x' = omega
    y' = alpha*y + beta_s1*sin(x) + beta_c1*cos(x) + beta_s2*sin(2x) + beta_c2*cos(2x) + 
    A suitable exact lifted dictionary is for example:
    [1, x, y, x^2, sin(x), cos(x), sin(2x), cos(2x)]
    """

    def f(t, x):
        x = np.asarray(x, dtype=float)
        x1 = x[..., 0]
        x2 = x[..., 1]

        dx1 = np.full_like(x1, omega, dtype=float)
        dx2 = (
            alpha * x2
            + beta_s1 * np.sin(x1)
            + beta_c1 * np.cos(x1)
            + beta_s2 * np.sin(2.0 * x1)
            + beta_c2 * np.cos(2.0 * x1)
            + beta_x * x1
            + beta_x2 * x1**2
        )

        return np.stack([dx1, dx2], axis=-1)

    return f


def closed_trig_large_system(
    omega=1.0,
    alpha=-0.8,
    beta_s1=0.7,
    beta_c1=-0.5,
    beta_s2=0.4,
    beta_c2=0.2,
    beta_s3=-0.25,
    beta_c3=0.15,
    beta_x=0.3,
    beta_x2=-0.08,
):
    """
    2D nonlinear system with an exact finite-dimensional closure in a
    custom polynomial + trigonometric observable dictionary.

    Dynamics:
        x' = omega
        y' = alpha*y
             + beta_s1*sin(x) + beta_c1*cos(x)
             + beta_s2*sin(2x) + beta_c2*cos(2x)
             + beta_s3*sin(3x) + beta_c3*cos(3x)
             + beta_x*x + beta_x2*x^2

    A suitable exact lifted dictionary is for example:
        [1, x, y, x^2, sin(x), cos(x), sin(2x), cos(2x), sin(3x), cos(3x)]
    """
    def f(t, x):
        x = np.asarray(x, dtype=float)
        x1 = x[..., 0]
        x2 = x[..., 1]

        dx1 = np.full_like(x1, omega, dtype=float)
        dx2 = (
            alpha * x2
            + beta_s1 * np.sin(x1)
            + beta_c1 * np.cos(x1)
            + beta_s2 * np.sin(2.0 * x1)
            + beta_c2 * np.cos(2.0 * x1)
            + beta_s3 * np.sin(3.0 * x1)
            + beta_c3 * np.cos(3.0 * x1)
            + beta_x * x1
            + beta_x2 * x1**2
        )

        return np.stack([dx1, dx2], axis=-1)

    return f