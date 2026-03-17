import numpy as np
import pysindy as ps


class SINDyBaseline:
    """
    Thin wrapper around PySINDy so it matches the style of the repo.

    Supports:
    - continuous-time SINDy: fit x_dot = f(x), simulate forward
    - discrete-time SINDy: fit x_{k+1} = F(x_k), simulate forward

    For your current one-step benchmarking setup, discrete-time is the easiest
    because your repo already works heavily with one-step pairs.
    """

    def __init__(
        self,
        discrete_time: bool = True,
        poly_order: int = 3,
        include_bias: bool = True,
        include_interaction: bool = True,
        threshold: float = 0.1,
        alpha: float = 0.0,
        differentiation_method: str = "finite_difference",
    ):
        self.discrete_time = discrete_time
        self.poly_order = poly_order
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.threshold = threshold
        self.alpha = alpha
        self.differentiation_method = differentiation_method

        self.model = None
        self.feature_names_ = None

    def _build_model(self, state_dim: int):
        feature_names = [f"x{i+1}" for i in range(state_dim)]

        library = ps.PolynomialLibrary(
            degree=self.poly_order,
            include_bias=self.include_bias,
            include_interaction=self.include_interaction,
        )

        optimizer = ps.STLSQ(
            threshold=self.threshold,
            alpha=self.alpha,
        )

        if self.discrete_time:
            model = ps.DiscreteSINDy(
                optimizer=optimizer,
                feature_library=library,
                feature_names=feature_names,
            )
        else:
            if self.differentiation_method == "smoothed_finite_difference":
                diff_method = ps.SmoothedFiniteDifference()
            else:
                diff_method = ps.FiniteDifference()

            model = ps.SINDy(
                optimizer=optimizer,
                feature_library=library,
                differentiation_method=diff_method,
                feature_names=feature_names,
            )

        self.feature_names_ = feature_names
        return model

    def fit_discrete_pairs(self, X: np.ndarray, Y: np.ndarray):
        """
        Fit discrete SINDy from one-step pairs:
            x_{k+1} = F(x_k)

        X: (N, d)
        Y: (N, d)
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must both have shape (N, d).")
        if X.shape != Y.shape:
            raise ValueError(f"Shape mismatch: X {X.shape}, Y {Y.shape}")

        self.model = self._build_model(state_dim=X.shape[1])
        self.model.fit(X, t=1, x_next=Y)
        return self

    def fit_continuous_trajectories(self, X, dt: float):
        """
        Fit continuous-time SINDy from trajectory data.
        X can be:
          - array (T, d)
          - array (T, n_traj, d)
          - list of arrays [(T1,d), (T2,d), ...]
        """
        if isinstance(X, np.ndarray) and X.ndim == 3:
            trajectories = [X[:, i, :] for i in range(X.shape[1])]
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            trajectories = X
        else:
            trajectories = X

        state_dim = trajectories[0].shape[1] if isinstance(trajectories, list) else trajectories.shape[1]
        self.model = self._build_model(state_dim=state_dim)
        self.model.fit(trajectories, t=dt)
        return self

    def predict_one_step(self, x: np.ndarray) -> np.ndarray:
        """
        One-step prediction for a batch or single point.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fit yet.")

        x = np.asarray(x, dtype=float)
        single = (x.ndim == 1)
        if single:
            x = x[None, :]

        y = self.model.predict(x)
        return y[0] if single else y

    def rollout(self, x0: np.ndarray, steps: int) -> np.ndarray:
        """
        Roll out for (steps+1, d), including x0.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fit yet.")

        x0 = np.asarray(x0, dtype=float)

        if self.discrete_time:
            traj = [x0.copy()]
            x = x0.copy()
            for _ in range(steps):
                x = self.predict_one_step(x)
                traj.append(np.asarray(x, dtype=float).copy())
            return np.asarray(traj)

        # continuous-time simulation
        t = np.arange(steps + 1, dtype=float)
        traj = self.model.simulate(x0, t=t)
        return np.asarray(traj)

    def equations(self):
        if self.model is None:
            raise RuntimeError("Model has not been fit yet.")
        return self.model.equations()

    def print(self):
        if self.model is None:
            raise RuntimeError("Model has not been fit yet.")
        self.model.print()

    def get_coefficients(self):
        if self.model is None:
            raise RuntimeError("Model has not been fit yet.")
        return self.model.coefficients()