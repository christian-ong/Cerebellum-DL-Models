import numpy as np
import pysindy as ps
from src.models.expander import SPECIFIC_BASES

class SINDyBaseline:
    """
    Thin wrapper around PySINDy.

    Supported library types:
        - polynomial
        - fourier
        - poly_fourier
        - specific
    """

    VAR_NAMES = ["x", "y", "z", "w", "v", "u"]

    def __init__(
        self,
        discrete_time: bool = True,
        poly_order: int = 3,
        include_bias: bool = True,
        include_interaction: bool = True,
        threshold: float = 0.1,
        alpha: float = 0.0,
        differentiation_method: str = "finite_difference",
        library_type: str = "polynomial",
        fourier_n_frequencies: int = 1,
        specific_system: str = None,
        specific_basis_size: int = None,
    ):
        self.discrete_time = discrete_time
        self.poly_order = poly_order
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.threshold = threshold
        self.alpha = alpha
        self.differentiation_method = differentiation_method

        self.library_type = library_type
        self.fourier_n_frequencies = fourier_n_frequencies
        self.specific_system = specific_system
        self.specific_basis_size = specific_basis_size

        self.model = None
        self.feature_names_ = None

    def _make_numpy_expr_fn(self, expr: str, state_dim: int):
        """
        Compile strings like:
            x^2*y
            sin(x)
            y*cos(2*x)
            1
        into NumPy callables with fixed arity equal to state_dim.

        For 2D systems:
            lambda x, y: ...
        For 3D systems:
            lambda x, y, z: ...
        """
        expr_py = expr.replace("^", "**")
        vars_used = self.VAR_NAMES[:state_dim]
        sig = ", ".join(vars_used)

        if expr.strip() == "1":
            body = f"np.ones_like({vars_used[0]})"
        else:
            body = expr_py

        fn = eval(
            f"lambda {sig}: {body}",
            {"np": np, "sin": np.sin, "cos": np.cos, "__builtins__": {}},
        )
        return fn

    def _make_name_fn(self, expr: str, state_dim: int):
        """
        Return a function with the same arity as the state dimension so PySINDy
        can generate readable feature names for CustomLibrary.
        """
        vars_used = self.VAR_NAMES[:state_dim]
        sig = ", ".join(vars_used)
        return eval(
            f"lambda {sig}: {expr!r}",
            {"__builtins__": {}},
        )

    def _build_feature_library(self, state_dim: int):
        if self.library_type == "polynomial":
            return ps.PolynomialLibrary(
                degree=self.poly_order,
                include_bias=self.include_bias,
                include_interaction=self.include_interaction,
            )

        if self.library_type == "fourier":
            return ps.FourierLibrary(
                n_frequencies=self.fourier_n_frequencies,
                include_sin=True,
                include_cos=True,
            )

        if self.library_type == "poly_fourier":
            poly_lib = ps.PolynomialLibrary(
                degree=self.poly_order,
                include_bias=self.include_bias,
                include_interaction=self.include_interaction,
            )
            fourier_lib = ps.FourierLibrary(
                n_frequencies=self.fourier_n_frequencies,
                include_sin=True,
                include_cos=True,
            )
            return ps.ConcatLibrary([poly_lib, fourier_lib])

        if self.library_type == "specific":
            if self.specific_system is None:
                raise ValueError("specific_system must be provided when library_type='specific'.")
            if self.specific_system not in SPECIFIC_BASES:
                raise ValueError(f"No specific SINDy basis defined for system '{self.specific_system}'.")

            basis_list = SPECIFIC_BASES[self.specific_system]
            if self.specific_basis_size is not None:
                if self.specific_basis_size > len(basis_list):
                    raise ValueError(
                        f"specific_basis_size={self.specific_basis_size} exceeds "
                        f"available basis size {len(basis_list)} for system '{self.specific_system}'."
                    )
                basis_list = basis_list[: self.specific_basis_size]

            library_functions = [
                self._make_numpy_expr_fn(expr, state_dim=state_dim)
                for expr in basis_list
            ]
            function_names = [
                self._make_name_fn(expr, state_dim=state_dim)
                for expr in basis_list
            ]

            return ps.CustomLibrary(
                library_functions=library_functions,
                function_names=function_names,
                interaction_only=False,
                include_bias=False,
            )

        raise ValueError(f"Unknown library_type: {self.library_type}")

    def _build_model(self, state_dim: int):
        self.feature_names_ = [f"x{i+1}" for i in range(state_dim)]
        library = self._build_feature_library(state_dim=state_dim)

        optimizer = ps.STLSQ(
            threshold=self.threshold,
            alpha=self.alpha,
        )

        if self.discrete_time:
            model = ps.DiscreteSINDy(
                optimizer=optimizer,
                feature_library=library,
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
            )

        return model

    def fit_discrete_pairs(self, X: np.ndarray, Y: np.ndarray):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must both have shape (N, d).")
        if X.shape != Y.shape:
            raise ValueError(f"Shape mismatch: X {X.shape}, Y {Y.shape}")

        self.model = self._build_model(state_dim=X.shape[1])
        self.model.fit(X, t=1, x_next=Y, feature_names=self.feature_names_)
        return self

    def fit_continuous_trajectories(self, X, dt: float):
        if isinstance(X, np.ndarray) and X.ndim == 3:
            trajectories = [X[:, i, :] for i in range(X.shape[1])]
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            trajectories = X
        else:
            trajectories = X

        state_dim = trajectories[0].shape[1] if isinstance(trajectories, list) else trajectories.shape[1]
        self.model = self._build_model(state_dim=state_dim)
        self.model.fit(trajectories, t=dt, feature_names=self.feature_names_)
        return self

    def predict_one_step(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fit yet.")

        x = np.asarray(x, dtype=float)
        single = (x.ndim == 1)
        if single:
            x = x[None, :]

        y = self.model.predict(x)
        return y[0] if single else y

    def rollout(self, x0: np.ndarray, steps: int) -> np.ndarray:
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
# class SINDyBaseline:
#     """
#     Thin wrapper around PySINDy so it matches the style of the repo.

#     Supports:
#     - continuous-time SINDy: fit x_dot = f(x), simulate forward
#     - discrete-time SINDy: fit x_{k+1} = F(x_k), simulate forward

#     For your current one-step benchmarking setup, discrete-time is the easiest
#     because your repo already works heavily with one-step pairs.
#     """

#     def __init__(
#         self,
#         discrete_time: bool = True,
#         poly_order: int = 3,
#         include_bias: bool = True,
#         include_interaction: bool = True,
#         threshold: float = 0.1,
#         alpha: float = 0.0,
#         differentiation_method: str = "finite_difference",
#     ):
#         self.discrete_time = discrete_time
#         self.poly_order = poly_order
#         self.include_bias = include_bias
#         self.include_interaction = include_interaction
#         self.threshold = threshold
#         self.alpha = alpha
#         self.differentiation_method = differentiation_method

#         self.model = None
#         self.feature_names_ = None

#     def _build_model(self, state_dim: int):
#         feature_names = [f"x{i+1}" for i in range(state_dim)]

#         library = ps.PolynomialLibrary(
#             degree=self.poly_order,
#             include_bias=self.include_bias,
#             include_interaction=self.include_interaction,
#         )

#         optimizer = ps.STLSQ(
#             threshold=self.threshold,
#             alpha=self.alpha,
#         )

#         if self.discrete_time:
#             model = ps.DiscreteSINDy(
#                 optimizer=optimizer,
#                 feature_library=library,
#             )
#         else:
#             if self.differentiation_method == "smoothed_finite_difference":
#                 diff_method = ps.SmoothedFiniteDifference()
#             else:
#                 diff_method = ps.FiniteDifference()

#             model = ps.SINDy(
#                 optimizer=optimizer,
#                 feature_library=library,
#                 differentiation_method=diff_method,
#             )

#         self.feature_names_ = feature_names
#         return model

#     def fit_discrete_pairs(self, X: np.ndarray, Y: np.ndarray):
#         """
#         Fit discrete SINDy from one-step pairs:
#             x_{k+1} = F(x_k)

#         X: (N, d)
#         Y: (N, d)
#         """
#         X = np.asarray(X, dtype=float)
#         Y = np.asarray(Y, dtype=float)

#         if X.ndim != 2 or Y.ndim != 2:
#             raise ValueError("X and Y must both have shape (N, d).")
#         if X.shape != Y.shape:
#             raise ValueError(f"Shape mismatch: X {X.shape}, Y {Y.shape}")

#         self.model = self._build_model(state_dim=X.shape[1])
#         self.model.fit(X, x_next=Y, t=1, feature_names=self.feature_names_)
#         return self

#     def fit_continuous_trajectories(self, X, dt: float):
#         """
#         Fit continuous-time SINDy from trajectory data.
#         X can be:
#           - array (T, d)
#           - array (T, n_traj, d)
#           - list of arrays [(T1,d), (T2,d), ...]
#         """
#         if isinstance(X, np.ndarray) and X.ndim == 3:
#             trajectories = [X[:, i, :] for i in range(X.shape[1])]
#         elif isinstance(X, np.ndarray) and X.ndim == 2:
#             trajectories = X
#         else:
#             trajectories = X

#         state_dim = trajectories[0].shape[1] if isinstance(trajectories, list) else trajectories.shape[1]
#         self.model = self._build_model(state_dim=state_dim)
#         self.model.fit(trajectories, t=dt, feature_names=self.feature_names_)
#         return self

#     def predict_one_step(self, x: np.ndarray) -> np.ndarray:
#         """
#         One-step prediction for a batch or single point.
#         """
#         if self.model is None:
#             raise RuntimeError("Model has not been fit yet.")

#         x = np.asarray(x, dtype=float)
#         single = (x.ndim == 1)
#         if single:
#             x = x[None, :]

#         y = self.model.predict(x)
#         return y[0] if single else y

#     def rollout(self, x0: np.ndarray, steps: int) -> np.ndarray:
#         """
#         Roll out for (steps+1, d), including x0.
#         """
#         if self.model is None:
#             raise RuntimeError("Model has not been fit yet.")

#         x0 = np.asarray(x0, dtype=float)

#         if self.discrete_time:
#             traj = [x0.copy()]
#             x = x0.copy()
#             for _ in range(steps):
#                 x = self.predict_one_step(x)
#                 traj.append(np.asarray(x, dtype=float).copy())
#             return np.asarray(traj)

#         # continuous-time simulation
#         t = np.arange(steps + 1, dtype=float)
#         traj = self.model.simulate(x0, t=t)
#         return np.asarray(traj)

#     def equations(self):
#         if self.model is None:
#             raise RuntimeError("Model has not been fit yet.")
#         return self.model.equations()

#     def print(self):
#         if self.model is None:
#             raise RuntimeError("Model has not been fit yet.")
#         self.model.print()

#     def get_coefficients(self):
#         if self.model is None:
#             raise RuntimeError("Model has not been fit yet.")
#         return self.model.coefficients()