import numpy as np
from scipy.integrate import odeint

from src.models.expander import SPECIFIC_BASES

try:
    import pysindy as ps
except ImportError:
    ps = None


class SINDyBaseline:
    """
    SINDy wrapper with:
    - polynomial library
    - fourier library
    - poly_fourier library
    - exact hand-written 'specific' library using SPECIFIC_BASES

    For library_type='specific', we manually build the feature matrix Theta(X)
    and run PySINDy's sparse optimizer (STLSQ) directly on Theta.
    """

    VAR_NAMES = ["x", "y", "z", "w", "v", "u"]

    def __init__(
        self,
        discrete_time=True,
        poly_order=3,
        include_bias=True,
        include_interaction=True,
        threshold=0.1,
        alpha=0.0,
        differentiation_method="finite_difference",
        library_type="polynomial",
        fourier_n_frequencies=1,
        specific_system=None,
        specific_basis_size=None,
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
        self.saved_coefficients = None
        self.powers = None
        self.dt = 1.0
        self.state_dim = None

        self.feature_library = None
        self.feature_names_list = None

        self._specific_basis_exprs = None
        self._specific_basis_fns = None

    # --------------------------------------------------
    # Standard PySINDy feature libraries
    # --------------------------------------------------

    def _build_feature_library(self, state_dim: int):
        if ps is None:
            raise ImportError(
                "pysindy is required to use SINDyBaseline. Install the optional dependency to enable SINDy support."
            )

        if self.library_type == "polynomial":
            return ps.PolynomialLibrary(
                degree=self.poly_order,
                include_bias=self.include_bias,
                include_interaction=self.include_interaction,
            )

        if self.library_type == "fourier":
            return ps.FourierLibrary(n_frequencies=self.fourier_n_frequencies)

        if self.library_type == "poly_fourier":
            poly = ps.PolynomialLibrary(
                degree=self.poly_order,
                include_bias=self.include_bias,
                include_interaction=self.include_interaction,
            )
            fourier = ps.FourierLibrary(n_frequencies=self.fourier_n_frequencies)
            return ps.ConcatLibrary([poly, fourier])

        if self.library_type == "specific":
            return None

        raise ValueError(f"Unknown library_type: {self.library_type}")

    def _build_model(self, state_dim: int):
        if self.library_type == "specific":
            return None

        optimizer = ps.STLSQ(threshold=self.threshold, alpha=self.alpha)
        library = self._build_feature_library(state_dim)

        if self.discrete_time:
            return ps.DiscreteSINDy(
                optimizer=optimizer,
                feature_library=library,
            )

        if self.differentiation_method == "smoothed_finite_difference":
            diff = ps.SmoothedFiniteDifference()
        else:
            diff = ps.FiniteDifference()

        return ps.SINDy(
            optimizer=optimizer,
            feature_library=library,
            differentiation_method=diff,
        )
    
    def _compile_pysindy_features(self, feature_names):
        """Dynamically builds and compiles a single, blazing-fast native NumPy function."""
        import re
        
        exprs = []
        for expr in feature_names:
            if expr.strip() == "1":
                exprs.append("np.ones(X.shape[0], dtype=float)")
                continue

            # Format powers and trig frequencies
            expr_py = expr.replace("^", "**")
            expr_py = re.sub(r'(\d+)\s+(x\d+)', r'\1 * \2', expr_py)
            
            # Format polynomial interactions
            for _ in range(3): 
                expr_py = re.sub(r'(x\d+(?:\*\*\d+)?)\s+(x\d+)', r'\1 * \2', expr_py)
            exprs.append(expr_py)
            
        # 1. Write the raw code for a native Python function
        func_str = "def mega_eval(X, sin, cos):\n"
        for i in range(self.state_dim):
            func_str += f"    x{i} = X[:, {i}]\n"
            
        func_str += f"    return np.column_stack([{', '.join(exprs)}])\n"
        
        # 2. Compile it natively into memory
        local_vars = {"np": np}
        exec(func_str, local_vars)
        mega_eval = local_vars["mega_eval"]
        
        # 3. Wrap it so we can inject np.sin and np.cos
        def final_fn(X):
            return mega_eval(X, np.sin, np.cos)
            
        return [final_fn]
    # --------------------------------------------------
    # Specific library helpers
    # --------------------------------------------------

    def _get_specific_basis_exprs(self):
        if self.specific_system is None:
            raise ValueError("specific_system must be provided when library_type='specific'.")

        if self.specific_system not in SPECIFIC_BASES:
            raise ValueError(f"No specific basis defined for system '{self.specific_system}'.")

        basis = SPECIFIC_BASES[self.specific_system]

        if self.specific_basis_size is None:
            return basis

        if self.specific_basis_size > len(basis):
            raise ValueError(
                f"specific_basis_size={self.specific_basis_size} exceeds available "
                f"basis size {len(basis)} for system '{self.specific_system}'."
            )

        return basis[: self.specific_basis_size]

    def _compile_specific_basis_fn(self, expr):
        expr_py = expr.replace("^", "**")
        allowed_names = {
            "sin": np.sin,
            "cos": np.cos,
        }

        def fn(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X[None, :]

            var_dict = {}
            for i in range(X.shape[1]):
                if i < len(self.VAR_NAMES):
                    name = self.VAR_NAMES[i]
                    var_dict[name] = X[:, i]
                    var_dict[f"{name}r"] = X[:, i]
                else:
                    name = f"x{i+1}"
                    var_dict[name] = X[:, i]
                    var_dict[f"{name}r"] = X[:, i]

            out = eval(expr_py, {"__builtins__": {}}, {**allowed_names, **var_dict})
            if np.isscalar(out):
                out = np.full(X.shape[0], float(out))
            return np.asarray(out, dtype=float)

        return fn

    def _ensure_specific_basis(self):
        if self._specific_basis_exprs is not None and self._specific_basis_fns is not None:
            return

        self._specific_basis_exprs = self._get_specific_basis_exprs()
        self._specific_basis_fns = [
            self._compile_specific_basis_fn(expr) for expr in self._specific_basis_exprs
        ]
        self.feature_names_list = list(self._specific_basis_exprs)

    def _specific_transform(self, X):
        self._ensure_specific_basis()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        cols = [fn(X) for fn in self._specific_basis_fns]
        return np.column_stack(cols)

    def _differentiate_trajectory(self, X, dt):
        X = np.asarray(X, dtype=float)
        if self.differentiation_method == "smoothed_finite_difference":
            diff = ps.SmoothedFiniteDifference()
        else:
            diff = ps.FiniteDifference()
        return diff._differentiate(X, dt)

    def _fit_specific_discrete_pairs(self, X, Y):
        Theta = self._specific_transform(X)
        opt = ps.STLSQ(threshold=self.threshold, alpha=self.alpha)
        opt.fit(Theta, Y)

        self.model = None
        self.feature_library = None
        self.saved_coefficients = np.asarray(opt.coef_, dtype=float)
        self.powers = None
        self.state_dim = int(np.asarray(X).shape[1])
        return self

    def _fit_specific_continuous_trajectories(self, X_traj, dt):
        self.dt = dt
        Theta_list = []
        xdot_list = []

        for i in range(X_traj.shape[1]):
            Xi = np.asarray(X_traj[:, i, :], dtype=float)
            Theta_i = self._specific_transform(Xi)
            xdot_i = self._differentiate_trajectory(Xi, dt)

            Theta_list.append(Theta_i)
            xdot_list.append(xdot_i)

        Theta = np.vstack(Theta_list)
        Xdot = np.vstack(xdot_list)

        opt = ps.STLSQ(threshold=self.threshold, alpha=self.alpha)
        opt.fit(Theta, Xdot)

        self.model = None
        self.feature_library = None
        self.saved_coefficients = np.asarray(opt.coef_, dtype=float)
        self.powers = None
        self.state_dim = int(np.asarray(X_traj).shape[-1])
        return self

    # --------------------------------------------------
    # Standard library fast-power helper
    # --------------------------------------------------

    def _get_fast_powers(self, lib, state_dim):
        try:
            features = lib.get_feature_names([f"x{i}" for i in range(state_dim)])
            powers = np.zeros((len(features), state_dim), dtype=int)
            for i, f in enumerate(features):
                if f == "1" or f.strip() == "1":
                    continue
                if "sin" in f or "cos" in f or "(" in f:
                    return None

                terms = f.split(" ")
                for term in terms:
                    if not term:
                        continue
                    if "^" in term:
                        var, p = term.split("^")
                        idx = int(var.replace("x", ""))
                        powers[i, idx] = int(p)
                    else:
                        idx = int(term.replace("x", ""))
                        powers[i, idx] = 1
            return powers
        except Exception:
            return None

    def _compute_polynomial_features(self, x: np.ndarray) -> np.ndarray:
        """
        Compute polynomial features up to self.poly_order for a single state vector x.
        This is a best-effort fallback to avoid calling into PySINDy during heavy loops.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        state_dim = x.size

        feats = []
        # optional bias
        if self.include_bias:
            feats.append(1.0)

        # degree 1..poly_order
        from itertools import combinations_with_replacement

        for deg in range(1, self.poly_order + 1):
            for terms in combinations_with_replacement(range(state_dim), deg):
                prod = 1.0
                for t in terms:
                    prod *= x[t]
                feats.append(prod)

        feats = np.asarray(feats, dtype=float)
        expected = int(self.saved_coefficients.shape[1])
        if feats.size == expected:
            return feats

        # If sizes differ, try to align by trimming or padding zeros
        if feats.size > expected:
            return feats[:expected]
        out = np.zeros((expected,), dtype=float)
        out[: feats.size] = feats
        return out

    # --------------------------------------------------
    # Fit
    # --------------------------------------------------

    def fit_discrete_pairs(self, X: np.ndarray, Y: np.ndarray):
        if self.library_type == "specific":
            return self._fit_specific_discrete_pairs(X, Y)

        self.state_dim = int(np.asarray(X).shape[1])
        self.model = self._build_model(state_dim=X.shape[1])
        self.model.fit(X, t=1.0, x_next=Y)
        self._finalize_fit(X.shape[1])
        return self

    def fit_continuous_trajectories(self, X_traj, dt: float):
        if self.library_type == "specific":
            return self._fit_specific_continuous_trajectories(X_traj, dt)

        self.dt = dt
        self.state_dim = int(np.asarray(X_traj).shape[-1])
        trajectories = [X_traj[:, i, :] for i in range(X_traj.shape[1])]
        self.model = self._build_model(state_dim=X_traj.shape[-1])
        self.model.fit(trajectories, t=dt)
        self._finalize_fit(X_traj.shape[-1])
        return self

    def _finalize_fit(self, state_dim):
        self.saved_coefficients = self.model.coefficients()
        self.feature_library = self.model.feature_library
        self.feature_names_list = self.feature_library.get_feature_names(
            [f"x{i}" for i in range(state_dim)]
        )
        self.powers = self._get_fast_powers(self.feature_library, state_dim)
        
        if self.powers is None:
            self._compiled_pysindy_fns = self._compile_pysindy_features(self.feature_names_list)
        else:
            self._compiled_pysindy_fns = None

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    def load_model(self, coefficients: np.ndarray, state_dim: int):
        self.saved_coefficients = coefficients
        self.state_dim = int(state_dim)

        if self.library_type == "specific":
            self._ensure_specific_basis()
            self.feature_library = None
            self.powers = None
            return self

        lib = self._build_feature_library(state_dim=state_dim)
        lib.fit(np.zeros((1, state_dim)))
        self.feature_library = lib
        self.feature_names_list = lib.get_feature_names([f"x{i}" for i in range(state_dim)])
        self.powers = self._get_fast_powers(lib, state_dim)
        
        if self.powers is None:
            self._compiled_pysindy_fns = self._compile_pysindy_features(self.feature_names_list)
        else:
            self._compiled_pysindy_fns = None
            
        return self

    # --------------------------------------------------
    # Rollout
    # --------------------------------------------------

    def rollout(self, x0: np.ndarray, steps: int) -> np.ndarray:
        if self.saved_coefficients is None:
            raise RuntimeError("Model not fit or loaded.")

        x0 = np.asarray(x0, dtype=float)
        coef_T = self.saved_coefficients.T
        local_powers = self.powers

        # 1. The SPECIFIC Library Loop
        if self.library_type == "specific":
            if self.discrete_time:
                traj = np.zeros((steps + 1, x0.shape[0]))
                traj[0] = x0
                curr_x = x0
                for i in range(1, steps + 1):
                    feat = self._specific_transform(curr_x)[0]
                    curr_x = feat @ coef_T
                    traj[i] = curr_x
                return traj

            t = np.arange(steps + 1) * self.dt

            def rhs(x_state, t_dummy):
                feat = self._specific_transform(x_state)[0]
                return feat @ coef_T

            return odeint(rhs, x0, t)

        # 2. The STANDARD PySINDy Library Loop
        lib = self.feature_library

        if self.discrete_time:
            traj = np.zeros((steps + 1, x0.shape[0]))
            traj[0] = x0
            curr_x = x0
            for i in range(1, steps + 1):
                if local_powers is not None:
                    feat = np.prod(np.power(curr_x, local_powers), axis=1)
                elif getattr(self, "_compiled_pysindy_fns", None) is not None:
                    x_2d = curr_x[None, :]
                    cols = [fn(x_2d) for fn in self._compiled_pysindy_fns]
                    feat = np.column_stack(cols)[0]
                else:
                    try:
                        feat = lib.transform(curr_x[None, :])[0]
                    except Exception:
                        if self.library_type in {"polynomial", "poly_fourier"}:
                            feat = self._compute_polynomial_features(curr_x)
                        else:
                            feat = np.zeros((self.saved_coefficients.shape[1],), dtype=float)
                curr_x = feat @ coef_T
                traj[i] = curr_x
            return traj

        t = np.arange(steps + 1) * self.dt

        def rhs(x_state, t_dummy):
            if local_powers is not None:
                feat = np.prod(np.power(x_state, local_powers), axis=1)
            elif getattr(self, "_compiled_pysindy_fns", None) is not None:
                x_2d = x_state[None, :]
                cols = [fn(x_2d) for fn in self._compiled_pysindy_fns]
                feat = np.column_stack(cols)[0]
            else:
                try:
                    feat = lib.transform(x_state[None, :])[0]
                except Exception:
                    if self.library_type in {"polynomial", "poly_fourier"}:
                        feat = self._compute_polynomial_features(x_state)
                    else:
                        feat = np.zeros((self.saved_coefficients.shape[1],), dtype=float)
            return feat @ coef_T

        return odeint(rhs, x0, t)

    # --------------------------------------------------
    # Reporting
    # --------------------------------------------------

    def equations(self):
        if self.library_type != "specific":
            return self.model.equations() if self.model else []

        if self.saved_coefficients is None:
            return []

        coef = np.asarray(self.saved_coefficients, dtype=float)
        if coef.ndim == 1:
            coef = coef[None, :]

        target_names = self.VAR_NAMES[: coef.shape[0]]
        eqs = []

        for i, lhs in enumerate(target_names):
            terms = []
            for c, name in zip(coef[i], self.feature_names_list):
                if abs(c) > 0:
                    terms.append(f"{c:.6g} {name}")
            eqs.append(" + ".join(terms) if terms else "0")

        return eqs

    def print(self):
        eqs = self.equations()
        for i, eq in enumerate(eqs):
            lhs = self.VAR_NAMES[i] if i < len(self.VAR_NAMES) else f"x{i}"
            if self.discrete_time:
                print(f"({lhs})[k+1] = {eq}")
            else:
                print(f"d{lhs}/dt = {eq}")

    def get_coefficients(self):
        return self.saved_coefficients