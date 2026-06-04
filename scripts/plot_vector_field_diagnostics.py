import argparse
import os
from typing import Optional, Tuple, List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.data_generation.load_data import resolve_split_npz_path
from src.eval.diagnostics import build_true_dynamics_from_dataset


EPS = 1e-12
NULLCLINE_COLORS = [
    "red",
    "orange",
    "magenta",
    "cyan",
    "lime",
    "yellow",
]


def parse_lim(text: Optional[str]) -> Optional[Tuple[float, float]]:
    if text is None:
        return None

    parts = [float(v.strip()) for v in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected limit as 'min,max', got: {text}")
    if parts[0] >= parts[1]:
        raise ValueError(f"Invalid limit {text}: min must be smaller than max.")

    return parts[0], parts[1]


def parse_components(text: str, state_dim: int, x_dim: int, y_dim: int) -> List[int]:
    text = text.strip().lower()

    if text == "auto":
        if state_dim <= 3:
            return list(range(state_dim))
        return sorted(set([x_dim, y_dim]))

    if text == "all":
        return list(range(state_dim))

    comps = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not comps:
        raise ValueError("No components were provided.")

    for c in comps:
        if c < 0 or c >= state_dim:
            raise ValueError(f"Component {c} is out of range for state_dim={state_dim}.")

    return sorted(set(comps))


def infer_axis_limits(values: np.ndarray, pad_fraction: float = 0.08) -> Tuple[float, float]:
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Cannot infer axis limits from non-finite data.")

    if abs(hi - lo) < EPS:
        return lo - 1.0, hi + 1.0

    pad = pad_fraction * (hi - lo)
    return lo - pad, hi + pad


def fmt_tick(x: float) -> str:
    if not np.isfinite(x):
        return "nan"

    ax = abs(float(x))
    if ax < 1e-12:
        return "0"
    if 1e-2 <= ax < 1e3:
        return f"{x:.2g}"
    return f"{x:.1e}"


def make_symlog_norm(values: np.ndarray, linthresh: float) -> Tuple[mcolors.Normalize, float]:
    max_abs = float(np.nanmax(np.abs(values)))
    max_abs = max(max_abs, EPS)

    norm = mcolors.SymLogNorm(
        linthresh=linthresh,
        linscale=1.0,
        vmin=-max_abs,
        vmax=max_abs,
    )
    return norm, max_abs


def make_speed_norm(speed: np.ndarray) -> Tuple[mcolors.Normalize, float, float]:
    finite = speed[np.isfinite(speed)]
    finite = finite[finite > 0]

    if finite.size == 0:
        return mcolors.Normalize(vmin=0.0, vmax=1.0), 0.0, 1.0

    vmin = max(float(np.nanpercentile(finite, 1)), 1e-8)
    vmax = max(float(np.nanpercentile(finite, 99)), vmin * 10.0)

    return mcolors.LogNorm(vmin=vmin, vmax=vmax), vmin, vmax


def add_three_tick_colorbar(fig, mappable, ax, *, label: str, kind: str, max_abs=None, vmin=None, vmax=None):
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(label)

    if kind == "signed":
        ticks = [-float(max_abs), 0.0, float(max_abs)]
        labels = [fmt_tick(ticks[0]), "0", fmt_tick(ticks[2])]
    elif kind == "positive":
        vmin = float(vmin)
        vmax = float(vmax)
        mid = np.sqrt(max(vmin, EPS) * max(vmax, EPS))
        ticks = [vmin, mid, vmax]
        labels = [fmt_tick(vmin), fmt_tick(mid), fmt_tick(vmax)]
    else:
        raise ValueError(f"Unknown colorbar kind: {kind}")

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    return cbar


def contour_if_available(ax, XX, YY, Z, *, color, linewidth=1.6, alpha=0.9):
    zmin = float(np.nanmin(Z))
    zmax = float(np.nanmax(Z))

    if zmin <= 0.0 <= zmax:
        ax.contour(
            XX,
            YY,
            Z,
            levels=[0.0],
            colors=color,
            linestyles="--",
            linewidths=linewidth,
            alpha=alpha,
        )
        return True

    return False


def add_nullcline_legend(ax, components: List[int], present: dict, *, loc: str = "best"):
    handles = []

    for comp in components:
        if not present.get(comp, False):
            continue

        color = NULLCLINE_COLORS[comp % len(NULLCLINE_COLORS)]
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=1.7,
                linestyle="--",
                label=rf"$\dot x_{comp + 1}=0$",
            )
        )

    if handles:
        ax.legend(handles=handles, loc=loc, framealpha=0.9)


def get_plane_pairs(state_dim: int, x_dim: int, y_dim: int, all_plane_pairs: bool) -> List[Tuple[int, int]]:
    if all_plane_pairs:
        pairs = []
        for i in range(state_dim):
            for j in range(i + 1, state_dim):
                pairs.append((i, j))
        return pairs
    return [(x_dim, y_dim)]


def build_base_state(flat: np.ndarray, state_dim: int, slice_values: Optional[str]) -> np.ndarray:
    if slice_values is not None:
        base_state = np.asarray([float(v.strip()) for v in slice_values.split(",")], dtype=float)
        if base_state.shape[0] != state_dim:
            raise ValueError(f"slice_values must have length {state_dim}, got {base_state.shape[0]}.")
        return base_state

    return np.nanmean(flat, axis=0)


def compute_plane_fields(
    *,
    f_true,
    base_state: np.ndarray,
    state_dim: int,
    x_dim: int,
    y_dim: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    grid_resolution: int,
    components: List[int],
):
    xs = np.linspace(xlim[0], xlim[1], grid_resolution)
    ys = np.linspace(ylim[0], ylim[1], grid_resolution)
    XX, YY = np.meshgrid(xs, ys)

    grid_pts = np.repeat(base_state[None, :], XX.size, axis=0)
    grid_pts[:, x_dim] = XX.ravel()
    grid_pts[:, y_dim] = YY.ravel()

    vf = np.asarray(f_true(0.0, grid_pts))
    if vf.shape != grid_pts.shape:
        raise ValueError(
            f"Expected vector field output shape {grid_pts.shape}, got {vf.shape}."
        )

    component_fields = {
        comp: vf[:, comp].reshape(XX.shape)
        for comp in components
    }
    speed = np.linalg.norm(vf, axis=1).reshape(XX.shape)

    return XX, YY, component_fields, speed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot true vector-field diagnostics: velocity components, speed, "
            "and nullclines for 2D systems or 2D slices of higher-dimensional systems."
        )
    )

    parser.add_argument("--data_path", type=str, required=True, help="Dataset directory/base path.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--filename", type=str, default=None)

    parser.add_argument("--x_dim", type=int, default=0, help="State dimension used as horizontal axis.")
    parser.add_argument("--y_dim", type=int, default=1, help="State dimension used as vertical axis.")
    parser.add_argument("--components", type=str, default="auto", help="Velocity components to plot: auto, all, or e.g. '0,1,2'.")
    parser.add_argument("--all_plane_pairs", action="store_true", help="Plot all unordered 2D state-coordinate planes.")

    parser.add_argument("--xlim", type=str, default=None, help="Optional global x-axis limit as 'min,max'.")
    parser.add_argument("--ylim", type=str, default=None, help="Optional global y-axis limit as 'min,max'.")
    parser.add_argument("--grid_resolution", type=int, default=300)
    parser.add_argument("--pad_fraction", type=float, default=0.08)

    parser.add_argument(
        "--slice_values",
        type=str,
        default=None,
        help=(
            "Optional comma-separated full state used for non-plotted dimensions. "
            "Example for Lorenz: '0,0,25'. If omitted, mean state from data is used."
        ),
    )

    parser.add_argument("--linthresh", type=float, default=1.0)
    parser.add_argument("--subtitle", type=str, default=None, help="Optional subtitle shown below the main title.")
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    split_path = resolve_split_npz_path(args.data_path, args.split)
    data = np.load(split_path, allow_pickle=True)
    X = data["X"]
    system = str(data["system"])
    state_dim = X.shape[-1]

    if args.x_dim == args.y_dim:
        raise ValueError("x_dim and y_dim must be different.")
    if args.x_dim < 0 or args.x_dim >= state_dim:
        raise ValueError(f"x_dim={args.x_dim} out of range for state_dim={state_dim}.")
    if args.y_dim < 0 or args.y_dim >= state_dim:
        raise ValueError(f"y_dim={args.y_dim} out of range for state_dim={state_dim}.")

    flat = X.reshape(-1, state_dim)
    base_state = build_base_state(flat, state_dim, args.slice_values)
    f_true = build_true_dynamics_from_dataset(split_path)

    plane_pairs = get_plane_pairs(state_dim, args.x_dim, args.y_dim, args.all_plane_pairs)
    components = parse_components(args.components, state_dim, args.x_dim, args.y_dim)

    n_derivative_panels = len(components)
    n_panels_per_row = n_derivative_panels + 1
    n_rows = len(plane_pairs)

    fig_width = max(5.5 * n_panels_per_row, 12.0)
    fig_height = max(4.9 * n_rows + 0.5, 5.4)

    fig, axes = plt.subplots(
        n_rows,
        n_panels_per_row,
        figsize=(fig_width, fig_height),
        constrained_layout=False,
        squeeze=False,
    )

    for row_idx, (x_dim, y_dim) in enumerate(plane_pairs):
        xlim = parse_lim(args.xlim) or infer_axis_limits(flat[:, x_dim], args.pad_fraction)
        ylim = parse_lim(args.ylim) or infer_axis_limits(flat[:, y_dim], args.pad_fraction)

        XX, YY, component_fields, speed = compute_plane_fields(
            f_true=f_true,
            base_state=base_state,
            state_dim=state_dim,
            x_dim=x_dim,
            y_dim=y_dim,
            xlim=xlim,
            ylim=ylim,
            grid_resolution=args.grid_resolution,
            components=components,
        )

        nullcline_present = {}

        for panel_idx, comp in enumerate(components):
            ax = axes[row_idx, panel_idx]
            field = component_fields[comp]

            norm, max_abs = make_symlog_norm(field, args.linthresh)
            mesh = ax.pcolormesh(
                XX,
                YY,
                field,
                shading="auto",
                cmap="RdBu_r",
                norm=norm,
            )

            color = NULLCLINE_COLORS[comp % len(NULLCLINE_COLORS)]
            nullcline_present[comp] = contour_if_available(
                ax,
                XX,
                YY,
                field,
                color=color,
                linewidth=2.0,
            )

            ax.set_title(rf"Plane $(x_{x_dim + 1},x_{y_dim + 1})$: $\dot x_{comp + 1}$")
            add_three_tick_colorbar(
                fig,
                mesh,
                ax,
                label=rf"$\dot x_{comp + 1}$ value",
                kind="signed",
                max_abs=max_abs,
            )
            add_nullcline_legend(ax, [comp], nullcline_present, loc="best")

        ax_speed = axes[row_idx, -1]
        speed_norm, speed_vmin, speed_vmax = make_speed_norm(speed)
        mesh_speed = ax_speed.pcolormesh(
            XX,
            YY,
            speed,
            shading="auto",
            cmap="viridis",
            norm=speed_norm,
        )

        for comp in components:
            field = component_fields[comp]
            color = NULLCLINE_COLORS[comp % len(NULLCLINE_COLORS)]
            nullcline_present[comp] = contour_if_available(
                ax_speed,
                XX,
                YY,
                field,
                color=color,
                linewidth=1.7,
                alpha=0.9,
            )

        ax_speed.set_title(rf"Plane $(x_{x_dim + 1},x_{y_dim + 1})$: $\|\dot x\|$")
        add_three_tick_colorbar(
            fig,
            mesh_speed,
            ax_speed,
            label=r"Speed $\|\dot x\|$",
            kind="positive",
            vmin=speed_vmin,
            vmax=speed_vmax,
        )
        add_nullcline_legend(ax_speed, components, nullcline_present, loc="best")

        for col_idx in range(n_panels_per_row):
            ax = axes[row_idx, col_idx]
            ax.set_xlabel(rf"$x_{x_dim + 1}$")
            ax.set_ylabel(rf"$x_{y_dim + 1}$")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.2)

    if args.all_plane_pairs:
        plane_note = "all 2D coordinate planes"
    else:
        plane_note = rf"plane $(x_{args.x_dim + 1},x_{args.y_dim + 1})$"

    slice_note = ""
    if state_dim > 2:
        fixed_text = ", ".join([rf"$x_{i + 1}={base_state[i]:.3g}$" for i in range(state_dim)])
        slice_note = f" | base state: {fixed_text}"

    fig.suptitle(
        f"{system}: true vector-field diagnostics ({plane_note}){slice_note}",
        fontsize=16,
        y=0.965,
    )

    if args.subtitle:
        fig.text(
            0.5,
            0.925,
            args.subtitle,
            ha="center",
            va="top",
            fontsize=12,
        )

    fig.subplots_adjust(top=0.80)

    outdir = args.outdir or os.path.join("data", "figures", "vector_fields", system)
    os.makedirs(outdir, exist_ok=True)

    filename = args.filename or f"{system}_vector_field_diagnostics.png"
    save_path = os.path.join(outdir, filename)

    fig.savefig(save_path, dpi=200)
    print(f"Saved vector-field diagnostics to: {save_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()