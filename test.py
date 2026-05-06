import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.ml_dmd_free import ML_DMD
from src.models.ml_linear_dynamics import ML_LinearDynamics


# --------------------------------------------------
# User settings
# --------------------------------------------------

model_path = "data/models/ml_dmd/vanderpol/default/model.pt"

grid_min = -2.0
grid_max = 2.0
grid_n = 101

n_modes_to_plot = 8
n_traj_overlay = 10
traj_steps = 150

# Quality scoring controls
n_eval_traj = 14
eval_steps = 90

# Split threshold controls: "auto", "median", or "quantile"
split_method = "auto"
split_quantile = 0.6

# Shared panel color scaling
color_percentiles = (2.0, 98.0)

# Notebook-inspired look
# "dark" resembles Duffing.ipynb style best.
theme = "dark"  # "dark" or "light"
colormap_name = "inferno"  # thermal-like colormap

# Figure export
save_figures = False
save_dir = "data/figures/koopman_viz"
save_dpi = 220

# Optional fixed points per system for visual context.
# Set fixed_points_override to a (k, 2) array to force custom points.
fixed_points_override = None
SYSTEM_FIXED_POINTS = {
    "vanderpol": np.array([[0.0, 0.0]], dtype=np.float32),
    "duffing": np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
}


def normalize_vector(x, lb=0.35):
    x = np.asarray(x)
    dx = np.max(x) - np.min(x)
    if dx < 1e-12:
        return np.full_like(x, 0.5, dtype=np.float64)
    return (((x - np.min(x)) / dx) + lb) / (1.0 + lb)


def to_unit_interval(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min + eps)


def compute_grid(n_items, max_cols=3):
    ncols = min(max_cols, max(1, int(np.ceil(np.sqrt(n_items)))))
    nrows = int(np.ceil(n_items / ncols))
    return nrows, ncols


def apply_theme(theme_name="dark"):
    if theme_name == "dark":
        plt.rcParams.update(
            {
                "figure.facecolor": "#000000",
                "axes.facecolor": "#000000",
                "axes.edgecolor": "#e5e7eb",
                "axes.labelcolor": "#e5e7eb",
                "xtick.color": "#e5e7eb",
                "ytick.color": "#e5e7eb",
                "text.color": "#e5e7eb",
                "grid.color": "#6b7280",
                "grid.alpha": 0.35,
                "axes.grid": True,
                "grid.linewidth": 0.7,
            }
        )
    else:
        plt.rcParams.update(
            {
                "figure.facecolor": "#ffffff",
                "axes.facecolor": "#ffffff",
                "axes.edgecolor": "#111827",
                "axes.labelcolor": "#111827",
                "xtick.color": "#111827",
                "ytick.color": "#111827",
                "text.color": "#111827",
                "grid.color": "#d1d5db",
                "grid.alpha": 0.6,
                "axes.grid": True,
                "grid.linewidth": 0.7,
            }
        )


def setup_axis(ax):
    ax.set_xlim(grid_min, grid_max)
    ax.set_ylim(grid_min, grid_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def get_fixed_points(system_name):
    if fixed_points_override is not None:
        arr = np.asarray(fixed_points_override, dtype=np.float32)
        return arr.reshape(-1, 2)
    return SYSTEM_FIXED_POINTS.get(system_name, np.empty((0, 2), dtype=np.float32))


def plot_fixed_points(ax, fixed_points):
    if fixed_points.shape[0] == 0:
        return
    ax.scatter(
        fixed_points[:, 0],
        fixed_points[:, 1],
        marker="x",
        color="#ef4444",
        s=80,
        linewidths=1.8,
        zorder=4,
    )


def save_figure(fig, name):
    if not save_figures:
        return
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{name}.png")
    fig.savefig(out_path, dpi=save_dpi, bbox_inches="tight")


def choose_split_threshold(values, method="auto", quantile=0.6):
    vals = np.asarray(values, dtype=np.float64)
    vals = np.clip(vals, 0.0, 1.0)

    if method == "median":
        return float(np.median(vals))
    if method == "quantile":
        return float(np.quantile(vals, quantile))

    # Auto: attempt bimodal valley split; fallback to median.
    hist, edges = np.histogram(vals, bins=80, range=(0.0, 1.0))
    smooth = np.convolve(hist, np.array([1, 2, 3, 2, 1]), mode="same")

    peaks = np.where((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:]))[0] + 1
    if len(peaks) >= 2:
        top2 = peaks[np.argsort(smooth[peaks])[-2:]]
        i1, i2 = int(np.min(top2)), int(np.max(top2))
        if i2 - i1 > 2:
            valley = i1 + int(np.argmin(smooth[i1:i2 + 1]))
            return float(0.5 * (edges[valley] + edges[valley + 1]))

    return float(np.median(vals))


def build_model_from_checkpoint(ckpt):
    train_args = ckpt["train_args"]
    model_name = ckpt.get("model", "ml_dmd")

    if model_name == "ml_dmd":
        model = ML_DMD(
            state_dim=ckpt["state_dim"],
            expansion_degree=train_args["expansion_degree"],
            bias=str(train_args.get("bias", "true")).lower() == "true",
            sine_cosine_expansion=str(train_args.get("sine_cosine_expansion", "false")).lower() == "true",
            expansion_type=train_args["expansion_type"],
            system=ckpt["system"],
        )
    elif model_name == "ml_lineardynamics":
        model = ML_LinearDynamics(
            state_dim=ckpt["state_dim"],
            expansion_degree=train_args["expansion_degree"],
            bias=str(train_args.get("bias", "true")).lower() == "true",
            sine_cosine_expansion=str(train_args.get("sine_cosine_expansion", "false")).lower() == "true",
            expansion_type=train_args["expansion_type"],
            system=ckpt["system"],
        )
    else:
        raise ValueError(f"Unsupported model type in checkpoint: {model_name}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_name


def extract_koopman_operator(model):
    if hasattr(model, "K"):
        return model.K.weight.detach().cpu().numpy()
    if hasattr(model, "Phi") and hasattr(model, "Lambda"):
        phi_param = model.Phi.detach().cpu().numpy()
        lambda_param = model.Lambda.detach().cpu().numpy()
        return phi_param @ lambda_param @ np.linalg.pinv(phi_param)
    raise ValueError("Model does not expose either K or (Phi, Lambda).")


# --------------------------------------------------
# Load model checkpoint and rebuild model
# --------------------------------------------------

ckpt = torch.load(model_path, map_location="cpu")
model, model_name = build_model_from_checkpoint(ckpt)

apply_theme(theme)

z_scale = model.z_scale.detach().cpu().numpy()
system_name = ckpt["system"]
fixed_points = get_fixed_points(system_name)

print("Loaded model:", system_name)
print("Model type:", model_name)


# --------------------------------------------------
# Build Koopman eigensystem
# --------------------------------------------------

K = extract_koopman_operator(model)

eigvals, _ = np.linalg.eig(K)
eigvals_left, W = np.linalg.eig(K.T)


# --------------------------------------------------
# Plot eigenvalue spectrum
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(6.2, 6.2))
point_color = "#f9fafb" if theme == "dark" else "#111827"
circle_color = "#9ca3af" if theme == "dark" else "#374151"
axis_color = "#f3f4f6" if theme == "dark" else "#111827"

ax.scatter(eigvals.real, eigvals.imag, color=point_color, s=28)

theta = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), "--", color=circle_color, linewidth=1.2)

ax.axhline(0, color=axis_color, linewidth=0.9)
ax.axvline(0, color=axis_color, linewidth=0.9)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Eigenvalue Spectrum")
ax.set_xlabel("Real")
ax.set_ylabel("Imag")
fig.tight_layout()
save_figure(fig, "eigenvalue_spectrum")
plt.show()


# --------------------------------------------------
# Evaluate lifted states on grid
# --------------------------------------------------

x = np.linspace(grid_min, grid_max, grid_n)
y = np.linspace(grid_min, grid_max, grid_n)
X, Y = np.meshgrid(x, y)
points = np.stack([X.flatten(), Y.flatten()], axis=1).astype(np.float32)

with torch.no_grad():
    z_raw = model.expand(torch.tensor(points, dtype=torch.float32)).cpu().numpy()

z = z_raw / z_scale
phi_grid = z @ W
print("Lifted grid shape:", z.shape)


# --------------------------------------------------
# Mode quality scoring
# --------------------------------------------------

rng = np.random.default_rng(0)
num_modes_total = W.shape[1]
residual_accum = np.zeros(num_modes_total, dtype=np.float64)

for _ in range(n_eval_traj):
    x0 = rng.uniform(grid_min, grid_max, size=(1, model.state_dim)).astype(np.float32)
    xt = torch.tensor(x0, dtype=torch.float32)

    traj = [x0.flatten()]
    with torch.no_grad():
        for _ in range(eval_steps):
            xt = model(xt)
            traj.append(xt.cpu().numpy().flatten())

    traj = np.asarray(traj, dtype=np.float32)
    with torch.no_grad():
        z_roll_raw = model.expand(torch.tensor(traj, dtype=torch.float32)).cpu().numpy()

    z_roll = z_roll_raw / z_scale
    phi_roll = z_roll @ W

    lhs = phi_roll[1:, :]
    rhs = phi_roll[:-1, :] * eigvals_left[None, :]

    num = np.linalg.norm(lhs - rhs, axis=0)
    den = np.linalg.norm(phi_roll[:-1, :], axis=0) + 1e-12
    residual_accum += num / den

residual_mean = residual_accum / max(1, n_eval_traj)

spatial_std = np.std(np.real(phi_grid), axis=0)
stability = np.exp(-np.abs(np.abs(eigvals_left) - 1.0) / 0.15)

residual_score = 1.0 - to_unit_interval(residual_mean)
spatial_score = to_unit_interval(spatial_std)
stability_score = to_unit_interval(stability)

mode_score = 0.50 * residual_score + 0.30 * stability_score + 0.20 * spatial_score
mode_order = np.argsort(mode_score)[::-1]
mode_ids = mode_order[: min(n_modes_to_plot, len(mode_order))]

print("\nTop modes by quality score:")
for rank, idx in enumerate(mode_ids, start=1):
    lam = eigvals_left[idx]
    print(
        f"{rank:2d}. mode={idx:2d}  score={mode_score[idx]:.3f}  "
        f"res={residual_mean[idx]:.3e}  |lambda|={abs(lam):.4f}"
    )


# --------------------------------------------------
# Shared color scaling across selected mode panels
# --------------------------------------------------

sel_real = np.real(phi_grid[:, mode_ids])
global_vmin, global_vmax = np.percentile(sel_real, color_percentiles)
if global_vmax - global_vmin < 1e-12:
    global_vmin = float(np.min(sel_real))
    global_vmax = float(np.max(sel_real) + 1e-6)


def shared_color_values(values):
    unit = (values - global_vmin) / (global_vmax - global_vmin + 1e-12)
    unit = np.clip(unit, 0.0, 1.0)
    return normalize_vector(unit, lb=0.35)


# --------------------------------------------------
# Plot top Koopman eigenfunctions (ranked by quality)
# --------------------------------------------------

nrows, ncols = compute_grid(len(mode_ids), max_cols=3)
fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.0 * nrows))
axes = np.atleast_1d(axes).ravel()

scatter_size = max(2.2, 26000.0 / (grid_n * grid_n))

for i, mode_idx in enumerate(mode_ids):
    ax = axes[i]
    eigfunc_real = np.real(phi_grid[:, mode_idx])
    color_vals = shared_color_values(eigfunc_real)

    ax.scatter(
        points[:, 0],
        points[:, 1],
        c=color_vals,
        cmap=colormap_name,
        vmin=0.0,
        vmax=1.0,
        s=scatter_size,
        linewidths=0,
    )

    lam = eigvals_left[mode_idx]
    ax.set_title(
        f"m{mode_idx}  score={mode_score[mode_idx]:.2f}  |lambda|={abs(lam):.3f}",
        fontsize=10,
    )
    setup_axis(ax)
    plot_fixed_points(ax, fixed_points)

for j in range(len(mode_ids), len(axes)):
    axes[j].axis("off")

fig.suptitle("Koopman Eigenfunctions (Quality-Ranked)", y=0.995)
fig.tight_layout()
save_figure(fig, "ranked_eigenfunctions")
plt.show()


# --------------------------------------------------
# Leading eigenfunction with trajectory overlay
# --------------------------------------------------

lead_mode = mode_ids[0]
lead_complex = phi_grid[:, lead_mode]
lead_real = np.real(lead_complex)
lead_val = shared_color_values(lead_real)

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(
    points[:, 0],
    points[:, 1],
    c=lead_val,
    cmap=colormap_name,
    vmin=0.0,
    vmax=1.0,
    s=scatter_size,
    linewidths=0,
)

traj_color = "#ffffff" if theme == "dark" else "#111827"
for _ in range(n_traj_overlay):
    x0 = rng.uniform(grid_min, grid_max, size=(1, model.state_dim)).astype(np.float32)
    xt = torch.tensor(x0, dtype=torch.float32)

    traj = [x0.flatten()]
    with torch.no_grad():
        for _ in range(traj_steps):
            xt = model(xt)
            traj.append(xt.cpu().numpy().flatten())

    traj = np.asarray(traj)
    ax.plot(traj[:, 0], traj[:, 1], "-", color=traj_color, alpha=0.35, linewidth=1.0)

lam = eigvals_left[lead_mode]
ax.set_title(f"Leading Eigenfunction (lambda={lam.real:.3f}{lam.imag:+.3f}j)")
setup_axis(ax)
plot_fixed_points(ax, fixed_points)
fig.tight_layout()
save_figure(fig, "leading_with_trajectories")
plt.show()


# --------------------------------------------------
# Split leading eigenfunction into two regions
# --------------------------------------------------

split_thr = choose_split_threshold(lead_val, method=split_method, quantile=split_quantile)
mask_hi = lead_val > split_thr
mask_lo = ~mask_hi

fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
titles = [
    f"Leading Eigenfunction > {split_thr:.2f}",
    f"Leading Eigenfunction <= {split_thr:.2f}",
]
masks = [mask_hi, mask_lo]

for ax, title, mask in zip(axes, titles, masks):
    ax.scatter(
        points[mask, 0],
        points[mask, 1],
        c=lead_val[mask],
        cmap=colormap_name,
        vmin=0.0,
        vmax=1.0,
        s=scatter_size,
        linewidths=0,
    )

    for _ in range(max(1, n_traj_overlay // 2)):
        x0 = rng.uniform(grid_min, grid_max, size=(1, model.state_dim)).astype(np.float32)
        xt = torch.tensor(x0, dtype=torch.float32)
        traj = [x0.flatten()]
        with torch.no_grad():
            for _ in range(traj_steps):
                xt = model(xt)
                traj.append(xt.cpu().numpy().flatten())
        traj = np.asarray(traj)
        ax.plot(traj[:, 0], traj[:, 1], "-", color=traj_color, alpha=0.33, linewidth=0.9)

    setup_axis(ax)
    plot_fixed_points(ax, fixed_points)
    ax.set_title(title)

fig.tight_layout()
save_figure(fig, "leading_split")
plt.show()


# --------------------------------------------------
# Magnitude and phase view for leading mode
# --------------------------------------------------

lead_mag = normalize_vector(np.abs(lead_complex))
lead_phase = normalize_vector(np.angle(lead_complex))

fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))

axes[0].scatter(
    points[:, 0],
    points[:, 1],
    c=lead_mag,
    cmap=colormap_name,
    vmin=0.0,
    vmax=1.0,
    s=scatter_size,
    linewidths=0,
)
axes[0].set_title("|phi| for leading mode")
setup_axis(axes[0])
plot_fixed_points(axes[0], fixed_points)

axes[1].scatter(
    points[:, 0],
    points[:, 1],
    c=lead_phase,
    cmap=colormap_name,
    vmin=0.0,
    vmax=1.0,
    s=scatter_size,
    linewidths=0,
)
axes[1].set_title("arg(phi) for leading mode")
setup_axis(axes[1])
plot_fixed_points(axes[1], fixed_points)

fig.tight_layout()
save_figure(fig, "leading_mag_phase")
plt.show()


# --------------------------------------------------
# Quantitative mode summary
# --------------------------------------------------

top_k = min(12, len(mode_order))
summary_modes = mode_order[:top_k]

fig, ax = plt.subplots(figsize=(10.5, 4.8))
ax.bar(np.arange(top_k), mode_score[summary_modes], color="#f59e0b", alpha=0.9)
ax.set_xticks(np.arange(top_k))
ax.set_xticklabels([f"m{m}" for m in summary_modes], rotation=0)
ax.set_ylabel("mode quality score")
ax.set_xlabel("mode index")
ax.set_title("Mode Ranking Summary")

for i, m in enumerate(summary_modes):
    ax.text(i, mode_score[m] + 0.01, f"|l|={abs(eigvals_left[m]):.2f}\nr={residual_mean[m]:.1e}", ha="center", va="bottom", fontsize=8)

fig.tight_layout()
save_figure(fig, "mode_summary")
plt.show()
