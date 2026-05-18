import os
import numpy as np
import matplotlib.pyplot as plt


SYSTEM = "vanderpol"
OUTDIR = f"data/figures/regression_dmd/{SYSTEM}/delay_depth_ablation"
os.makedirs(OUTDIR, exist_ok=True)


RUNS = {
    "Raw delay": [
        (5, "ablation_rawdelay_q5_r10"),
        (10, "ablation_rawdelay_q10_r20"),
        (25, "ablation_rawdelay_q25_r40"),
        (50, "ablation_rawdelay_q50_r40"),
        (100, "ablation_rawdelay_q100_r40"),
        (150, "ablation_rawdelay_q150_r40"),
        (200, "ablation_rawdelay_q200_r40"),
    ],
    "Hankel-SVD delay": [
        (5, "ablation_hsvd_q5_hr10_r10"),
        (10, "ablation_hsvd_q10_hr20_r20"),
        (25, "ablation_hsvd_q25_hr40_r40"),
        (50, "ablation_hsvd_q50_hr40_r40"),
        (100, "ablation_hsvd_q100_hr40_r40"),
        (150, "ablation_hsvd_q150_hr40_r40"),
        (200, "ablation_hsvd_q200_hr40_r40"),
    ],
}


def get_metric_at(summary, x_key, y_key, target):
    xs = np.asarray(summary[x_key])
    ys = np.asarray(summary[y_key])
    idx = np.where(xs == target)[0]
    if len(idx) == 0:
        raise ValueError(f"Could not find {target} in {x_key}: {xs}")
    return float(ys[idx[0]])


def load_results(runs):
    xs = []
    horizon100 = []
    rollout100 = []
    one_step = []

    for q, name in runs:
        path = f"data/figures/regression_dmd/{SYSTEM}/{name}/test_summary.npz"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing summary: {path}")

        s = np.load(path)

        xs.append(q)
        one_step.append(float(np.asarray(s["one_step_rmse"])))
        horizon100.append(get_metric_at(s, "horizons", "horizon_rmse", 100))
        rollout100.append(get_metric_at(s, "rollout_horizons", "rollout_rmse", 100))

    return np.asarray(xs), np.asarray(one_step), np.asarray(horizon100), np.asarray(rollout100)


def plot_single_curve(label, runs, filename):
    x, one_step, horizon100, rollout100 = load_results(runs)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(x, one_step, marker="o", label="One-step RMSE")
    ax.plot(x, horizon100, marker="o", label="Terminal horizon RMSE, h=100")
    ax.plot(x, rollout100, marker="o", label="Full rollout RMSE, h=100")

    ax.set_yscale("log")
    ax.set_xlabel("Delay depth q")
    ax.set_ylabel("RMSE")
    ax.set_title(label)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, filename), dpi=200)
    plt.close(fig)


def plot_comparison():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for label, runs in RUNS.items():
        x, _, _, rollout100 = load_results(runs)
        ax.plot(x, rollout100, marker="o", label=f"{label}, rollout h=100")

    ax.set_yscale("log")
    ax.set_xlabel("Delay depth q")
    ax.set_ylabel("Rollout-100 RMSE")
    ax.set_title("Effect of delay depth on model performance")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "delay_depth_ablation_comparison.png"), dpi=200)
    plt.close(fig)


def main():
    plot_single_curve("Raw delay EDMD", RUNS["Raw delay"], "raw_delay_ablation.png")
    plot_single_curve("Hankel-SVD delay EDMD", RUNS["Hankel-SVD delay"], "hankel_svd_ablation.png")
    plot_comparison()
    print(f"Saved plots to {OUTDIR}")


if __name__ == "__main__":
    main()