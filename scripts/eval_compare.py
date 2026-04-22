import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
"""
Compare saved evaluation summaries across runs.

Loads summary files from previous eval runs and compares scalar metrics
without recomputing model predictions.
"""
'''
python -m scripts.eval_compare \
  --summary_paths \
    data/figures/regression_dmd/vanderpol/default/test_summary.npz \
    data/figures/ml_dmd/vanderpol/default/test_summary.npz \
  --labels regression_dmd ml_dmd \
  --metric composite_score
'''

def load_summary(path: str):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def scalarize(x):
    arr = np.asarray(x)
    if arr.shape == ():
        return float(arr.item())
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    raise ValueError("Expected scalar-like array.")

def infer_group_name(summary_paths):
    """
    Infer a dataset/system grouping name from summary paths.

    Expected summary path structure:
    data/figures/<model>/<system>/<run_name>/test_summary.npz
    """
    groups = []
    for path in summary_paths:
        norm = os.path.normpath(path)
        parts = norm.split(os.sep)
        if len(parts) >= 4:
            # .../<model>/<system>/<run_name>/test_summary.npz
            groups.append(parts[-3])
        else:
            groups.append("unknown")

    unique_groups = sorted(set(groups))
    if len(unique_groups) == 1:
        return unique_groups[0]
    return "mixed"

def main():
    parser = argparse.ArgumentParser(description="Compare scalar metrics across saved eval summaries.")
    parser.add_argument("--summary_paths", type=str, nargs="+", required=True)
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--metric", type=str, default="composite_score")
    parser.add_argument("--outdir", type=str, default="data/figures/comparisons")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--group_name", type=str, default=None, help="Optional subfolder name inside the comparison output directory.")

    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.summary_paths):
        raise ValueError("If provided, --labels must have same length as --summary_paths")

    group_name = args.group_name if args.group_name is not None else infer_group_name(args.summary_paths)
    target_outdir = os.path.join(args.outdir, group_name)
    os.makedirs(target_outdir, exist_ok=True)

    labels = args.labels if args.labels is not None else [os.path.basename(os.path.dirname(p)) for p in args.summary_paths]

    values = []
    for path in args.summary_paths:
        summary = load_summary(path)
        if args.metric not in summary:
            raise KeyError(f"Metric '{args.metric}' not found in {path}")
        values.append(scalarize(summary[args.metric]))

    values = np.asarray(values, dtype=float)

    # Sort from best (lowest) to worst
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = values[order]

    # Print terminal summary
    print(f"\nComparison metric: {args.metric} (lower is better)")
    for rank, (label, value) in enumerate(zip(labels, values), start=1):
        print(f"  {rank}. {label}: {value:.6e}")

    print(f"Best model: {labels[0]} = {values[0]:.6e}")

    # Use log scale automatically if spread is large
    positive_values = values[values > 0]
    use_log = False
    if len(positive_values) >= 2:
        ratio = np.max(positive_values) / np.min(positive_values)
        use_log = ratio >= 50.0

    fig_height = max(3.5, 0.8 * len(labels) + 1.5)
    plt.figure(figsize=(9, fig_height))

    y = np.arange(len(labels))
    bars = plt.barh(y, values)

    plt.yticks(y, labels)
    plt.xlabel(f"{args.metric} (lower is better)")
    plt.title(args.title if args.title else f"Model comparison: {args.metric}")

    if use_log:
        plt.xscale("log")

    # Put best model at the top
    plt.gca().invert_yaxis()

    # Annotate values
    x_min = np.min(positive_values) if len(positive_values) > 0 else 0.0
    x_max = np.max(values) if len(values) > 0 else 1.0
    x_offset = 0.06 * x_max if not use_log else 1.15

    for bar, value in zip(bars, values):
        y_text = bar.get_y() + bar.get_height() / 2
        if use_log:
            x_text = value * x_offset
        else:
            x_text = value + x_offset
        plt.text(x_text, y_text, f"{value:.2e}", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(target_outdir, f"compare_{args.metric}.png"), dpi=200)    
    plt.close()

    np.savez(
        os.path.join(target_outdir, f"compare_{args.metric}.npz"),
        labels=np.array(labels, dtype=object),
        values=values,
        metric=np.array(args.metric, dtype=object),
    )

    print(f"Saved comparison outputs to: {target_outdir}")
    print(f"Saved figure: {os.path.join(target_outdir, f'compare_{args.metric}.png')}")


if __name__ == "__main__":
    main()