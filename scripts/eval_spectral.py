import argparse
import os
import numpy as np

from src.eval.eval_runner import MODEL_CHOICES, prepare_eval_context, save_metadata_json
from src.eval.spectral import (
    extract_transition_matrix,
    extract_eigendecomposition,
    eigenvalue_summary,
    save_spectral_summary_npz,
    maybe_extract_dt,
)
from src.eval.plot_eigenvalues import plot_eigenvalues
from src.eval.plot_matrices import plot_transition_matrix

"""
Spectral analysis for a trained model.

Extracts transition-matrix and eigenvalue information when available,
saves a spectral summary, and produces eigenvalue and matrix plots.
Use this to inspect learned modes and stability structure.
"""
'''
python -m scripts.eval_spectral \
  --model regression_dmd \
  --data_path data/trajectories/nonlinear/vanderpol \
  --model_path data/models/regression_dmd/vanderpol/default/model.npz
'''

def main():
    parser = argparse.ArgumentParser(description="Spectral / mode analysis for trained models.")

    parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    parser.add_argument("--plot_eigs", action="store_true")
    parser.add_argument("--plot_matrix", action="store_true")
    parser.add_argument("--matrix_threshold", type=float, default=1e-2)
    parser.add_argument(
        "--reuse_if_exists",
        action="store_true",
        help="If spectral outputs already exist, skip recomputation and exit.",
    )

    args = parser.parse_args()

    if not args.plot_eigs and not args.plot_matrix:
        args.plot_eigs = True
        args.plot_matrix = True

    ctx = prepare_eval_context(
        args=args,
        split=args.split,
        subdir="spectral",
        need_cache=False,
    )

    spectral_summary_path = os.path.join(ctx.figdir, "spectral_summary.npz")
    if args.reuse_if_exists and os.path.exists(spectral_summary_path):
        print(f"[eval_spectral] Found existing spectral summary: {spectral_summary_path}")
        print("[eval_spectral] --reuse_if_exists set, skipping recomputation.")
        return
    elif os.path.exists(spectral_summary_path):
        print(f"[eval_spectral] Found existing spectral summary but recomputing: {spectral_summary_path}")
    else:
        print(f"[eval_spectral] No spectral summary found at: {spectral_summary_path}. Recomputing...")

    K = extract_transition_matrix(args.model, ctx.model, ctx.extras)
    eigvals, eigvecs = extract_eigendecomposition(args.model, ctx.model, ctx.extras)
    dt = maybe_extract_dt(ctx.data)

    summary = eigenvalue_summary(eigvals, dt=dt)
    save_spectral_summary_npz(
        spectral_summary_path,
        matrix=K,
        eigvals=eigvals,
        eigvecs=eigvecs,
        extra_summary=summary,
    )

    if args.plot_eigs and eigvals is not None:
        plot_eigenvalues(eigvals, ctx.figdir)

    if args.plot_matrix and K is not None:
        plot_transition_matrix(
            model=ctx.model,
            model_name=args.model,
            figdir=ctx.figdir,
            threshold=args.matrix_threshold,
            matrix=K,
        )

    save_metadata_json(
        ctx,
        os.path.join(ctx.figdir, "metadata.json"),
        extra={
            "has_transition_matrix": K is not None,
            "has_eigenvalues": eigvals is not None,
            "dt": dt,
        },
    )

    print(f"Saved spectral analysis to: {ctx.figdir}")


if __name__ == "__main__":
    main()