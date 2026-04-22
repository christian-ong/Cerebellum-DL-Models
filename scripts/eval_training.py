import argparse
import os

from src.eval.training_analysis import (
    load_losses,
    summarize_losses,
    save_training_summary_npz,
    infer_loss_file_from_model_path,
)
from src.eval.plot_training_losses import plot_training_losses

"""
Post-hoc analysis of saved training curves.

Reads the losses saved during training, summarizes convergence behavior,
and plots training and validation losses without retraining the model.
"""
'''
python -m scripts.eval_training \
  --model_path data/models/ml_dmd/vanderpol/default/model.pt
'''

def main():
    parser = argparse.ArgumentParser(description="Analyze saved training curves for a trained model run.")

    parser.add_argument("--model_path", type=str, required=True, help="Path to model.pt or model.npz")
    parser.add_argument("--loss_file", type=str, default=None, help="Optional explicit path to losses.npz")
    parser.add_argument("--ignore_first_epochs", type=int, default=0)

    args = parser.parse_args()

    loss_file = args.loss_file if args.loss_file is not None else infer_loss_file_from_model_path(args.model_path)
    if not os.path.exists(loss_file):
        raise FileNotFoundError(f"Could not find loss file: {loss_file}")

    run_dir = os.path.dirname(loss_file)
    figdir = os.path.join(run_dir, "training")
    os.makedirs(figdir, exist_ok=True)

    losses = load_losses(loss_file)
    summary = summarize_losses(losses)
    save_training_summary_npz(os.path.join(figdir, "training_summary.npz"), summary)

    plot_training_losses(
        loss_file=loss_file,
        figdir=figdir,
        ignore_first_epochs=args.ignore_first_epochs,
    )

    print(f"Saved training analysis to: {figdir}")


if __name__ == "__main__":
    main()