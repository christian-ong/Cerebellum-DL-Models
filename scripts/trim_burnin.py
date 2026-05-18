import argparse
import os
from typing import Dict, Optional

import numpy as np

from src.data_generation.load_data import resolve_split_npz_path


def parse_splits(text: str):
    return [s.strip() for s in text.split(",") if s.strip()]


def infer_burn_steps(*, data: Dict, burn_in: Optional[float], burn_steps: Optional[int]) -> int:
    if burn_steps is not None:
        return int(burn_steps)

    if burn_in is None:
        raise ValueError("Provide either --burn_in or --burn_steps.")

    if "dt" not in data.files:
        raise ValueError("Cannot use --burn_in because dataset has no 'dt'. Use --burn_steps instead.")

    dt = float(data["dt"])
    if dt <= 0:
        raise ValueError(f"Invalid dt={dt}.")

    return int(round(float(burn_in) / dt))


def make_trimmed_time(data, burn_steps: int, n_steps_after_trim: int):
    if "t" in data.files:
        t_old = np.asarray(data["t"])
        t_trim = t_old[burn_steps:] - t_old[burn_steps]
        return t_trim

    if "dt" not in data.files:
        raise ValueError("Dataset has no 't' and no 'dt', so time vector cannot be reconstructed.")

    dt = float(data["dt"])
    return np.arange(n_steps_after_trim, dtype=float) * dt


def infer_new_x0(X_trim: np.ndarray) -> np.ndarray:
    # X can be:
    #   (T, d)
    #   (T, n_traj, d)
    return np.asarray(X_trim[0]).copy()


def trim_split(
    *,
    data_path: str,
    split: str,
    out_path: str,
    burn_in: Optional[float],
    burn_steps_arg: Optional[int],
    min_remaining_steps: int,
    overwrite: bool,
    skip_missing: bool,
):
    try:
        in_file = resolve_split_npz_path(data_path, split)
    except FileNotFoundError:
        if skip_missing:
            print(f"Skipping missing split: {split}")
            return
        raise

    data = np.load(in_file, allow_pickle=True)
    X = np.asarray(data["X"])

    burn_steps = infer_burn_steps(data=data, burn_in=burn_in, burn_steps=burn_steps_arg)

    if burn_steps <= 0:
        raise ValueError("Burn-in must remove at least one time step.")

    if burn_steps >= X.shape[0] - min_remaining_steps:
        raise ValueError(
            f"Burn-in removes too much data for split='{split}'. "
            f"burn_steps={burn_steps}, trajectory length={X.shape[0]}, "
            f"min_remaining_steps={min_remaining_steps}."
        )

    X_trim = X[burn_steps:]
    t_trim = make_trimmed_time(data, burn_steps, X_trim.shape[0])

    payload = {}
    for key in data.files:
        if key == "X":
            payload[key] = X_trim
        elif key == "t":
            payload[key] = t_trim
        elif key == "T":
            payload[key] = np.array(float(t_trim[-1]))
        elif key == "x0":
            payload[key] = infer_new_x0(X_trim)
        else:
            payload[key] = data[key]

    if "t" not in payload:
        payload["t"] = t_trim

    if "T" not in payload:
        payload["T"] = np.array(float(t_trim[-1]))

    if "x0" not in payload:
        payload["x0"] = infer_new_x0(X_trim)

    payload["burn_in_removed_steps"] = np.array(burn_steps)
    if "dt" in data.files:
        payload["burn_in_removed_time"] = np.array(burn_steps * float(data["dt"]))

    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, f"{split}.npz")

    if os.path.exists(out_file) and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {out_file}. "
            "Use --overwrite to replace it."
        )

    np.savez(out_file, **payload)

    print(
        f"Saved {out_file} | "
        f"X: {X.shape} -> {X_trim.shape} | "
        f"removed {burn_steps} steps"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Trim a fixed burn-in period from train/val/test trajectory datasets."
    )

    parser.add_argument("--data_path", required=True, help="Input dataset directory/base path.")
    parser.add_argument("--out_path", required=True, help="Output dataset directory.")
    parser.add_argument("--burn_in", type=float, default=None, help="Burn-in time to remove.")
    parser.add_argument("--burn_steps", type=int, default=None, help="Burn-in steps to remove.")
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--min_remaining_steps", type=int, default=3)
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.burn_in is None and args.burn_steps is None:
        raise ValueError("Provide either --burn_in or --burn_steps.")
    if args.burn_in is not None and args.burn_steps is not None:
        raise ValueError("Use either --burn_in or --burn_steps, not both.")

    for split in parse_splits(args.splits):
        trim_split(
            data_path=args.data_path,
            split=split,
            out_path=args.out_path,
            burn_in=args.burn_in,
            burn_steps_arg=args.burn_steps,
            min_remaining_steps=args.min_remaining_steps,
            overwrite=args.overwrite,
            skip_missing=args.skip_missing,
        )


if __name__ == "__main__":
    main()