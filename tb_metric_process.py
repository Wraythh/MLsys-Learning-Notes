#!/usr/bin/env python
"""
Compute the average of the first 200 scalar values for tag 'perf/tokens/s'
in a TensorBoard log directory.

Usage:
    python avg_tokens_per_sec.py \
        --logdir /data/vjuicefs_ai_gpt_nlp/public_data/11171634/tensorboard/vivorl_v1_0_benchmark/benchmark_baseline_0701_n32 \
        --tag perf/tokens/s \
        --steps 200
"""
import argparse
import os
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

def collect_scalars(run_dir: Path, tag: str):
    """Return a list of scalar values for `tag` in a single run directory."""
    ea = event_accumulator.EventAccumulator(str(run_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [s.value for s in ea.Scalars(tag)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True,
                        help="Root TensorBoard log directory")
    parser.add_argument("--tag", default="perf/tokens/s",
                        help="Scalar tag to extract")
    parser.add_argument("--steps", type=int, default=200,
                        help="How many steps to average over")
    args = parser.parse_args()

    root = Path(args.logdir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    values = []
    # Walk every subdirectory; each run usually has its own tfevents files
    for path, _, files in os.walk(root):
        if any(f.startswith("events") for f in files):
            scalars = collect_scalars(Path(path), args.tag)
            if scalars:
                values.extend(scalars)  # append all before slicing below

    if not values:
        print(f"No values found for tag '{args.tag}'")
        return

    # Use the first N values across the concatenated list
    n = min(args.steps, len(values))
    avg = sum(values[:n]) / n
    print(f"Average of '{args.tag}' for the first {n} steps: {avg:.6f}")

if __name__ == "__main__":
    main()
