#!/usr/bin/env python3
"""Validate contact-force sidecars emitted by movi_ab_worker.py."""

import argparse
from pathlib import Path

import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset_root", type=Path)
  parser.add_argument("--num-samples", type=int, default=5)
  parser.add_argument("--plot-dir", type=Path, default=None)
  args = parser.parse_args()

  samples = sorted(args.dataset_root.glob("*/physics.npz"))[:args.num_samples]
  if not samples:
    raise FileNotFoundError(f"No physics.npz files found below {args.dataset_root}")

  plot_dir = args.plot_dir or args.dataset_root / "physics_validation"
  plot_dir.mkdir(parents=True, exist_ok=True)
  saw_contact = False
  for physics_path in samples:
    with np.load(physics_path) as data:
      force = data["c_force_raw"].astype(np.float32)
      floor = float(data["c_floor"])
      static = data["c_static"].astype(np.int64)

    if force.ndim != 3 or force.shape[-1] != 6:
      raise ValueError(f"{physics_path}: c_force_raw must have shape (T, N, 6), got {force.shape}")
    point0_norm = np.linalg.norm(force[:, 0, :3], axis=1)
    frame_norm = np.linalg.norm(force[..., :3], axis=-1).max(axis=1)
    nonzero_frames = np.flatnonzero(frame_norm > 0)
    saw_contact |= nonzero_frames.size > 0
    print(
        f"{physics_path.parent.name}: c_force_raw.max()={force.max():.6g}, "
        f"force_frames={nonzero_frames.tolist()}, c_floor={floor:.6g}, "
        f"c_static={static.tolist()}"
    )

    try:
      import matplotlib.pyplot as plt
      plt.figure(figsize=(7, 3))
      plt.plot(point0_norm)
      plt.xlabel("frame")
      plt.ylabel("||force(point 0)||")
      plt.tight_layout()
      plt.savefig(plot_dir / f"{physics_path.parent.name}_point0_force.png")
      plt.close()
    except ImportError:
      print("matplotlib unavailable; skipped force plot")

  if not saw_contact:
    raise RuntimeError("No contact forces were recorded; inspect scene duration/spawn settings.")


if __name__ == "__main__":
  main()
