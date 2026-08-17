#!/usr/bin/env python3
"""Validate trajectory diversity for MOVi physics exports."""

import argparse
import collections
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def _load_pickle(path):
  with path.open("rb") as fp:
    return pickle.load(fp)


def _find_samples(dataset_root, prefer_physics=True):
  physics_paths = {path.parent: path for path in dataset_root.glob("**/physics.npz")}
  pkl_paths = {path.parent: path for path in dataset_root.glob("**/point_cloud_states.pkl")}
  sample_dirs = sorted(set(physics_paths) | set(pkl_paths))

  samples = []
  for sample_dir in sample_dirs:
    physics_path = physics_paths.get(sample_dir)
    pkl_path = pkl_paths.get(sample_dir)
    if prefer_physics and physics_path is not None:
      samples.append(physics_path)
    elif pkl_path is not None:
      samples.append(pkl_path)
    elif physics_path is not None:
      samples.append(physics_path)
  return samples


def _load_positions(path):
  if path.suffix == ".npz":
    with np.load(path, allow_pickle=True) as data:
      if "x_s_raw" not in data:
        raise KeyError(f"{path} does not contain x_s_raw")
      positions = np.asarray(data["x_s_raw"][..., :3], dtype=np.float32)
    return positions

  payload = _load_pickle(path)
  if "point_states" not in payload:
    raise KeyError(f"{path} does not contain point_states")
  return np.asarray(payload["point_states"][..., :3], dtype=np.float32)


def _load_contact_frames(sample_path, force_threshold):
  physics_path = sample_path if sample_path.name == "physics.npz" else sample_path.parent / "physics.npz"
  if physics_path.exists():
    with np.load(physics_path, allow_pickle=True) as data:
      if "c_force_raw" not in data:
        return []
      force = np.asarray(data["c_force_raw"][..., :3], dtype=np.float32)
    frame_norm = np.linalg.norm(force, axis=-1).max(axis=1)
    return np.flatnonzero(frame_norm > force_threshold).astype(int).tolist()

  metadata_path = sample_path.parent / "metadata.json"
  if metadata_path.exists():
    with metadata_path.open("r", encoding="utf-8") as fp:
      metadata = json.load(fp)
    return metadata.get("physics", {}).get("force_nonzero_frames", [])

  return []


def _trajectory_hash(positions, max_t_raw):
  prefix = np.ascontiguousarray(positions[:max_t_raw], dtype=np.float32)
  digest = hashlib.sha256()
  digest.update(str(prefix.shape).encode("ascii"))
  digest.update(prefix.tobytes())
  return digest.hexdigest()


def _path_length_after_contact(positions, contact_frames):
  if not contact_frames:
    return np.nan
  first_contact = int(contact_frames[0])
  if first_contact >= positions.shape[0] - 1:
    return 0.0
  com = positions[:, :, :3].mean(axis=1)
  deltas = np.diff(com[first_contact:], axis=0)
  return float(np.linalg.norm(deltas, axis=1).sum())


def _print_histogram(title, values):
  print(title)
  if not values:
    print("  none")
    return
  for value, count in sorted(collections.Counter(values).items()):
    print(f"  {value}: {count}")


def _print_distribution(title, values):
  finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
  print(title)
  if finite.size == 0:
    print("  none")
    return
  percentiles = np.percentile(finite, [0, 25, 50, 75, 100])
  print(
      "  count={count} mean={mean:.6g} min={min:.6g} p25={p25:.6g} "
      "median={median:.6g} p75={p75:.6g} max={max:.6g}".format(
          count=finite.size,
          mean=float(finite.mean()),
          min=float(percentiles[0]),
          p25=float(percentiles[1]),
          median=float(percentiles[2]),
          p75=float(percentiles[3]),
          max=float(percentiles[4]),
      ))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset_root", type=Path)
  parser.add_argument("--max-T-raw", type=int, default=60,
                      help="Number of raw frames included in the trajectory hash.")
  parser.add_argument("--force-threshold", type=float, default=1e-6,
                      help="Contact-force norm threshold for contact-frame detection.")
  parser.add_argument("--include-pkl-when-npz-exists", action="store_true",
                      help="Validate point_cloud_states.pkl even when physics.npz exists.")
  args = parser.parse_args()

  samples = _find_samples(
      args.dataset_root,
      prefer_physics=not args.include_pkl_when_npz_exists)
  if not samples:
    raise FileNotFoundError(f"No physics.npz or point_cloud_states.pkl files below {args.dataset_root}")

  hashes = collections.defaultdict(list)
  first_contact_frames = []
  post_contact_path_lengths = []
  no_contact = []

  for sample_path in samples:
    positions = _load_positions(sample_path)
    hashes[_trajectory_hash(positions, args.max_T_raw)].append(sample_path)
    contact_frames = _load_contact_frames(sample_path, args.force_threshold)
    if contact_frames:
      first_contact_frames.append(int(contact_frames[0]))
    else:
      no_contact.append(sample_path)
    post_contact_path_lengths.append(
        _path_length_after_contact(positions, contact_frames))

  duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
  unique_count = len(hashes)
  total_count = len(samples)
  print(f"unique_trajectories / total_trajectories: {unique_count} / {total_count}")
  print(f"duplicate_groups: {len(duplicate_groups)}")
  duplicate_total = sum(len(paths) for paths in duplicate_groups)
  print(f"trajectories_in_duplicate_groups: {duplicate_total}")

  if duplicate_groups:
    print("duplicates:")
    for group_index, paths in enumerate(duplicate_groups, start=1):
      print(f"  group {group_index} ({len(paths)} files):")
      for path in paths:
        print(f"    {path}")

  _print_histogram("first_contact_frame_distribution:", first_contact_frames)
  print(f"no_contact_samples: {len(no_contact)}")
  for path in no_contact:
    print(f"  {path}")
  _print_distribution(
      "post_contact_COM_path_length_distribution:",
      post_contact_path_lengths)


if __name__ == "__main__":
  main()
