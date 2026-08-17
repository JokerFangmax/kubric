import json
from pathlib import Path

import pytest

from challenges.movi import movi_shards


def _write_sample(sample_dir: Path, sample_id: int) -> None:
  sample_dir.mkdir(parents=True, exist_ok=True)
  (sample_dir / "metadata.json").write_text(
      json.dumps({"sample_id": sample_id, "kind": "metadata"}),
      encoding="utf-8")
  (sample_dir / "events.json").write_text(
      json.dumps({"sample_id": sample_id, "kind": "events"}),
      encoding="utf-8")
  (sample_dir / "data_ranges.json").write_text(
      json.dumps({"depth": [0.0, 1.0]}),
      encoding="utf-8")
  (sample_dir / "point_cloud_states.pkl").write_bytes(
      f"point-cloud-{sample_id}".encode("utf-8"))
  (sample_dir / "rgba_00000.png").write_bytes(
      f"rgba-{sample_id}".encode("utf-8"))
  (sample_dir / "depth_00000.tiff").write_bytes(
      f"depth-{sample_id}".encode("utf-8"))


def _build_raw_dataset(tmp_path: Path, num_samples: int = 3) -> Path:
  raw_root = tmp_path / "raw"
  raw_root.mkdir()
  for sample_id in range(num_samples):
    _write_sample(raw_root / f"{sample_id:05d}", sample_id=sample_id)
  (raw_root / "logs").mkdir()
  (raw_root / "status").mkdir()
  return raw_root


def test_pack_webdataset_round_trip(tmp_path: Path):
  raw_root = _build_raw_dataset(tmp_path, num_samples=3)
  output_root = tmp_path / "packed"

  written = movi_shards.pack_dataset(
      source_root=raw_root,
      output_root=output_root,
      output_format="webdataset",
      shard_size=2,
  )

  web_root = written["webdataset"]
  manifest = json.loads((web_root / "dataset_manifest.json").read_text(
      encoding="utf-8"))
  assert manifest["num_samples"] == 3
  assert manifest["num_shards"] == 2

  shard_paths = sorted((web_root / "shards").glob("*.tar"))
  assert [path.name for path in shard_paths] == [
      "shard-000000.tar",
      "shard-000001.tar",
  ]

  records = list(movi_shards.iter_webdataset_records(shard_paths[0]))
  assert [record["sample_id"] for record in records] == ["00000", "00001"]
  payload_files = movi_shards.unpack_payload_tar(records[0]["payload_tar"])
  assert payload_files["rgba_00000.png"] == b"rgba-0"
  assert payload_files["depth_00000.tiff"] == b"depth-0"


@pytest.mark.skipif(movi_shards.tf is None, reason="tensorflow is not installed")
def test_pack_tfrecord_round_trip(tmp_path: Path):
  raw_root = _build_raw_dataset(tmp_path, num_samples=3)
  output_root = tmp_path / "packed"

  written = movi_shards.pack_dataset(
      source_root=raw_root,
      output_root=output_root,
      output_format="tfrecord",
      shard_size=2,
  )

  tfrecord_root = written["tfrecord"]
  manifest_lines = (tfrecord_root / "manifest.jsonl").read_text(
      encoding="utf-8").strip().splitlines()
  assert len(manifest_lines) == 3

  shard_paths = sorted((tfrecord_root / "shards").glob("*.tfrecord"))
  assert [path.name for path in shard_paths] == [
      "shard-000000.tfrecord",
      "shard-000001.tfrecord",
  ]

  records = list(movi_shards.iter_tfrecord_records(shard_paths[1]))
  assert [record["sample_id"] for record in records] == ["00002"]
  payload_files = movi_shards.unpack_payload_tar(records[0]["payload_tar"])
  assert payload_files["metadata.json"]
  assert payload_files["point_cloud_states.pkl"] == b"point-cloud-2"
