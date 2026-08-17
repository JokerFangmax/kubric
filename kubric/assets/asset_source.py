# Copyright 2024 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import difflib
import functools
import hashlib
import logging
import os
import pathlib
import shutil
import tarfile
import tempfile
import time

import numpy as np
import tensorflow as tf

from typing import Optional, Dict, Any, Type
import weakref

from kubric import core
from kubric import file_io
from kubric.kubric_typing import PathLike


class ClosableResource:
  """TODO(klausg): documentation."""
  _set_of_open_resources = weakref.WeakSet()

  def __init__(self):
    super().__init__()
    self.is_closed = False
    self._set_of_open_resources.add(self)

  def close(self):
    try:
      self._set_of_open_resources.remove(self)
    except (ValueError, KeyError):
      pass  # not listed anymore. Ignore.

  @classmethod
  def close_all(cls):
    while True:
      try:
        r = cls._set_of_open_resources.pop()
      except KeyError:
        break
      r.close()


class AssetSource(ClosableResource):
  """TODO(klausg): documentation."""

  @classmethod
  def from_manifest(
      cls,
      manifest_path: PathLike,
      scratch_dir: Optional[PathLike] = None,
      cache_dir: Optional[PathLike] = None,
  ) -> "AssetSource":
    if manifest_path == "gs://kubric-public/assets/ShapeNetCore.v2.json":
      raise ValueError(f"The path `{manifest_path}` is a placeholder for the real path. "
                       "Please visit https://shapenet.org, agree to terms and conditions."
                       "After logging in, you will find the manifest URL here:"
                       "https://shapenet.org/download/kubric")

    manifest_path = file_io.as_path(manifest_path)
    if cache_dir == "":
      cache_dir = None
    cache_dir = None if cache_dir is None else pathlib.Path(cache_dir)
    if cache_dir is not None:
      manifest_path = cls._cache_manifest_file(manifest_path, cache_dir)
    manifest = file_io.read_json(manifest_path)
    name = manifest.get("name", manifest_path.stem)  # default to filename
    data_dir = manifest.get("data_dir", manifest_path.parent)  # default to manifest dir
    assets = manifest["assets"]
    return cls(
        name=name,
        data_dir=data_dir,
        assets=assets,
        scratch_dir=scratch_dir,
        cache_dir=cache_dir)

  def __init__(
      self,
      name: str,
      data_dir: PathLike,
      assets: Dict[str, Any],
      scratch_dir: Optional[PathLike] = None,
      cache_dir: Optional[PathLike] = None,
  ):
    super().__init__()
    self.name = name
    self.data_dir = file_io.as_path(data_dir)
    self.cache_dir = None if cache_dir is None else pathlib.Path(cache_dir)
    self._uses_persistent_cache = self.cache_dir is not None
    logging.info("Created AssetSource '%s' with '%d' assets at URI='%s'",
                 name, len(assets), self.data_dir)
    if self._uses_persistent_cache:
      cache_key = hashlib.sha1(str(self.data_dir).encode("utf-8")).hexdigest()[:12]
      self.local_dir = self.cache_dir / f"{name}_{cache_key}"
      self.local_dir.mkdir(parents=True, exist_ok=True)
    else:
      self.local_dir = pathlib.Path(tempfile.mkdtemp(prefix=name, dir=scratch_dir))
    self._assets = assets

  def close(self):
    if self.is_closed:
      return
    try:
      if not self._uses_persistent_cache:
        shutil.rmtree(self.local_dir)
    finally:
      self.is_closed = True
      super().close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  @functools.cached_property
  def db(self):
    import pandas as pd
    db = pd.DataFrame([{"id": k} | v["kwargs"] | v["metadata"]
                       for k, v in self._assets.items()])

    def get_category_id(x):
      if x['category'] in self.categories:
        return self.categories.index(x['category'])
      else:
        return np.nan

    if "category_id" not in db:
      db["category_id"] = db.apply(get_category_id, axis=1)
    return db

  @functools.cached_property
  def categories(self):
    return sorted(filter(None, {v["metadata"].get("category", "")
                                for v in self._assets.values()}))

  @functools.cached_property
  def all_asset_ids(self):
    return sorted(self._assets.keys())

  @staticmethod
  def _resolve_asset_type(asset_type: str) -> Type:
    types = {
        "FileBasedObject": core.FileBasedObject,
        "Texture": core.Texture,
    }
    if asset_type not in types:
      raise KeyError(f"Unknown asset_type {asset_type!r}. "
                     f"Available types: {types!r}")
    return types[asset_type]

  def _resolve_asset_path(self, path: Optional[str], asset_id: str) -> Optional[PathLike]:
    if path is None:
      return None
    elif path == "":
      path = f"{asset_id}.tar.gz"

    return self.data_dir / path

  @staticmethod
  def _adjust_paths(asset_kwargs: Dict[str, Any], asset_dir: PathLike) -> Dict[str, Any]:
    """If present, replace '{asset_dir}' prefix with actual asset_dir in each kwarg value."""
    def _adjust_path(p):
      if isinstance(p, str) and p.startswith("{asset_dir}/"):
        return str(asset_dir / p[12:])
      elif isinstance(p, dict):
        return {key: _adjust_path(value) for key, value in p.items()}
      else:
        return p

    return {k: _adjust_path(v) for k, v in asset_kwargs.items()}

  def create(self, asset_id: str, add_metadata: bool = True, **kwargs) -> Type[core.Asset]:
    """
    Create an instance of an asset by a given id.

    Performs the following steps
    1. check if asset_id is found in manifest and retrieve entry
    2. determine Asset class and full path (can be remote or local cache or missing)
    3. if path is not none, then fetch and unpack the zipped asset to scratch_dir
    4. construct kwargs from asset_entry->kwargs, override with **kwargs and then
    adjust paths (ones that start with “{{asset_dir}}”
    5. create asset by calling constructor with kwargs
    6. set metadata (if add_metadata is True)
    7. return asset

    Args:
        asset_id (str): the id of the asset to be created
                        (corresponds to its key in the manifest file and
                        typically also to the filename)
        add_metadata (bool): whether to add the metadata from the asset to the instance
        **kwargs: additional kwargs to be passed to the asset constructor

    Returns:
      An instance of the specified asset (subtype of kubric.core.Asset)
    """
    # find corresponding asset entry
    asset_entry = self._assets.get(asset_id)
    if not asset_entry:
      close_matches = difflib.get_close_matches(asset_id, possibilities=self.all_asset_ids, n=1)
      if close_matches:
        raise KeyError(f"Unknown asset with id='{asset_id}'. Did you mean '{close_matches[0]}'?")

    # determine type and path
    asset_type = self._resolve_asset_type(asset_entry["asset_type"])
    asset_path = self._resolve_asset_path(asset_entry.get("path", ""), asset_id)

    # fetch and unpack tar.gz file if necessary
    asset_dir = None if asset_path is None else self.fetch(asset_path, asset_id)

    # construct kwargs
    asset_kwargs = asset_entry.get("kwargs", {})
    asset_kwargs.update(kwargs)
    asset_kwargs = self._adjust_paths(asset_kwargs, asset_dir)
    if asset_type == core.FileBasedObject:
      asset_kwargs["asset_id"] = asset_id
    # create the asset
    asset = asset_type(**asset_kwargs)
    # set the metadata
    if add_metadata:
      asset.metadata.update(asset_entry.get("metadata", {}))

    return asset

  def fetch(self, asset_path, asset_id):
    local_path = self.local_dir / (asset_id + ".tar.gz")
    asset_dir = self.local_dir / asset_id
    if self._is_asset_dir_ready(asset_dir):
      return asset_dir

    self.local_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = self.local_dir / ".locks" / asset_id
    with self._directory_lock(lock_dir):
      if self._is_asset_dir_ready(asset_dir):
        return asset_dir

      if not local_path.exists():
        logging.info("Caching asset %s from %s", asset_id, str(asset_path))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = local_path.with_suffix(local_path.suffix + f".tmp.{os.getpid()}")
        if tmp_path.exists():
          tmp_path.unlink()
        tf.io.gfile.copy(asset_path, tmp_path)
        os.replace(tmp_path, local_path)

      if not self._is_asset_dir_ready(asset_dir):
        self._extract_asset_archive(local_path, asset_id, asset_dir)

    return self.local_dir / asset_id

  @staticmethod
  def _is_asset_dir_ready(asset_dir: pathlib.Path) -> bool:
    return asset_dir.exists() and (asset_dir / "data.json").exists()

  @classmethod
  def _is_asset_ready(cls, local_path: pathlib.Path, asset_dir: pathlib.Path) -> bool:
    return local_path.exists() and cls._is_asset_dir_ready(asset_dir)

  @staticmethod
  def _directory_lock(lock_dir: pathlib.Path, timeout: float = 600.0, poll: float = 0.2):
    class _LockContext:
      def __enter__(self_nonlocal):
        start_time = time.time()
        lock_dir.parent.mkdir(parents=True, exist_ok=True)
        while True:
          try:
            lock_dir.mkdir()
            return self_nonlocal
          except FileExistsError:
            if time.time() - start_time > timeout:
              raise TimeoutError(f"Timed out waiting for cache lock {lock_dir}")
            time.sleep(poll)

      def __exit__(self_nonlocal, exc_type, exc_val, exc_tb):
        shutil.rmtree(lock_dir, ignore_errors=True)
        return False

    return _LockContext()

  @staticmethod
  def _cache_manifest_file(manifest_path: PathLike, cache_dir: pathlib.Path) -> PathLike:
    manifest_path = file_io.as_path(manifest_path)
    manifest_cache_dir = cache_dir / "_manifests"
    manifest_cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_name = pathlib.Path(str(manifest_path)).name
    manifest_key = hashlib.sha1(str(manifest_path).encode("utf-8")).hexdigest()[:12]
    cached_manifest_path = manifest_cache_dir / f"{manifest_name}.{manifest_key}"
    if cached_manifest_path.exists():
      return cached_manifest_path

    lock_dir = manifest_cache_dir / ".locks" / manifest_key
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    with AssetSource._directory_lock(lock_dir):
      if cached_manifest_path.exists():
        return cached_manifest_path
      logging.info("Caching manifest %s to %s", manifest_path, cached_manifest_path)
      tmp_path = cached_manifest_path.with_suffix(cached_manifest_path.suffix + f".tmp.{os.getpid()}")
      if tmp_path.exists():
        tmp_path.unlink()
      tf.io.gfile.copy(manifest_path, tmp_path)
      os.replace(tmp_path, cached_manifest_path)
    return cached_manifest_path

  def _extract_asset_archive(self, local_path: pathlib.Path, asset_id: str, asset_dir: pathlib.Path):
    extract_root = self.local_dir / f".extract_{asset_id}_{os.getpid()}"
    shutil.rmtree(extract_root, ignore_errors=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    try:
      with tarfile.open(local_path, "r:gz") as tar:
        list_of_files = tar.getnames()
        if asset_id in list_of_files and tar.getmember(asset_id).isdir():
          assert f"{asset_id}/data.json" in list_of_files, list_of_files
          tar.extractall(extract_root)
          extracted_dir = extract_root / asset_id
        else:
          assert "data.json" in list_of_files, list_of_files
          extracted_dir = extract_root / asset_id
          extracted_dir.mkdir(parents=True, exist_ok=True)
          tar.extractall(extracted_dir)
        logging.debug("Extracted %s", repr([m.name for m in tar.getmembers()]))

      if self._is_asset_dir_ready(asset_dir):
        return

      try:
        os.replace(extracted_dir, asset_dir)
      except FileExistsError:
        if not self._is_asset_dir_ready(asset_dir):
          raise
    finally:
      shutil.rmtree(extract_root, ignore_errors=True)

  def get_test_split(self, fraction=0.1):
    """
    Generates a train/test split for the asset source.

    Args:
      fraction: the fraction of the asset source to use for the held-out set.

    Returns:
      train_ids: list of asset ID strings
      test_ids: list of asset ID strings
    """
    rng = np.random.default_rng(42)
    test_size = int(round(len(self.all_asset_ids) * fraction))
    test_ids = rng.choice(self.all_asset_ids, size=test_size, replace=False)
    train_ids = [i for i in self.all_asset_ids if i not in test_ids]
    return train_ids, test_ids
