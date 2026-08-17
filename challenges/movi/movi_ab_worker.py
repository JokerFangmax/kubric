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

"""
Worker file for the Multi-Object Video (MOVi) datasets A and B.
Objects:
  * The number of objects is randomly chosen between
    --min_num_objects (3) and --max_num_objects (10)
  * The objects are randomly chosen from either the CLEVR (MOVi-A) or the
    KuBasic set.
  * They are either rubber or metallic with different different colors and sizes


MOVid-A
  --camera=clevr --background=clevr --objects_set=clevr
  --min_num_objects=3 --max_num_objects=10

MOVid-B
  --camera=random --background=colored --objects_set=kubasic
  --min_num_objects=3 --max_num_objects=10

"""

import contextlib
import logging
import pathlib

import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np

try:
  import trimesh
except ImportError:  # pragma: no cover
  trimesh = None

# --- Some configuration values
# the region in which to place objects [(min), (max)]
SPAWN_REGION = [(-5, -5, 1), (5, 5, 5)]
VELOCITY_RANGE = [(-4., -4., 0.), (4., 4., 0.)]
CLEVR_OBJECTS = ("cube", "cylinder", "sphere")
KUBASIC_OBJECTS = ("cube", "cylinder", "sphere", "cone", "torus", "gear",
                   "torus_knot", "sponge", "spot", "teapot", "suzanne")
PHYSICS_OBJECTS = ("sphere", "cube", "cylinder", "suzanne", "gear", "torus_knot")
POINT_STATE_FILENAME = "point_cloud_states.pkl"
PHYSICS_FILENAME = "physics.npz"
POINT_STATE_COMPONENTS = (
    "position_x", "position_y", "position_z",
    "velocity_x", "velocity_y", "velocity_z",
)
POINT_KIND_VOLUME = 0
POINT_KIND_SURFACE = 1
POINT_KIND_NAMES = ("volume", "surface")


@contextlib.contextmanager
def _temporary_numpy_seed(seed):
  previous_state = np.random.get_state()
  np.random.seed(seed)
  try:
    yield
  finally:
    np.random.set_state(previous_state)


def _parse_range(value, name, expected_length=2):
  parts = [float(part) for part in str(value).split(",")]
  if len(parts) != expected_length:
    raise ValueError(f"{name} must contain {expected_length} comma-separated values.")
  if expected_length == 2 and parts[0] > parts[1]:
    raise ValueError(f"{name} lower bound must be <= upper bound.")
  return tuple(parts)


def _sample_unit_vector(rng, dimensions=3):
  vector = rng.normal(size=(dimensions,))
  norm = np.linalg.norm(vector)
  while norm < 1e-8:
    vector = rng.normal(size=(dimensions,))
    norm = np.linalg.norm(vector)
  return vector / norm


def _choose_shape_name(flags, rng, object_index):
  if flags.physics_diversity:
    if flags.physics_shape_selection == "cycle":
      if flags.shape_cycle_index >= 0:
        cycle_index = flags.shape_cycle_index + object_index
      elif flags.seed is not None:
        cycle_index = flags.seed + object_index
      else:
        cycle_index = rng.randint(0, 2**31 - 1) + object_index
      return PHYSICS_OBJECTS[cycle_index % len(PHYSICS_OBJECTS)]
    return rng.choice(PHYSICS_OBJECTS)
  if flags.objects_set == "clevr":
    return rng.choice(CLEVR_OBJECTS)
  return rng.choice(KUBASIC_OBJECTS)


def _sample_log_uniform(rng, minimum, maximum):
  return float(np.exp(rng.uniform(np.log(minimum), np.log(maximum))))


def _place_for_contact_window(obj, simulator, rng, flags):
  contact_frame_min, contact_frame_max = _parse_range(
      flags.target_contact_frames, "--target_contact_frames")
  spawn_xy_min, spawn_xy_max = _parse_range(flags.physics_spawn_xy, "--physics_spawn_xy")
  vertical_velocity_min, vertical_velocity_max = _parse_range(
      flags.vertical_velocity_range, "--vertical_velocity_range")
  gravity = abs(float(simulator.scene.gravity[2]))
  if gravity <= 0.0:
    gravity = 9.81

  for _ in range(flags.physics_placement_trials):
    obj.quaternion = kb.randomness.random_rotation(rng=rng)
    target_frame = rng.uniform(contact_frame_min, contact_frame_max)
    target_time = target_frame / float(simulator.scene.frame_rate)
    vertical_velocity = rng.uniform(vertical_velocity_min, vertical_velocity_max)
    drop_clearance = max(
        0.1,
        0.5 * gravity * target_time**2 - vertical_velocity * target_time)
    floor_z = float(flags.contact_floor_z)
    obj.position = (
        rng.uniform(spawn_xy_min, spawn_xy_max),
        rng.uniform(spawn_xy_min, spawn_xy_max),
        0.0,
    )
    bottom_offset = float(obj.aabbox[0, 2])
    obj.position = (
        obj.position[0],
        obj.position[1],
        floor_z + drop_clearance - bottom_offset,
    )
    if not simulator.check_overlap(obj):
      return vertical_velocity, target_frame, drop_clearance

  logging.warning(
      "Contact-window placement failed for %s; falling back to broad spawn region.",
      obj.uid)
  kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION, rng=rng)
  return rng.uniform(vertical_velocity_min, vertical_velocity_max), None, None


def _apply_physics_diversity(obj, rng, flags, size, vertical_velocity):
  mass_min, mass_max = _parse_range(flags.mass_range, "--mass_range")
  friction_min, friction_max = _parse_range(flags.friction_range, "--friction_range")
  restitution_min, restitution_max = _parse_range(
      flags.restitution_range, "--restitution_range")
  linear_speed_min, linear_speed_max = _parse_range(
      flags.linear_velocity_range, "--linear_velocity_range")
  angular_speed_min, angular_speed_max = _parse_range(
      flags.angular_velocity_range, "--angular_velocity_range")

  if flags.mass_sampling == "log":
    obj.mass = _sample_log_uniform(rng, mass_min, mass_max)
  else:
    obj.mass = float(rng.uniform(mass_min, mass_max))
  obj.friction = float(rng.uniform(friction_min, friction_max))
  obj.restitution = float(rng.uniform(restitution_min, restitution_max))

  xy_direction = _sample_unit_vector(rng, dimensions=2)
  xy_speed = rng.uniform(linear_speed_min, linear_speed_max)
  obj.velocity = (
      float(xy_direction[0] * xy_speed),
      float(xy_direction[1] * xy_speed),
      float(vertical_velocity),
  )

  angular_axis = _sample_unit_vector(rng, dimensions=3)
  angular_speed = rng.uniform(angular_speed_min, angular_speed_max)
  obj.angular_velocity = tuple((angular_axis * angular_speed).astype(float))
  obj.metadata.update({
      "mass": float(obj.mass),
      "friction": float(obj.friction),
      "restitution": float(obj.restitution),
      "initial_velocity": tuple(float(v) for v in obj.velocity),
      "initial_angular_velocity": tuple(float(v) for v in obj.angular_velocity),
      "mass_sampling": flags.mass_sampling,
      "base_size": float(size),
  })


def _load_render_mesh(obj, mesh_cache):
  if obj.render_filename is None:
    raise ValueError(f"{obj.uid} does not define a render mesh.")

  mesh_path = pathlib.Path(obj.render_filename).resolve()
  cache_key = str(mesh_path)
  if cache_key not in mesh_cache:
    loaded_mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if isinstance(loaded_mesh, trimesh.Scene):
      geometries = tuple(loaded_mesh.geometry.values())
      if not geometries:
        raise ValueError(f"Render mesh scene for {obj.uid} is empty.")
      loaded_mesh = trimesh.util.concatenate(geometries)
    if not isinstance(loaded_mesh, trimesh.Trimesh):
      raise ValueError(f"Unsupported mesh type {type(loaded_mesh)!r} for {obj.uid}.")
    if loaded_mesh.is_empty:
      raise ValueError(f"Render mesh for {obj.uid} is empty.")
    mesh_cache[cache_key] = loaded_mesh

  return mesh_cache[cache_key]


def _sample_points_from_bounds(obj, num_points, rng):
  bounds = np.asarray(obj.bounds, dtype=np.float32)
  if num_points <= 0:
    return np.zeros((0, 3), dtype=np.float32)
  return rng.uniform(bounds[0], bounds[1], size=(num_points, 3)).astype(np.float32)


def _sample_points_from_bounds_surface(obj, num_points, rng):
  bounds = np.asarray(obj.bounds, dtype=np.float32)
  if num_points <= 0:
    return np.zeros((0, 3), dtype=np.float32)

  lower = bounds[0]
  upper = bounds[1]
  extents = upper - lower
  face_areas = np.array([
      extents[1] * extents[2], extents[1] * extents[2],
      extents[0] * extents[2], extents[0] * extents[2],
      extents[0] * extents[1], extents[0] * extents[1],
  ], dtype=np.float32)
  probabilities = face_areas / np.sum(face_areas)
  face_indices = rng.choice(6, size=num_points, p=probabilities)

  points = rng.uniform(lower, upper, size=(num_points, 3)).astype(np.float32)
  points[face_indices == 0, 0] = lower[0]
  points[face_indices == 1, 0] = upper[0]
  points[face_indices == 2, 1] = lower[1]
  points[face_indices == 3, 1] = upper[1]
  points[face_indices == 4, 2] = lower[2]
  points[face_indices == 5, 2] = upper[2]
  return points


def _sample_exact_mesh_volume_points(mesh, num_points, rng):
  batches = []
  remaining = num_points
  max_attempts = 8

  for _ in range(max_attempts):
    if remaining <= 0:
      break
    batch_count = max(remaining, num_points)
    with _temporary_numpy_seed(int(rng.randint(0, 2**31 - 1))):
      sampled = trimesh.sample.volume_mesh(mesh, count=batch_count)
    sampled = np.asarray(sampled, dtype=np.float32).reshape((-1, 3))
    if sampled.size == 0:
      continue
    sampled = sampled[:remaining]
    batches.append(sampled)
    remaining -= sampled.shape[0]

  if remaining > 0:
    raise ValueError(f"volume_mesh returned only {num_points - remaining}/{num_points} points")

  return np.concatenate(batches, axis=0).astype(np.float32)


def _sample_exact_mesh_surface_points(mesh, num_points, rng):
  sampled = np.zeros((0, 3), dtype=np.float32)
  if hasattr(trimesh.sample, "sample_surface_even"):
    with _temporary_numpy_seed(int(rng.randint(0, 2**31 - 1))):
      sampled, _ = trimesh.sample.sample_surface_even(mesh, count=num_points)
    sampled = np.asarray(sampled, dtype=np.float32).reshape((-1, 3))

  if sampled.shape[0] < num_points:
    remaining = num_points - sampled.shape[0]
    with _temporary_numpy_seed(int(rng.randint(0, 2**31 - 1))):
      extra, _ = trimesh.sample.sample_surface(mesh, count=remaining)
    extra = np.asarray(extra, dtype=np.float32).reshape((-1, 3))
    sampled = np.concatenate([sampled, extra], axis=0)

  if sampled.shape[0] > num_points:
    sampled = sampled[:num_points]

  if sampled.shape != (num_points, 3):
    raise ValueError(f"surface sampling returned shape {sampled.shape}, expected {(num_points, 3)}")
  return sampled.astype(np.float32)


def _sample_object_volume_points(obj, num_points, mesh_cache, rng):
  if num_points <= 0:
    return np.zeros((0, 3), dtype=np.float32), "disabled"

  if trimesh is None:
    logging.warning(
        "trimesh is unavailable; falling back to local bounds sampling for %s.",
        obj.uid)
    local_points = _sample_points_from_bounds(obj, num_points, rng)
    return local_points * np.asarray(obj.scale, dtype=np.float32), "bounds_fallback"

  try:
    mesh = _load_render_mesh(obj, mesh_cache)
    local_points = _sample_exact_mesh_volume_points(mesh, num_points, rng)
    return local_points * np.asarray(obj.scale, dtype=np.float32), "mesh_volume"
  except Exception as exc:  # pylint: disable=broad-except
    logging.warning(
        "Volume sampling failed for %s (%s); falling back to local bounds sampling.",
        obj.uid, exc)
    local_points = _sample_points_from_bounds(obj, num_points, rng)
    return local_points * np.asarray(obj.scale, dtype=np.float32), "bounds_fallback"


def _sample_object_surface_points(obj, num_points, mesh_cache, rng):
  if num_points <= 0:
    return np.zeros((0, 3), dtype=np.float32), "disabled"

  if trimesh is None:
    logging.warning(
        "trimesh is unavailable; falling back to local bounds-surface sampling for %s.",
        obj.uid)
    local_points = _sample_points_from_bounds_surface(obj, num_points, rng)
    return local_points * np.asarray(obj.scale, dtype=np.float32), "bounds_surface_fallback"

  try:
    mesh = _load_render_mesh(obj, mesh_cache)
    local_points = _sample_exact_mesh_surface_points(mesh, num_points, rng)
    return local_points * np.asarray(obj.scale, dtype=np.float32), "mesh_surface_even"
  except Exception as exc:  # pylint: disable=broad-except
    logging.warning(
        "Surface sampling failed for %s (%s); falling back to local bounds-surface sampling.",
        obj.uid, exc)
    local_points = _sample_points_from_bounds_surface(obj, num_points, rng)
    return local_points * np.asarray(obj.scale, dtype=np.float32), "bounds_surface_fallback"


def _combine_sampled_points(surface_points, volume_points):
  blocks = []
  kinds = []
  if surface_points.shape[0] > 0:
    blocks.append(surface_points.astype(np.float32))
    kinds.append(np.full((surface_points.shape[0],), POINT_KIND_SURFACE, dtype=np.uint8))
  if volume_points.shape[0] > 0:
    blocks.append(volume_points.astype(np.float32))
    kinds.append(np.full((volume_points.shape[0],), POINT_KIND_VOLUME, dtype=np.uint8))

  if not blocks:
    return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint8)
  return np.concatenate(blocks, axis=0), np.concatenate(kinds, axis=0)


def _get_object_mesh_geometry(obj, mesh_cache):
  if trimesh is None:
    return None, None

  try:
    mesh = _load_render_mesh(obj, mesh_cache)
  except Exception as exc:  # pylint: disable=broad-except
    logging.warning("Mesh geometry export failed for %s (%s).", obj.uid, exc)
    return None, None

  vertices = np.asarray(mesh.vertices, dtype=np.float32)
  faces = np.asarray(mesh.faces, dtype=np.int32)
  if vertices.ndim != 2 or vertices.shape[1] != 3:
    logging.warning("Unexpected vertex shape %s for %s.", vertices.shape, obj.uid)
    return None, None
  if faces.ndim != 2 or faces.shape[1] != 3:
    logging.warning("Unexpected face shape %s for %s.", faces.shape, obj.uid)
    return None, None

  scaled_vertices = vertices * np.asarray(obj.scale, dtype=np.float32)
  return scaled_vertices, faces


def _get_scale_vector(scale):
  scale_vector = np.asarray(scale, dtype=np.float32)
  if scale_vector.ndim == 0:
    scale_vector = np.full((3,), float(scale_vector), dtype=np.float32)
  else:
    scale_vector = scale_vector.reshape((-1,))
    if scale_vector.shape[0] == 1:
      scale_vector = np.full((3,), float(scale_vector[0]), dtype=np.float32)
    elif scale_vector.shape[0] != 3:
      raise ValueError(f"Expected scale with 1 or 3 values, got shape {scale_vector.shape}.")
  return np.abs(scale_vector).astype(np.float32)


def _compute_bounds_sampling_measures(obj):
  bounds = np.asarray(obj.bounds, dtype=np.float32)
  scale_vector = _get_scale_vector(obj.scale)
  extents = np.maximum(bounds[1] - bounds[0], 0.) * scale_vector
  surface_area = 2. * (
      extents[0] * extents[1] +
      extents[0] * extents[2] +
      extents[1] * extents[2])
  volume = extents[0] * extents[1] * extents[2]
  return {
      "surface_area": float(surface_area),
      "volume": float(volume),
      "surface_measure_method": "bounds_area",
      "volume_measure_method": "bounds_volume",
  }


def _compute_mesh_sampling_measures(obj, mesh_cache):
  if trimesh is None:
    raise RuntimeError("trimesh is unavailable")

  mesh = _load_render_mesh(obj, mesh_cache)
  scale_vector = _get_scale_vector(obj.scale)
  scaled_vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale_vector[None, :]
  scaled_mesh = trimesh.Trimesh(
      vertices=scaled_vertices,
      faces=np.asarray(mesh.faces, dtype=np.int32),
      process=False)

  surface_area = float(scaled_mesh.area)
  volume = float(abs(scaled_mesh.volume))
  return {
      "surface_area": surface_area,
      "volume": volume,
      "surface_measure_method": "mesh_area",
      "volume_measure_method": "mesh_volume",
  }


def _clip_point_count(value, minimum, maximum):
  clipped = max(int(np.rint(value)), 0)
  clipped = max(clipped, max(int(minimum), 0))
  if int(maximum) >= 0:
    clipped = min(clipped, int(maximum))
  return clipped


def _resolve_object_point_counts(obj, mesh_cache, point_count_config):
  measures = _compute_bounds_sampling_measures(obj)
  if trimesh is not None:
    try:
      mesh_measures = _compute_mesh_sampling_measures(obj, mesh_cache)
      if np.isfinite(mesh_measures["surface_area"]) and mesh_measures["surface_area"] > 0:
        measures["surface_area"] = mesh_measures["surface_area"]
        measures["surface_measure_method"] = mesh_measures["surface_measure_method"]
      if np.isfinite(mesh_measures["volume"]) and mesh_measures["volume"] > 0:
        measures["volume"] = mesh_measures["volume"]
        measures["volume_measure_method"] = mesh_measures["volume_measure_method"]
    except Exception as exc:  # pylint: disable=broad-except
      logging.warning(
          "Sampling measure estimation failed for %s (%s); using scaled bounds.",
          obj.uid, exc)

  strategy = point_count_config.point_count_strategy
  if strategy == "adaptive":
    surface_density = max(float(point_count_config.surface_points_per_unit_area), 0.)
    volume_density = max(float(point_count_config.volume_points_per_unit_volume), 0.)
    surface_count = 0
    volume_count = 0
    if surface_density > 0:
      surface_count = _clip_point_count(
          measures["surface_area"] * surface_density,
          point_count_config.min_surface_points_per_object,
          point_count_config.max_surface_points_per_object)
    if volume_density > 0:
      volume_count = _clip_point_count(
          measures["volume"] * volume_density,
          point_count_config.min_volume_points_per_object,
          point_count_config.max_volume_points_per_object)
  else:
    surface_count = max(int(point_count_config.num_surface_points_per_object), 0)
    volume_count = max(int(point_count_config.num_volume_points_per_object), 0)

  measures["point_count_strategy"] = strategy
  measures["surface_count"] = int(surface_count)
  measures["volume_count"] = int(volume_count)
  return measures


def _point_sampling_enabled(point_count_config):
  if point_count_config.point_count_strategy == "adaptive":
    return (
        point_count_config.surface_points_per_unit_area > 0 or
        point_count_config.volume_points_per_unit_volume > 0)
  return (
      point_count_config.num_surface_points_per_object > 0 or
      point_count_config.num_volume_points_per_object > 0)


def _quaternions_to_rotation_matrices(quaternions):
  quaternions = np.asarray(quaternions, dtype=np.float32)
  norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
  quaternions = quaternions / np.clip(norms, a_min=1e-8, a_max=None)
  w = quaternions[:, 0]
  x = quaternions[:, 1]
  y = quaternions[:, 2]
  z = quaternions[:, 3]

  return np.stack([
      1. - 2. * (y * y + z * z),
      2. * (x * y - z * w),
      2. * (x * z + y * w),
      2. * (x * y + z * w),
      1. - 2. * (x * x + z * z),
      2. * (y * z - x * w),
      2. * (x * z - y * w),
      2. * (y * z + x * w),
      1. - 2. * (x * x + y * y),
  ], axis=-1).reshape((-1, 3, 3)).astype(np.float32)


def _compute_point_states(obj, local_points, frame_ids):
  num_frames = len(frame_ids)
  if local_points.size == 0:
    return np.zeros((num_frames, 0, len(POINT_STATE_COMPONENTS)), dtype=np.float32)

  positions = obj.get_values_over_time("position", frames=frame_ids)
  quaternions = obj.get_values_over_time("quaternion", frames=frame_ids)
  velocities = obj.get_values_over_time("velocity", frames=frame_ids)
  angular_velocities = obj.get_values_over_time("angular_velocity", frames=frame_ids)

  rotations = _quaternions_to_rotation_matrices(quaternions)
  world_offsets = np.einsum("tij,pj->tpi", rotations, local_points, optimize=True)
  world_positions = positions[:, None, :] + world_offsets
  world_velocities = velocities[:, None, :] + np.cross(
      angular_velocities[:, None, :], world_offsets)
  return np.concatenate([world_positions, world_velocities], axis=-1).astype(np.float32)


def _collect_point_cloud_states(
    scene,
    assets,
    point_count_config,
    rng,
):
  frame_ids = np.arange(scene.frame_start-1, scene.frame_end, dtype=np.int32)
  mesh_cache = {}
  point_states = []
  local_points = []
  point_kinds = []
  instances = []
  point_offset = 0

  for instance_index, obj in enumerate(assets):
    count_payload = _resolve_object_point_counts(obj, mesh_cache, point_count_config)
    surface_points, surface_sampling_method = _sample_object_surface_points(
        obj, count_payload["surface_count"], mesh_cache, rng)
    volume_points, volume_sampling_method = _sample_object_volume_points(
        obj, count_payload["volume_count"], mesh_cache, rng)
    sampled_points, sampled_point_kinds = _combine_sampled_points(
        surface_points, volume_points)
    mesh_vertices_local, mesh_faces = _get_object_mesh_geometry(obj, mesh_cache)
    instance_states = _compute_point_states(obj, sampled_points, frame_ids)
    next_offset = point_offset + sampled_points.shape[0]

    point_states.append(instance_states)
    local_points.append(sampled_points.astype(np.float32))
    point_kinds.append(sampled_point_kinds.astype(np.uint8))
    instance_payload = {
        "instance_index": instance_index,
        "uid": obj.uid,
        "name": obj.name,
        "asset_id": getattr(obj, "asset_id", None),
        "num_points": int(sampled_points.shape[0]),
        "point_count_strategy": count_payload["point_count_strategy"],
        "num_surface_points": int(surface_points.shape[0]),
        "num_volume_points": int(volume_points.shape[0]),
        "surface_area": float(count_payload["surface_area"]),
        "volume": float(count_payload["volume"]),
        "point_range": [point_offset, next_offset],
        "sampling_method": (
            f"surface:{surface_sampling_method},volume:{volume_sampling_method}"
        ),
        "sampling_methods": {
            "surface": surface_sampling_method,
            "volume": volume_sampling_method,
        },
        "surface_measure_method": count_payload["surface_measure_method"],
        "volume_measure_method": count_payload["volume_measure_method"],
    }
    if mesh_vertices_local is not None and mesh_faces is not None:
      instance_payload["mesh_vertices_local"] = mesh_vertices_local.astype(np.float32)
      instance_payload["mesh_faces"] = mesh_faces.astype(np.int32)
      instance_payload["mesh_num_vertices"] = int(mesh_vertices_local.shape[0])
      instance_payload["mesh_num_faces"] = int(mesh_faces.shape[0])
    instances.append(instance_payload)
    point_offset = next_offset

  total_state_dim = len(POINT_STATE_COMPONENTS)
  if point_states:
    packed_states = np.concatenate(point_states, axis=1).astype(np.float32)
    packed_local_points = np.concatenate(local_points, axis=0).astype(np.float32)
    packed_point_kinds = np.concatenate(point_kinds, axis=0).astype(np.uint8)
  else:
    packed_states = np.zeros((len(frame_ids), 0, total_state_dim), dtype=np.float32)
    packed_local_points = np.zeros((0, 3), dtype=np.float32)
    packed_point_kinds = np.zeros((0,), dtype=np.uint8)

  return {
      "frame_ids": frame_ids,
      "point_states": packed_states,
      "local_points": packed_local_points,
      "point_kinds": packed_point_kinds,
      "point_kind_names": list(POINT_KIND_NAMES),
      "state_components": POINT_STATE_COMPONENTS,
      "instances": instances,
  }


def _build_physics_payload(
    point_cloud_states,
    animation,
    contact_forces,
    assets,
    simulator,
    floor,
    simulation_frame_start=0,
):
  """Build PV-Simulator physics arrays aligned with sampled point states."""
  x_s_raw = np.asarray(point_cloud_states["point_states"], dtype=np.float32)
  frame_ids = np.asarray(point_cloud_states["frame_ids"], dtype=np.int64)
  instances = point_cloud_states["instances"]
  num_frames, num_points, _ = x_s_raw.shape
  num_objects = len(instances)

  c_force_raw = np.zeros((num_frames, num_points, 6), dtype=np.float32)
  point_obj_idx = np.zeros((num_points,), dtype=np.int64)
  c_mat = np.zeros((num_objects, 2), dtype=np.float32)
  c_mass = np.zeros((num_objects,), dtype=np.float32)
  c_static = np.zeros((num_objects,), dtype=np.int64)
  c_init = np.zeros((num_objects, 7), dtype=np.float32)

  assets_by_uid = {asset.uid: asset for asset in assets}
  contact_indices = frame_ids - simulation_frame_start
  for object_idx, instance in enumerate(instances):
    point_start, point_end = instance["point_range"]
    point_obj_idx[point_start:point_end] = object_idx
    obj = assets_by_uid[instance["uid"]]
    bullet_idx = obj.linked_objects[simulator]
    dynamics = simulator.get_dynamics_info(bullet_idx)
    c_mass[object_idx] = dynamics[0]
    c_mat[object_idx] = (dynamics[1], dynamics[5])  # friction, restitution
    c_static[object_idx] = int(dynamics[0] == 0.0)

    animation_index = int(contact_indices[0])
    c_init[object_idx, :3] = animation[obj]["position"][animation_index]
    c_init[object_idx, 3:6] = animation[obj]["velocity"][animation_index]
    c_init[object_idx, 6] = 1.0

    contact_data = contact_forces.get(obj)
    if contact_data is None:
      continue
    if np.any(contact_indices < 0) or np.any(contact_indices >= len(contact_data["force"])):
      raise ValueError("Point-state frame IDs are outside the simulated contact-force range.")
    object_force = contact_data["force"][contact_indices]
    object_contact_point = contact_data["contact_point"][contact_indices]
    c_force_raw[:, point_start:point_end, :3] = object_force[:, None, :]
    c_force_raw[:, point_start:point_end, 3:] = object_contact_point[:, None, :]

  floor_bullet_idx = floor.linked_objects[simulator]
  floor_position, _ = simulator.get_position_and_rotation(floor_bullet_idx)
  return {
      "x_s_raw": x_s_raw,
      "c_force_raw": c_force_raw,
      "c_floor": np.asarray(float(floor_position[2]), dtype=np.float32),
      "c_mat": c_mat,
      "c_mass": c_mass,
      "c_static": c_static,
      "c_init": c_init,
      "point_obj_idx": point_obj_idx,
  }

# --- CLI arguments
parser = kb.ArgumentParser()
# Configuration for the objects of the scene
parser.add_argument("--objects_set", choices=["clevr", "kubasic"],
                    default="clevr")
parser.add_argument("--min_num_objects", type=int, default=2,
                    help="minimum number of objects")
parser.add_argument("--max_num_objects", type=int, default=2,
                    help="maximum number of objects")
# Configuration for the floor and background
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--background", choices=["clevr", "colored"],
                    default="clevr")

# Configuration for the camera
parser.add_argument("--camera", choices=["clevr", "random"], default="clevr")

# Configuration for the source of the assets
parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--asset_cache_dir", type=str, default=None,
                    help="Persistent local cache directory for manifests and downloaded "
                         "asset archives. When set, repeated workers reuse local copies "
                         "instead of refetching from remote storage.")
parser.add_argument("--save_state", dest="save_state", action="store_true")
parser.add_argument("--point_count_strategy", choices=["fixed", "adaptive"],
                    default="fixed",
                    help="How to choose the number of sampled points per object. "
                         "'fixed' uses the same counts for every object. "
                         "'adaptive' scales counts with mesh surface area and volume.")
parser.add_argument("--num_surface_points_per_object", type=int, default=100,
                    help="Number of near-uniformly sampled mesh-surface points per "
                         "foreground object.")
parser.add_argument("--num_volume_points_per_object", type=int, default=100,
                    help="Number of uniformly sampled interior volume points per "
                         "foreground object. Set to 0 to disable the interior component.")
parser.add_argument("--surface_points_per_unit_area", type=float, default=16.0,
                    help="Adaptive mode only. Surface sampling density in points per "
                         "unit mesh area.")
parser.add_argument("--volume_points_per_unit_volume", type=float, default=32.0,
                    help="Adaptive mode only. Interior sampling density in points per "
                         "unit mesh volume.")
parser.add_argument("--min_surface_points_per_object", type=int, default=32,
                    help="Adaptive mode only. Minimum number of surface points per "
                         "foreground object.")
parser.add_argument("--max_surface_points_per_object", type=int, default=-1,
                    help="Adaptive mode only. Maximum number of surface points per "
                         "foreground object. Use -1 to disable the cap.")
parser.add_argument("--min_volume_points_per_object", type=int, default=16,
                    help="Adaptive mode only. Minimum number of interior points per "
                         "foreground object.")
parser.add_argument("--max_volume_points_per_object", type=int, default=-1,
                    help="Adaptive mode only. Maximum number of interior points per "
                         "foreground object. Use -1 to disable the cap.")
parser.add_argument("--physics_diversity", action="store_true",
                    help="Enable broader physics sampling for trajectory datasets.")
parser.add_argument("--physics_shape_selection", choices=["random", "cycle"],
                    default="random",
                    help="Shape selection mode used when --physics_diversity is enabled.")
parser.add_argument("--shape_cycle_index", type=int, default=-1,
                    help="Cycle index for deterministic shape coverage. Defaults to seed.")
parser.add_argument("--mass_range", type=str, default="0.5,5.0",
                    help="Min,max object mass range in kg for --physics_diversity.")
parser.add_argument("--mass_sampling", choices=["linear", "log"], default="log",
                    help="Mass sampling distribution for --physics_diversity.")
parser.add_argument("--friction_range", type=str, default="0.1,0.8",
                    help="Min,max object friction range for --physics_diversity.")
parser.add_argument("--restitution_range", type=str, default="0.2,0.8",
                    help="Min,max object restitution range for --physics_diversity.")
parser.add_argument("--linear_velocity_range", type=str, default="0.5,4.0",
                    help="Min,max horizontal initial speed for --physics_diversity.")
parser.add_argument("--vertical_velocity_range", type=str, default="-0.25,0.25",
                    help="Min,max vertical initial speed for contact-window placement.")
parser.add_argument("--angular_velocity_range", type=str, default="0.5,8.0",
                    help="Min,max initial angular speed for --physics_diversity.")
parser.add_argument("--target_contact_frames", type=str, default="8,12",
                    help="Target first-contact frame range for --physics_diversity.")
parser.add_argument("--physics_spawn_xy", type=str, default="-3.0,3.0",
                    help="Min,max XY spawn range for contact-window placement.")
parser.add_argument("--contact_floor_z", type=float, default=0.0,
                    help="Approximate floor contact Z used for contact-window placement.")
parser.add_argument("--physics_placement_trials", type=int, default=100,
                    help="Maximum placement attempts for --physics_diversity.")
parser.set_defaults(save_state=False, frame_end=24, frame_rate=12,
                    resolution=256)
FLAGS = parser.parse_args()

# --- Common setups & resources
print("start setup")
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
print("finish common setup")
simulator = PyBullet(scene, scratch_dir)
print("finish simulator setup")
renderer = Blender(scene, scratch_dir, samples_per_pixel=16)
print("finish renderer setup")
print("before AssetSource.from_manifest")
kubasic = kb.AssetSource.from_manifest(
    FLAGS.kubasic_assets,
    cache_dir=FLAGS.asset_cache_dir)
print("after AssetSource.from_manifest")

# --- Populate the scene
# Floor / Background
logging.info("Creating a large gray floor...")
floor_material = kb.PrincipledBSDFMaterial(roughness=1., specular=0.)
floor = kubasic.create("dome", name="floor", material=floor_material,
                       scale=2.0,
                       friction=FLAGS.floor_friction,
                       restitution=FLAGS.floor_restitution,
                       static=True, background=True)
scene += floor
if FLAGS.background == "clevr":
  floor_material.color = kb.Color.from_name("gray")
  scene.metadata["background"] = "clevr"
elif FLAGS.background == "colored":
  floor_material.color = kb.random_hue_color()
  scene.metadata["background"] = floor_material.color.hexstr

# Lights
logging.info("Adding four (studio) lights to the scene similar to CLEVR...")
scene.add(kb.assets.utils.get_clevr_lights(rng=rng))
scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

# Camera
logging.info("Setting up the Camera...")
scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
if FLAGS.camera == "clevr":  # Specific position + jitter
  scene.camera.position = [7.48113, -6.50764, 5.34367] + rng.rand(3)
if FLAGS.camera == "random":  # Random position in half-sphere-shell
  scene.camera.position = kb.sample_point_in_half_sphere_shell(
      inner_radius=7., outer_radius=9., offset=0.1)
scene.camera.look_at((0, 0, 0))

print("finish scene setup")

# Add random objects
num_objects = rng.randint(FLAGS.min_num_objects,
                          FLAGS.max_num_objects+1)
logging.info("Randomly placing %d objects:", num_objects)
for i in range(num_objects):
  shape_name = _choose_shape_name(FLAGS, rng, i)
  if FLAGS.objects_set == "clevr" and not FLAGS.physics_diversity:
    size_label, size = kb.randomness.sample_sizes("clevr", rng)
    color_label, random_color = kb.randomness.sample_color("clevr", rng)
  elif FLAGS.objects_set == "kubasic" and not FLAGS.physics_diversity:
    shape_name = rng.choice(KUBASIC_OBJECTS)
    size_label, size = kb.randomness.sample_sizes("uniform", rng)
    color_label, random_color = kb.randomness.sample_color("uniform_hue", rng)
  else:
    size_label, size = kb.randomness.sample_sizes("uniform", rng)
    color_label, random_color = kb.randomness.sample_color("uniform_hue", rng)

  material_name = rng.choice(["metal", "rubber"])
  obj = kubasic.create(
      asset_id=shape_name, scale=size,
      name=f"{size_label} {color_label} {material_name} {shape_name}")
  assert isinstance(obj, kb.FileBasedObject)

  if material_name == "metal":
    obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=1.0,
                                             roughness=0.2, ior=2.5)
    obj.friction = 0.4
    obj.restitution = 0.3
    obj.mass *= 2.7 * size**3
  else:  # material_name == "rubber"
    obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=0.,
                                             ior=1.25, roughness=0.7,
                                             specular=0.33)
    obj.friction = 0.8
    obj.restitution = 0.7
    obj.mass *= 1.1 * size**3

  obj.metadata = {
      "shape": shape_name.lower(),
      "size": size,
      "size_label": size_label,
      "material": material_name.lower(),
      "color": random_color.rgb,
      "color_label": color_label,
  }
  scene.add(obj)
  if FLAGS.physics_diversity:
    vertical_velocity, target_contact_frame, drop_clearance = _place_for_contact_window(
        obj, simulator, rng, FLAGS)
    _apply_physics_diversity(obj, rng, FLAGS, size, vertical_velocity)
    obj.metadata["target_contact_frame"] = (
        None if target_contact_frame is None else float(target_contact_frame))
    obj.metadata["drop_clearance"] = (
        None if drop_clearance is None else float(drop_clearance))
  else:
    kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION, rng=rng)
    # initialize velocity randomly but biased towards center
    obj.velocity = (rng.uniform(*VELOCITY_RANGE) -
                    [obj.position[0], obj.position[1], 0])

  logging.info("    Added %s at %s", obj.asset_id, obj.position)

print("finish object setup")

if FLAGS.save_state:
  logging.info("Saving the simulator state to '%s' prior to the simulation.",
               output_dir / "scene.bullet")
  simulator.save_state(output_dir / "scene.bullet")

# Run dynamic objects simulation
logging.info("Running the simulation ...")
animation, collisions, contact_forces = simulator.run(
    frame_start=0,
    frame_end=scene.frame_end+1,
    return_contact_forces=True,
)

# --- Rendering
if FLAGS.save_state:
  logging.info("Saving the renderer state to '%s' ",
               output_dir / "scene.blend")
  renderer.save_state(output_dir / "scene.blend")


logging.info("Rendering the scene ...")
data_stack = renderer.render(return_layers=("rgba", "depth", "normal", "segmentation"))

# --- Postprocessing
kb.compute_visibility(data_stack["segmentation"], scene.assets)
visible_foreground_assets = [asset for asset in scene.foreground_assets
                             if np.max(asset.metadata["visibility"]) > 0]
visible_foreground_assets = sorted(  # sort assets by their visibility
    visible_foreground_assets,
    key=lambda asset: np.sum(asset.metadata["visibility"]),
    reverse=True)

data_stack["segmentation"] = kb.adjust_segmentation_idxs(
    data_stack["segmentation"],
    scene.assets,
    visible_foreground_assets)
scene.metadata["num_instances"] = len(visible_foreground_assets)

# Save to image files
kb.write_image_dict(data_stack, output_dir)
kb.post_processing.compute_bboxes(data_stack["segmentation"],
                                  visible_foreground_assets)

# --- Point states
point_cloud_states = None
point_cloud_state_summary = None
physics_summary = None
if _point_sampling_enabled(FLAGS):
  logging.info("Sampling foreground points with %s point-count strategy.",
               FLAGS.point_count_strategy)
  point_cloud_states = _collect_point_cloud_states(
      scene=scene,
      assets=scene.foreground_assets,
      point_count_config=FLAGS,
      rng=rng)

  visible_index_by_uid = {
      asset.uid: index for index, asset in enumerate(visible_foreground_assets)
  }
  for instance_summary in point_cloud_states["instances"]:
    instance_summary["visible_instance_index"] = visible_index_by_uid.get(
        instance_summary["uid"], -1)

  kb.write_pkl(point_cloud_states, output_dir / POINT_STATE_FILENAME)
  physics_payload = _build_physics_payload(
      point_cloud_states=point_cloud_states,
      animation=animation,
      contact_forces=contact_forces,
      assets=scene.foreground_assets,
      simulator=simulator,
      floor=floor,
      simulation_frame_start=0,
  )
  np.savez_compressed(output_dir / PHYSICS_FILENAME, **physics_payload)
  force_norm_by_frame = np.linalg.norm(
      physics_payload["c_force_raw"][..., :3], axis=-1).max(axis=1)
  force_nonzero_frames = np.flatnonzero(force_norm_by_frame > 1e-6).astype(int)
  physics_summary = {
      "filename": PHYSICS_FILENAME,
      "shape": list(physics_payload["x_s_raw"].shape),
      "force_nonzero_threshold": 1e-6,
      "force_nonzero_frames": force_nonzero_frames.tolist(),
      "first_contact_frame": (
          None if force_nonzero_frames.size == 0 else int(force_nonzero_frames[0])),
      "post_contact_frames": (
          0 if force_nonzero_frames.size == 0
          else int(physics_payload["x_s_raw"].shape[0] - force_nonzero_frames[0] - 1)),
      "floor_height": float(physics_payload["c_floor"]),
  }
  point_cloud_instances_summary = []
  for instance_payload in point_cloud_states["instances"]:
    instance_summary = {
        "instance_index": instance_payload["instance_index"],
        "uid": instance_payload["uid"],
        "name": instance_payload["name"],
        "asset_id": instance_payload["asset_id"],
        "point_count_strategy": instance_payload["point_count_strategy"],
        "num_points": instance_payload["num_points"],
        "num_surface_points": instance_payload["num_surface_points"],
        "num_volume_points": instance_payload["num_volume_points"],
        "surface_area": instance_payload["surface_area"],
        "volume": instance_payload["volume"],
        "point_range": instance_payload["point_range"],
        "sampling_method": instance_payload["sampling_method"],
        "sampling_methods": instance_payload["sampling_methods"],
        "surface_measure_method": instance_payload["surface_measure_method"],
        "volume_measure_method": instance_payload["volume_measure_method"],
        "visible_instance_index": instance_payload["visible_instance_index"],
    }
    if "mesh_num_vertices" in instance_payload:
      instance_summary["mesh_num_vertices"] = instance_payload["mesh_num_vertices"]
      instance_summary["mesh_num_faces"] = instance_payload["mesh_num_faces"]
    point_cloud_instances_summary.append(instance_summary)
  point_cloud_state_summary = {
      "filename": POINT_STATE_FILENAME,
      "shape": list(point_cloud_states["point_states"].shape),
      "dtype": str(point_cloud_states["point_states"].dtype),
      "state_components": list(POINT_STATE_COMPONENTS),
      "point_kind_names": list(POINT_KIND_NAMES),
      "frame_ids": point_cloud_states["frame_ids"],
      "point_count_strategy": FLAGS.point_count_strategy,
      "num_surface_points_per_object": FLAGS.num_surface_points_per_object,
      "num_volume_points_per_object": FLAGS.num_volume_points_per_object,
      "surface_points_per_unit_area": FLAGS.surface_points_per_unit_area,
      "volume_points_per_unit_volume": FLAGS.volume_points_per_unit_volume,
      "min_surface_points_per_object": FLAGS.min_surface_points_per_object,
      "max_surface_points_per_object": FLAGS.max_surface_points_per_object,
      "min_volume_points_per_object": FLAGS.min_volume_points_per_object,
      "max_volume_points_per_object": FLAGS.max_volume_points_per_object,
      "num_points_per_object_total": (
          FLAGS.num_surface_points_per_object + FLAGS.num_volume_points_per_object
          if FLAGS.point_count_strategy == "fixed" else None),
      "instances": point_cloud_instances_summary,
  }

# --- Metadata
logging.info("Collecting and storing metadata for each object.")
metadata = {
    "flags": vars(FLAGS),
    "metadata": kb.get_scene_metadata(scene),
    "camera": kb.get_camera_info(scene.camera),
    "instances": kb.get_instance_info(scene, visible_foreground_assets),
}
if point_cloud_state_summary is not None:
  metadata["point_cloud_states"] = point_cloud_state_summary
if physics_summary is not None:
  metadata["physics"] = physics_summary
kb.write_json(filename=output_dir / "metadata.json", data=metadata)
kb.write_json(filename=output_dir / "events.json", data={
    "collisions":  kb.process_collisions(
        collisions, scene, assets_subset=visible_foreground_assets),
})

print("finish preprocessing")
kb.done()
