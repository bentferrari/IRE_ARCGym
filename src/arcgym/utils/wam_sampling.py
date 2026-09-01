"""RGB-WAM dataset primitives used exclusively by ARCGym data sampling."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


SCHEMA_NAME = "arcgym_centerline_rgb_wam"
SCHEMA_VERSION = 4
ROS_OPTICAL_AXES = {"x_axis": "right", "y_axis": "down", "z_axis": "forward"}


def numpy_value(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_quaternion_wxyz(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def quaternion_conjugate_wxyz(quaternion):
    result = np.asarray(quaternion, dtype=np.float64).copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_multiply_wxyz(left, right):
    lw, lx, ly, lz = np.moveaxis(np.asarray(left, dtype=np.float64), -1, 0)
    rw, rx, ry, rz = np.moveaxis(np.asarray(right, dtype=np.float64), -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def rotate_vectors_wxyz(quaternion, vectors):
    q = normalize_quaternion_wxyz(quaternion)
    vectors = np.asarray(vectors, dtype=np.float64)
    qv, qw = q[..., 1:], q[..., :1]
    cross = np.cross(qv, vectors)
    return vectors + 2.0 * (qw * cross + np.cross(qv, cross))


def inverse_rotate_vectors_wxyz(quaternion, vectors):
    return rotate_vectors_wxyz(quaternion_conjugate_wxyz(quaternion), vectors)


def relative_pose(previous_position, previous_quaternion, position, quaternion):
    """Compute inv(T_W_C_prev) @ T_W_C_current in the previous camera frame."""
    previous_quaternion = normalize_quaternion_wxyz(previous_quaternion)
    quaternion = normalize_quaternion_wxyz(quaternion)
    translation = inverse_rotate_vectors_wxyz(
        previous_quaternion,
        np.asarray(position, dtype=np.float64) - np.asarray(previous_position, dtype=np.float64),
    )
    rotation = normalize_quaternion_wxyz(
        quaternion_multiply_wxyz(quaternion_conjugate_wxyz(previous_quaternion), quaternion)
    )
    if rotation[0] < 0.0:
        rotation = -rotation
    return translation, rotation


def quaternion_to_matrix_wxyz(quaternion):
    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_centerline_csv(path):
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    names = set(table.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"Centerline CSV must have x,y,z columns: {path}")
    points = np.atleast_2d(np.column_stack((table["x"], table["y"], table["z"])))
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError(f"Centerline must contain at least two finite points: {path}")
    keep = np.concatenate(([True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10))
    points = points[keep]
    if len(points) < 2:
        raise ValueError(f"Centerline contains no non-zero segments: {path}")
    return points


def _normalize_rows(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _parallel_transport_frames(tangents):
    tangents = _normalize_rows(tangents)
    reference = np.asarray((0.0, 0.0, 1.0))
    if abs(np.dot(reference, tangents[0])) > 0.9:
        reference = np.asarray((0.0, 1.0, 0.0))
    normal1 = np.zeros_like(tangents)
    normal2 = np.zeros_like(tangents)
    normal1[0] = np.cross(tangents[0], reference)
    normal1[0] /= max(np.linalg.norm(normal1[0]), 1e-12)
    normal2[0] = np.cross(tangents[0], normal1[0])
    for idx in range(1, len(tangents)):
        axis = np.cross(tangents[idx - 1], tangents[idx])
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-10:
            normal1[idx] = normal1[idx - 1]
        else:
            axis /= axis_norm
            angle = math.atan2(axis_norm, np.dot(tangents[idx - 1], tangents[idx]))
            q = np.concatenate(([math.cos(angle / 2)], axis * math.sin(angle / 2)))
            normal1[idx] = rotate_vectors_wxyz(q, normal1[idx - 1])
        normal1[idx] -= np.dot(normal1[idx], tangents[idx]) * tangents[idx]
        normal1[idx] /= max(np.linalg.norm(normal1[idx]), 1e-12)
        normal2[idx] = np.cross(tangents[idx], normal1[idx])
    return normal1, normal2


class CenterlineGeometry:
    """One ordered, metric centerline in the per-environment dataset world frame."""

    def __init__(self, points_world_m, lookahead_offsets_m):
        self.points = np.asarray(points_world_m, dtype=np.float64)
        self.lookahead_offsets = np.asarray(lookahead_offsets_m, dtype=np.float64)
        if len(self.points) < 2 or np.any(np.diff(self.lookahead_offsets) <= 0):
            raise ValueError("Centerline needs >=2 points and strictly increasing lookahead offsets")
        self.arc = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1)))
        )
        gradients = np.gradient(self.points, self.arc, axis=0, edge_order=1)
        self.tangents = _normalize_rows(gradients)
        tangent_gradient = np.gradient(self.tangents, self.arc, axis=0, edge_order=1)
        self.curvature_vectors = tangent_gradient
        self.curvature_magnitudes = np.linalg.norm(tangent_gradient, axis=1)
        self.normal1, self.normal2 = _parallel_transport_frames(self.tangents)

    @property
    def length_m(self):
        return float(self.arc[-1])

    def reverse(self):
        return CenterlineGeometry(self.points[::-1].copy(), self.lookahead_offsets)

    def sample_arc(self, arc_positions):
        arc_positions = np.asarray(arc_positions, dtype=np.float64)
        return np.column_stack(
            [np.interp(arc_positions, self.arc, self.points[:, dim]) for dim in range(3)]
        )

    def project(self, point_world):
        point_world = np.asarray(point_world, dtype=np.float64)
        starts, segments = self.points[:-1], np.diff(self.points, axis=0)
        squared = np.einsum("ij,ij->i", segments, segments)
        fraction = np.clip(
            np.einsum("ij,ij->i", point_world - starts, segments) / np.maximum(squared, 1e-12),
            0.0,
            1.0,
        )
        projections = starts + fraction[:, None] * segments
        segment_idx = int(np.argmin(np.linalg.norm(projections - point_world, axis=1)))
        s = float(self.arc[segment_idx] + fraction[segment_idx] * math.sqrt(squared[segment_idx]))
        tangent = segments[segment_idx] / max(math.sqrt(squared[segment_idx]), 1e-12)
        return s, projections[segment_idx], tangent

    def camera_condition(self, camera_position, camera_quaternion_ros):
        s, closest, tangent = self.project(camera_position)
        targets_unclamped = s + self.lookahead_offsets
        valid = targets_unclamped <= self.length_m + 1e-10
        targets = np.minimum(targets_unclamped, self.length_m)
        lookahead_world = self.sample_arc(targets)
        lateral_camera = inverse_rotate_vectors_wxyz(
            camera_quaternion_ros, closest - camera_position
        )
        return {
            "progress_m": s,
            "progress_normalized": s / max(self.length_m, 1e-12),
            "remaining_m": max(0.0, self.length_m - s),
            "closest_camera": lateral_camera,
            "lateral_camera": lateral_camera,
            "lateral_distance_m": float(np.linalg.norm(closest - camera_position)),
            "tangent_camera": inverse_rotate_vectors_wxyz(camera_quaternion_ros, tangent),
            "lookahead_camera": inverse_rotate_vectors_wxyz(
                camera_quaternion_ros, lookahead_world - camera_position
            ),
            "lookahead_valid": valid,
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            temp,
            points_world_m=self.points,
            arc_length_m=self.arc,
            progress_normalized=self.arc / max(self.length_m, 1e-12),
            tangent_world=self.tangents,
            curvature_vector_world=self.curvature_vectors,
            curvature_magnitude=self.curvature_magnitudes,
            rmf_normal1_world=self.normal1,
            rmf_normal2_world=self.normal2,
        )
        os.replace(temp, path)


def build_centerline_geometry(
    source_path,
    mesh_scale,
    colon_quaternion,
    colon_position_dataset_world,
    initial_camera_position_dataset_world,
    lookahead_offsets_m,
):
    source = load_centerline_csv(source_path) * float(mesh_scale)
    points = source @ quaternion_to_matrix_wxyz(colon_quaternion).T
    points += np.asarray(colon_position_dataset_world, dtype=np.float64)
    geometry = CenterlineGeometry(points, lookahead_offsets_m)
    # Orient exactly once at run initialization. The order is then immutable.
    camera = np.asarray(initial_camera_position_dataset_world, dtype=np.float64)
    if np.linalg.norm(camera - geometry.points[-1]) < np.linalg.norm(camera - geometry.points[0]):
        geometry = geometry.reverse()
    return geometry


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _atomic_png(path, array):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    Image.fromarray(array).save(temp, format="PNG")
    os.replace(temp, path)


def _git_commit(repository_root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _info_float(info, key):
    value = info.get(key)
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    return float(np.asarray(value).reshape(-1)[0])


def _action_array(actions, num_envs):
    value = numpy_value(actions)
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float64)
    value = value.reshape(1, -1) if value.ndim == 1 else value.reshape(value.shape[0], -1)
    if value.shape[0] == 1 and num_envs > 1:
        value = np.repeat(value, num_envs, axis=0)
    return value


class RGBWAMDatasetWriter:
    """Stream synchronized post-step RGB-WAM transitions for vector environments."""

    def __init__(
        self,
        save_root,
        result_dir,
        run_dir,
        config,
        action_dim,
        base_env,
        centerline_path,
        centerline_scale,
        lookahead_offsets_m,
        save_rgb=True,
        flush_interval=100,
        repository_root=None,
    ):
        self.save_root = Path(save_root).expanduser().resolve()
        self.run_dir = Path(run_dir).expanduser().resolve() if run_dir else self.save_root / Path(result_dir).name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.action_dim = int(action_dim)
        self.save_rgb = bool(save_rgb)
        self.flush_interval = max(1, int(flush_interval))
        self.sample_index = 0
        self.previous_camera_poses = {}
        self.previous_sample_valid = {}
        self.previous_episode_ids = {}
        self.previous_image_paths = {}
        self.env_episode_ids = {i: 0 for i in range(int(config["env_config"]["num_envs"]))}
        self.env_episode_steps = {i: 0 for i in self.env_episode_ids}
        self.env_sample_indices = {i: 0 for i in self.env_episode_ids}

        camera_data = base_env.robot.egocamera.data
        positions_world = np.asarray(numpy_value(camera_data.pos_w), dtype=np.float64)
        quaternions_ros = normalize_quaternion_wxyz(numpy_value(camera_data.quat_w_ros))
        intrinsics = np.asarray(numpy_value(camera_data.intrinsic_matrices), dtype=np.float64)
        rgb_batch = self._rgb_batch(camera_data)
        if rgb_batch is None:
            raise RuntimeError("Simulator camera did not provide genuine RGB output")
        height, width = rgb_batch.shape[1:3]
        if intrinsics.shape[0] < 1 or intrinsics.shape[-2:] != (3, 3):
            raise RuntimeError("Simulator camera did not provide a valid intrinsic matrix")

        scene_origins = np.asarray(numpy_value(base_env.scene.env_origins), dtype=np.float64)
        self.scene_origins = scene_origins
        colon_positions_world = np.asarray(
            numpy_value(base_env.colon.colon_body.data.root_pos_w), dtype=np.float64
        )
        colon_position_local = colon_positions_world[0] - scene_origins[0]
        camera_position_local = positions_world[0] - scene_origins[0]
        self.geometry = build_centerline_geometry(
            centerline_path,
            centerline_scale,
            config["env_config"].get("colon_init_rot", (1, 0, 0, 0)),
            colon_position_local,
            camera_position_local,
            lookahead_offsets_m,
        )
        self.geometry_path = self.run_dir / "geometry" / "centerline_world.npz"
        self.geometry.save(self.geometry_path)

        clipping = config["robot_config"]["front_camera_clipping_range"]
        self.near_clip, self.far_clip = float(clipping[0]), float(clipping[1])
        self.action_duration = float(config["simulation_config"]["dt"] * config["simulation_config"]["render_interval"])
        self.rgb_shape = (height, width, 3)
        self.env_dirs, self.rgb_dirs, self.csv_paths = {}, {}, {}
        self.csv_files, self.writers = {}, {}
        for env_idx in self.env_episode_ids:
            env_dir = self.run_dir / f"env_{env_idx:03d}"
            rgb_dir = env_dir / "rgb"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            self.env_dirs[env_idx], self.rgb_dirs[env_idx] = env_dir, rgb_dir
            self.csv_paths[env_idx] = env_dir / "samples.csv"

        self.fieldnames = self._fieldnames(len(lookahead_offsets_m))
        for env_idx, path in self.csv_paths.items():
            stream = path.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(stream, fieldnames=self.fieldnames)
            writer.writeheader()
            self.csv_files[env_idx], self.writers[env_idx] = stream, writer

        k = intrinsics[0]
        self.metadata = {
            "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
            "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(repository_root or os.getcwd()),
            "sampling_command": list(sys.argv),
            "seed": config["run_config"]["seed"],
            "colon_id": config["run_config"]["colon_id"],
            "task_id": config["run_config"]["task_id"],
            "num_envs": len(self.env_episode_ids),
            "model_path": config["run_config"].get("model_path"),
            "rgb": {
                "type": "simulator_rgb",
                "storage_format": "uint8_png",
                "width": width,
                "height": height,
                "channels": 3,
                "channel_order": "RGB",
                "normalization": "none",
            },
            "camera": {
                "pose_definition": "camera_to_world",
                "world_frame": "per_environment_world = Isaac_world - scene.env_origins[env_idx]",
                "coordinate_convention": "ROS_optical",
                **ROS_OPTICAL_AXES,
                "quaternion_order": "wxyz",
                "source": "egocamera.data.pos_w and egocamera.data.quat_w_ros",
                "source_conversion": "Isaac Lab converts quat_w_world (+X forward,+Z up) to ROS optical",
                "intrinsics": {
                    "fx": float(k[0, 0]), "fy": float(k[1, 1]),
                    "cx": float(k[0, 2]), "cy": float(k[1, 2]),
                    "width": width, "height": height,
                },
                "near_clipping_m": self.near_clip,
                "far_clipping_m": self.far_clip,
            },
            "relative_pose": {
                "definition": "inv(T_world_camera_previous) @ T_world_camera_current",
                "expression_frame": "previous_ROS_optical_camera",
                "first_episode_row_valid": False,
            },
            "action": {
                "primary": "action_*=executed post-constraint physical increment",
                "policy": "policy_action_*=raw policy/random command before environment scaling",
                "constraint_active": "max_abs(executed-pre_constraint_scaled)>1e-9",
                "robot_type": config["robot_config"]["robot_type"],
                "semantics_6d": ["tip_left_right", "tip_up_down", "tip_forward_back", "tip_roll", "tip_yaw", "tip_pitch"],
                "coordinate_frame": "distal robot tip frame (not ROS optical camera frame)",
                "translation_scale": config["env_config"]["translation_action_scale"],
                "rotation_scale": config["env_config"]["rotation_action_scale"],
                "duration_s": self.action_duration,
                "physics_substeps": int(config["simulation_config"]["render_interval"]),
                "aggregation": "sum of identical constrained physical increments across physics substeps",
            },
            "centerline": {
                "source": str(Path(centerline_path).resolve()),
                "world_frame": "same per-environment world frame as camera_world_*",
                "mesh_to_meter_scale": float(centerline_scale),
                "global_file": "geometry/centerline_world.npz",
                "number_of_samples": len(self.geometry.points),
                "length_m": self.geometry.length_m,
                "ordering": "oriented once from initial camera-side endpoint toward goal; immutable during sampling",
                "lookahead_arc_offsets_m": [float(x) for x in lookahead_offsets_m],
                "lateral_vector": "camera origin -> closest centerline point",
                "reference_geometry": "undeformed initial colon centerline",
            },
            "transition_indexing": (
                "row i RGB/action/relative pose are post-step: row i action was executed after row i-1 "
                "and produced row i; source conditioning is read from row i-1"
            ),
            "csv_paths": {f"env_{i:03d}": str(path.relative_to(self.run_dir)) for i, path in self.csv_paths.items()},
        }
        _atomic_json(self.run_dir / "metadata.json", self.metadata)

    @staticmethod
    def _fieldnames(lookahead_count):
        fields = [
            "sample_index", "vec_step", "env_idx", "episode_id", "episode_step",
            "rgb_path", "frame_valid", "timestamp_s", "delta_time_s",
            "relative_pose_valid",
            "delta_camera_x", "delta_camera_y", "delta_camera_z",
            "delta_camera_quat_w", "delta_camera_quat_x", "delta_camera_quat_y", "delta_camera_quat_z",
            "camera_world_x", "camera_world_y", "camera_world_z",
            "camera_world_quat_w", "camera_world_quat_x", "camera_world_quat_y", "camera_world_quat_z",
            "centerline_closest_camera_x", "centerline_closest_camera_y", "centerline_closest_camera_z",
            "centerline_lateral_camera_x", "centerline_lateral_camera_y", "centerline_lateral_camera_z",
            "centerline_lateral_distance_m",
            "centerline_tangent_camera_x", "centerline_tangent_camera_y", "centerline_tangent_camera_z",
            "centerline_progress_m", "centerline_progress_normalized", "centerline_remaining_m",
        ]
        for idx in range(lookahead_count):
            fields += [f"centerline_lookahead_{idx}_{axis}" for axis in ("x", "y", "z")]
            fields += [f"centerline_lookahead_{idx}_valid"]
        fields += [f"action_{idx}" for idx in range(6)]
        fields += [f"policy_action_{idx}" for idx in range(6)]
        fields += [
            "action_constraint_active", "normalized_reward", "raw_step_reward", "env_reward",
            "roi_relative_x", "roi_relative_y", "roi_center_x_px", "roi_center_y_px",
            "roi_distance_from_center", "center_alignment", "roi_aligned", "lumen_visible",
            "goal_reached", "colon_id", "task_id",
        ]
        return fields

    @staticmethod
    def _rgb_batch(camera_data):
        output = getattr(camera_data, "output", {}) or {}
        rgb = numpy_value(output.get("rgb"))
        if rgb is None:
            return None
        rgb = np.asarray(rgb)
        if rgb.ndim != 4 or rgb.shape[-1] not in (3, 4):
            raise ValueError(f"Expected RGB shape [env,height,width,3 or 4], got {rgb.shape}")
        rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.asarray(rgb, dtype=np.float32)
            if np.isfinite(rgb).any() and float(np.nanmax(rgb)) <= 1.0:
                rgb = rgb * 255.0
            rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
            rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        return rgb

    def _save_rgb(self, env_idx, vec_step, rgb):
        name = f"frame_{int(vec_step):08d}.png"
        rgb_path = self.rgb_dirs[env_idx] / name
        valid = rgb.shape == self.rgb_shape and rgb.dtype == np.uint8
        if valid:
            _atomic_png(rgb_path, rgb)
        return str(rgb_path.relative_to(self.env_dirs[env_idx])) if valid else "", bool(valid)

    @staticmethod
    def _action_fields(prefix, values):
        flat = [] if values is None else np.asarray(values, dtype=np.float64).reshape(-1)
        return {f"{prefix}_{i}": float(flat[i]) if i < len(flat) else None for i in range(6)}

    def write_step(
        self, vec_step, policy_actions, pre_constraint_actions, executed_actions,
        rewards, dones, infos, base_env, colon_id, task_id,
    ):
        camera_data = base_env.robot.egocamera.data
        rgb_batch = self._rgb_batch(camera_data)
        positions = np.asarray(numpy_value(camera_data.pos_w), dtype=np.float64) - self.scene_origins
        quaternions = normalize_quaternion_wxyz(numpy_value(camera_data.quat_w_ros))
        policy = _action_array(policy_actions, len(infos))
        pre_constraint = _action_array(pre_constraint_actions, len(infos))
        executed = _action_array(executed_actions, len(infos))
        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        dones = np.asarray(dones, dtype=bool).reshape(-1)

        for env_idx, info in enumerate(infos):
            image_path = ""
            frame_valid = rgb_batch is not None and env_idx < len(rgb_batch)
            if frame_valid and self.save_rgb:
                image_path, frame_valid = self._save_rgb(env_idx, vec_step, rgb_batch[env_idx])
            pose_valid = (
                frame_valid
                and env_idx < len(positions)
                and env_idx < len(quaternions)
                and np.isfinite(positions[env_idx]).all()
                and np.isfinite(quaternions[env_idx]).all()
            )
            frame_valid = bool(pose_valid)
            camera_position = positions[env_idx] if pose_valid else np.zeros(3, dtype=np.float64)
            camera_quaternion = (
                quaternions[env_idx]
                if pose_valid
                else np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
            )
            episode_id = self.env_episode_ids[env_idx]
            previous_valid = (
                self.previous_sample_valid.get(env_idx, False)
                and self.previous_episode_ids.get(env_idx) == episode_id
                and bool(self.previous_image_paths.get(env_idx))
            )
            relative_valid = bool(pose_valid and previous_valid)
            if relative_valid:
                delta_position, delta_quaternion = relative_pose(
                    *self.previous_camera_poses[env_idx], camera_position, camera_quaternion
                )
            else:
                delta_position = np.zeros(3)
                delta_quaternion = np.asarray((1.0, 0.0, 0.0, 0.0))
            condition = self.geometry.camera_condition(camera_position, camera_quaternion)
            normalized_reward = _info_float(info, "normalized_step_reward")
            if normalized_reward is None:
                normalized_reward = float(np.clip(rewards[env_idx], -1, 1))
            constraint_active = False
            if pre_constraint is not None and executed is not None:
                constraint_active = bool(np.max(np.abs(pre_constraint[env_idx] - executed[env_idx])) > 1e-9)
            episode_step = self.env_episode_steps[env_idx]
            row = {
                "sample_index": self.env_sample_indices[env_idx], "vec_step": int(vec_step),
                "env_idx": env_idx, "episode_id": episode_id, "episode_step": episode_step,
                "rgb_path": image_path,
                "frame_valid": frame_valid, "timestamp_s": float(vec_step * self.action_duration),
                "delta_time_s": self.action_duration if relative_valid else 0.0,
                "relative_pose_valid": relative_valid,
                "delta_camera_x": delta_position[0], "delta_camera_y": delta_position[1], "delta_camera_z": delta_position[2],
                "delta_camera_quat_w": delta_quaternion[0], "delta_camera_quat_x": delta_quaternion[1],
                "delta_camera_quat_y": delta_quaternion[2], "delta_camera_quat_z": delta_quaternion[3],
                "camera_world_x": camera_position[0], "camera_world_y": camera_position[1], "camera_world_z": camera_position[2],
                "camera_world_quat_w": camera_quaternion[0], "camera_world_quat_x": camera_quaternion[1],
                "camera_world_quat_y": camera_quaternion[2], "camera_world_quat_z": camera_quaternion[3],
                "centerline_closest_camera_x": condition["closest_camera"][0],
                "centerline_closest_camera_y": condition["closest_camera"][1],
                "centerline_closest_camera_z": condition["closest_camera"][2],
                "centerline_lateral_camera_x": condition["lateral_camera"][0],
                "centerline_lateral_camera_y": condition["lateral_camera"][1],
                "centerline_lateral_camera_z": condition["lateral_camera"][2],
                "centerline_lateral_distance_m": condition["lateral_distance_m"],
                "centerline_tangent_camera_x": condition["tangent_camera"][0],
                "centerline_tangent_camera_y": condition["tangent_camera"][1],
                "centerline_tangent_camera_z": condition["tangent_camera"][2],
                "centerline_progress_m": condition["progress_m"],
                "centerline_progress_normalized": condition["progress_normalized"],
                "centerline_remaining_m": condition["remaining_m"],
                "action_constraint_active": constraint_active,
                "normalized_reward": normalized_reward, "raw_step_reward": _info_float(info, "raw_step_reward"),
                "env_reward": float(rewards[env_idx]), "roi_relative_x": _info_float(info, "roi_relative_x"),
                "roi_relative_y": _info_float(info, "roi_relative_y"), "roi_center_x_px": _info_float(info, "roi_center_x_px"),
                "roi_center_y_px": _info_float(info, "roi_center_y_px"),
                "roi_distance_from_center": _info_float(info, "roi_distance_from_center"),
                "center_alignment": _info_float(info, "center_alignment"), "roi_aligned": _info_float(info, "roi_aligned"),
                "lumen_visible": _info_float(info, "lumen_visible"), "goal_reached": bool(info.get("goal_reached", False)),
                "colon_id": colon_id, "task_id": task_id,
            }
            for idx, point in enumerate(condition["lookahead_camera"]):
                for axis_idx, axis in enumerate(("x", "y", "z")):
                    row[f"centerline_lookahead_{idx}_{axis}"] = float(point[axis_idx])
                row[f"centerline_lookahead_{idx}_valid"] = bool(condition["lookahead_valid"][idx])
            row.update(self._action_fields("action", None if executed is None else executed[env_idx]))
            row.update(self._action_fields("policy_action", None if policy is None else policy[env_idx]))
            self.writers[env_idx].writerow(row)
            self.sample_index += 1
            self.env_sample_indices[env_idx] += 1
            self.env_episode_steps[env_idx] += 1
            if pose_valid and not (env_idx < len(dones) and dones[env_idx]):
                self.previous_camera_poses[env_idx] = (positions[env_idx].copy(), quaternions[env_idx].copy())
                self.previous_sample_valid[env_idx] = True
                self.previous_episode_ids[env_idx] = episode_id
                self.previous_image_paths[env_idx] = image_path
            else:
                self.previous_camera_poses.pop(env_idx, None)
                self.previous_sample_valid[env_idx] = False
                self.previous_image_paths[env_idx] = ""
            if env_idx < len(dones) and dones[env_idx]:
                self.env_episode_ids[env_idx] += 1
                self.env_episode_steps[env_idx] = 0
            if self.env_sample_indices[env_idx] % self.flush_interval == 0:
                self.csv_files[env_idx].flush()

    def close(self):
        for stream in self.csv_files.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        self.metadata["completed_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        self.metadata["frames_written"] = self.sample_index
        _atomic_json(self.run_dir / "metadata.json", self.metadata)
        self.csv_files, self.writers = {}, {}
