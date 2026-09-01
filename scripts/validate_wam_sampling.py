#!/usr/bin/env python3
"""Validate a completed ARCGym centerline-conditioned RGB-WAM sampling run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arcgym.utils.wam_sampling import (  # noqa: E402
    CenterlineGeometry,
    normalize_quaternion_wxyz,
    quaternion_to_matrix_wxyz,
    relative_pose,
)


def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def vector(row, prefix, quaternion=False):
    axes = ("w", "x", "y", "z") if quaternion else ("x", "y", "z")
    return np.asarray([float(row[f"{prefix}_{axis}"]) for axis in axes], dtype=np.float64)


def finite_columns(row, columns):
    return all(np.isfinite(float(row[column])) for column in columns if row.get(column, "") != "")


class Validation:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def require(self, condition, message):
        if not condition:
            self.errors.append(message)

    def warn(self, condition, message):
        if not condition:
            self.warnings.append(message)


def debug_plot(run_dir, geometry, env_idx, rows, count):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib unavailable; skipping debug plots", file=sys.stderr)
        return
    output_dir = run_dir / "debug_validation"
    output_dir.mkdir(exist_ok=True)
    valid_rows = [row for row in rows if as_bool(row["frame_valid"]) and row["rgb_path"]][:count]
    for row in valid_rows:
        rgb = np.asarray(Image.open(run_dir / f"env_{env_idx:03d}" / row["rgb_path"]))
        position = vector(row, "camera_world")
        rotation = quaternion_to_matrix_wxyz(vector(row, "camera_world_quat", quaternion=True))
        closest_camera = vector(row, "centerline_closest_camera")
        lookahead = []
        idx = 0
        while f"centerline_lookahead_{idx}_x" in row:
            lookahead.append(vector(row, f"centerline_lookahead_{idx}"))
            idx += 1
        lookahead = np.asarray(lookahead)
        fig = plt.figure(figsize=(12, 5))
        ax_rgb = fig.add_subplot(121)
        ax_rgb.imshow(rgb)
        ax_rgb.set_title("RGB camera frame")
        ax_world = fig.add_subplot(122, projection="3d")
        ax_world.plot(*geometry.points.T, color="black", linewidth=1)
        ax_world.scatter(*position, color="red", label="camera")
        row_index = rows.index(row)
        if as_bool(row["relative_pose_valid"]) and row_index > 0:
            previous_position = vector(rows[row_index - 1], "camera_world")
            ax_world.scatter(*previous_position, color="magenta", label="previous camera")
            ax_world.plot(*np.column_stack((previous_position, position)), color="magenta", linestyle="--")
        closest_world = position + rotation @ closest_camera
        ax_world.scatter(*closest_world, color="blue", label="closest")
        for axis, color in zip(rotation.T, ("r", "g", "b")):
            end = position + 0.1 * axis
            ax_world.plot(*np.column_stack((position, end)), color=color)
        lookahead_world = position + lookahead @ rotation.T
        ax_world.plot(*lookahead_world.T, "o-", color="orange", label="lookahead")
        ax_world.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"env_{env_idx:03d}_frame_{int(row['vec_step']):08d}.png", dpi=140)
        plt.close(fig)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(0, 0, 0, color="red")
        ax.plot(lookahead[:, 0], lookahead[:, 1], lookahead[:, 2], "o-")
        ax.set_xlabel("+x right")
        ax.set_ylabel("+y down")
        ax.set_zlabel("+z forward")
        ax.set_title("ROS optical camera frame")
        fig.tight_layout()
        fig.savefig(output_dir / f"env_{env_idx:03d}_camera_{int(row['vec_step']):08d}.png", dpi=140)
        plt.close(fig)


def validate(run_dir, debug_plots=0):
    check = Validation()
    metadata_path = run_dir / "metadata.json"
    geometry_path = run_dir / "geometry" / "centerline_world.npz"
    check.require(metadata_path.is_file(), "metadata.json is missing")
    check.require(geometry_path.is_file(), "geometry/centerline_world.npz is missing")
    if check.errors:
        return check, {}
    metadata = json.loads(metadata_path.read_text())
    geometry_data = np.load(geometry_path)
    required_geometry = {
        "points_world_m", "arc_length_m", "progress_normalized", "tangent_world",
        "curvature_vector_world", "curvature_magnitude", "rmf_normal1_world", "rmf_normal2_world",
    }
    check.require(required_geometry.issubset(geometry_data.files), "global centerline geometry keys are incomplete")
    offsets = metadata["centerline"]["lookahead_arc_offsets_m"]
    geometry = CenterlineGeometry(geometry_data["points_world_m"], offsets)
    check.require(
        np.allclose(geometry.arc, geometry_data["arc_length_m"], atol=1e-8),
        "centerline arc lengths do not match points",
    )
    check.require(metadata["centerline"]["number_of_samples"] == len(geometry.points), "centerline sample count mismatch")
    check.require(metadata["rgb"]["storage_format"] == "uint8_png", "RGB storage format is not uint8 PNG")
    height, width = metadata["rgb"]["height"], metadata["rgb"]["width"]

    all_rows, environment_rows = [], {}
    translation_norms, rotation_angles, action_values = [], [], []
    progress_values, lateral_values = [], []
    lookahead_valid = [[] for _ in offsets]
    episode_keys = set()
    valid_transitions = 0
    required_columns = {
        "sample_index", "vec_step", "env_idx", "episode_id", "episode_step", "rgb_path",
        "timestamp_s", "delta_time_s", "relative_pose_valid",
        "camera_world_x", "camera_world_quat_w", "centerline_lateral_camera_x",
        "action_constraint_active", "action_0", "policy_action_0",
    }
    for env_idx in range(int(metadata["num_envs"])):
        env_dir = run_dir / f"env_{env_idx:03d}"
        csv_path = env_dir / "samples.csv"
        check.require(csv_path.is_file(), f"env {env_idx}: samples.csv missing")
        if not csv_path.is_file():
            continue
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            check.require(required_columns.issubset(reader.fieldnames or ()), f"env {env_idx}: CSV schema incomplete")
            rows = list(reader)
        environment_rows[env_idx] = rows
        previous = None
        for row_number, row in enumerate(rows):
            label = f"env {env_idx} row {row_number}"
            all_rows.append(row)
            episode = int(row["episode_id"])
            episode_step = int(row["episode_step"])
            episode_keys.add((env_idx, episode))
            frame_valid = as_bool(row["frame_valid"])
            if frame_valid:
                rgb_path = env_dir / row["rgb_path"]
                check.require(rgb_path.is_file(), f"{label}: RGB path missing")
                if rgb_path.is_file():
                    rgb = np.asarray(Image.open(rgb_path))
                    check.require(rgb.shape == (height, width, 3), f"{label}: wrong RGB dimensions")
                    check.require(rgb.dtype == np.uint8, f"{label}: RGB is not uint8 PNG")

            position = vector(row, "camera_world")
            quaternion = vector(row, "camera_world_quat", quaternion=True)
            check.require(np.isfinite(position).all(), f"{label}: nonfinite absolute position")
            check.require(np.isfinite(quaternion).all(), f"{label}: nonfinite absolute quaternion")
            check.require(abs(np.linalg.norm(quaternion) - 1.0) < 1e-5, f"{label}: non-unit absolute quaternion")
            relative_valid = as_bool(row["relative_pose_valid"])
            if episode_step == 0:
                check.require(not relative_valid, f"{label}: episode start has valid relative pose")
            if relative_valid:
                check.require(previous is not None, f"{label}: valid transition lacks previous row")
                if previous is not None:
                    check.require(int(previous["episode_id"]) == episode, f"{label}: transition crosses episode")
                    expected_t, expected_q = relative_pose(
                        vector(previous, "camera_world"), vector(previous, "camera_world_quat", True),
                        position, quaternion,
                    )
                    stored_t = vector(row, "delta_camera")
                    stored_q = vector(row, "delta_camera_quat", True)
                    check.require(np.allclose(stored_t, expected_t, atol=1e-7), f"{label}: relative translation mismatch")
                    check.require(
                        min(np.linalg.norm(stored_q - expected_q), np.linalg.norm(stored_q + expected_q)) < 1e-7,
                        f"{label}: relative rotation mismatch",
                    )
                    translation_norms.append(float(np.linalg.norm(stored_t)))
                    rotation_angles.append(float(2 * math.acos(np.clip(abs(stored_q[0]), 0, 1))))
                    valid_transitions += 1
            progress = float(row["centerline_progress_m"])
            normalized = float(row["centerline_progress_normalized"])
            remaining = float(row["centerline_remaining_m"])
            lateral = float(row["centerline_lateral_distance_m"])
            check.require(0 <= progress <= geometry.length_m + 1e-7, f"{label}: invalid progress")
            check.require(0 <= normalized <= 1 + 1e-7, f"{label}: invalid normalized progress")
            check.require(remaining >= -1e-7, f"{label}: negative remaining distance")
            expected = geometry.camera_condition(position, quaternion)
            check.require(
                np.allclose(vector(row, "centerline_closest_camera"), expected["closest_camera"], atol=1e-6),
                f"{label}: camera-relative closest point mismatch",
            )
            check.require(
                np.allclose(vector(row, "centerline_lateral_camera"), expected["lateral_camera"], atol=1e-6),
                f"{label}: lateral vector mismatch",
            )
            for idx in range(len(offsets)):
                stored_point = vector(row, f"centerline_lookahead_{idx}")
                stored_valid = as_bool(row[f"centerline_lookahead_{idx}_valid"])
                check.require(np.allclose(stored_point, expected["lookahead_camera"][idx], atol=1e-6), f"{label}: lookahead {idx} mismatch")
                check.require(stored_valid == bool(expected["lookahead_valid"][idx]), f"{label}: lookahead {idx} validity mismatch")
                lookahead_valid[idx].append(stored_valid)
            action = np.asarray([float(row[f"action_{idx}"]) for idx in range(6)])
            check.require(np.isfinite(action).all(), f"{label}: nonfinite action")
            action_values.append(action)
            progress_values.append(progress)
            lateral_values.append(lateral)
            previous = row if frame_valid else None
        if debug_plots:
            debug_plot(run_dir, geometry, env_idx, rows, debug_plots)

    check.require(metadata["rgb"]["width"] == metadata["camera"]["intrinsics"]["width"], "camera/RGB width mismatch")
    check.require(metadata["rgb"]["height"] == metadata["camera"]["intrinsics"]["height"], "camera/RGB height mismatch")
    check.require(metadata.get("frames_written", len(all_rows)) == len(all_rows), "metadata frame count mismatch")
    statistics = {
        "number_of_environments": len(environment_rows),
        "number_of_episodes": len(episode_keys),
        "number_of_frames": len(all_rows),
        "number_of_valid_transitions": valid_transitions,
        "translation_delta_m": describe(translation_norms),
        "rotation_delta_rad": describe(rotation_angles),
        "action": describe(np.asarray(action_values).reshape(-1) if action_values else []),
        "centerline_progress_m": describe(progress_values),
        "lateral_distance_m": describe(lateral_values),
        "lookahead_validity_rates": [float(np.mean(values)) if values else None for values in lookahead_valid],
    }
    return check, statistics


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return None
    return {
        "min": float(np.min(values)), "mean": float(np.mean(values)),
        "max": float(np.max(values)), "std": float(np.std(values)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--debug-plots", type=int, default=0, help="Write plots for up to N rows per environment")
    args = parser.parse_args()
    check, statistics = validate(args.run_directory.resolve(), max(0, args.debug_plots))
    print(json.dumps(statistics, indent=2))
    for warning in check.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in check.errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if check.errors:
        raise SystemExit(1)
    print("Validation passed")


if __name__ == "__main__":
    main()
