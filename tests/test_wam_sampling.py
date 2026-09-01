import csv
import json
import math
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from arcgym.utils.wam_sampling import (
    CenterlineGeometry,
    RGBWAMDatasetWriter,
    inverse_rotate_vectors_wxyz,
    normalize_quaternion_wxyz,
    quaternion_multiply_wxyz,
    quaternion_to_matrix_wxyz,
    relative_pose,
)


def axis_quaternion(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    return np.concatenate(([math.cos(angle / 2)], axis * math.sin(angle / 2)))


class GeometryTests(unittest.TestCase):
    def test_ros_optical_identity_axes(self):
        # In ROS optical coordinates these basis vectors mean right, down, forward.
        basis = np.eye(3)
        np.testing.assert_allclose(inverse_rotate_vectors_wxyz([1, 0, 0, 0], basis), basis)

    def test_camera_to_world_inversion(self):
        q = axis_quaternion([0, 0, 1], math.pi / 2)
        world = np.array([0, 1, 0])
        np.testing.assert_allclose(inverse_rotate_vectors_wxyz(q, world), [1, 0, 0], atol=1e-8)

    def test_relative_pose_definition(self):
        q = axis_quaternion([0, 0, 1], math.pi / 2)
        translation, rotation = relative_pose([0, 0, 0], q, [0, 1, 0], q)
        np.testing.assert_allclose(translation, [1, 0, 0], atol=1e-8)
        np.testing.assert_allclose(rotation, [1, 0, 0, 0], atol=1e-8)

    def test_zero_motion(self):
        translation, rotation = relative_pose([1, 2, 3], [1, 0, 0, 0], [1, 2, 3], [1, 0, 0, 0])
        np.testing.assert_allclose(translation, 0)
        np.testing.assert_allclose(rotation, [1, 0, 0, 0])

    def test_forward_translation(self):
        translation, _ = relative_pose([0, 0, 0], [1, 0, 0, 0], [0, 0, 2], [1, 0, 0, 0])
        np.testing.assert_allclose(translation, [0, 0, 2])

    def test_right_translation(self):
        translation, _ = relative_pose([0, 0, 0], [1, 0, 0, 0], [2, 0, 0], [1, 0, 0, 0])
        np.testing.assert_allclose(translation, [2, 0, 0])

    def test_yaw_and_pitch(self):
        yaw = axis_quaternion([0, 1, 0], 0.2)  # optical-frame yaw about down axis
        pitch = axis_quaternion([1, 0, 0], -0.3)  # optical-frame pitch about right axis
        target = quaternion_multiply_wxyz(yaw, pitch)
        _, relative = relative_pose([0, 0, 0], [1, 0, 0, 0], [0, 0, 0], target)
        np.testing.assert_allclose(relative, normalize_quaternion_wxyz(target), atol=1e-8)

    def test_world_point_to_camera(self):
        q = axis_quaternion([0, 0, 1], math.pi / 2)
        point_camera = inverse_rotate_vectors_wxyz(q, np.array([0, 2, 0]) - np.array([0, 1, 0]))
        np.testing.assert_allclose(point_camera, [1, 0, 0], atol=1e-8)

    def test_straight_projection(self):
        geometry = CenterlineGeometry([[0, 0, 0], [0, 0, 2]], [0.1, 0.5])
        s, closest, tangent = geometry.project([1, 0, 0.75])
        self.assertAlmostEqual(s, 0.75)
        np.testing.assert_allclose(closest, [0, 0, 0.75])
        np.testing.assert_allclose(tangent, [0, 0, 1])

    def test_curved_projection(self):
        geometry = CenterlineGeometry([[0, 0, 0], [1, 0, 0], [1, 1, 0]], [0.2])
        s, closest, _ = geometry.project([1.2, 0.4, 0])
        self.assertAlmostEqual(s, 1.4)
        np.testing.assert_allclose(closest, [1, 0.4, 0])

    def test_arc_lookahead(self):
        geometry = CenterlineGeometry([[0, 0, 0], [0, 0, 1], [0, 0, 2]], [0.25, 0.75])
        condition = geometry.camera_condition([0, 0, 0.5], [1, 0, 0, 0])
        np.testing.assert_allclose(condition["lookahead_camera"][:, 2], [0.25, 0.75])
        self.assertTrue(condition["lookahead_valid"].all())

    def test_endpoint_clamp_and_validity(self):
        geometry = CenterlineGeometry([[0, 0, 0], [0, 0, 1]], [0.05, 0.2])
        condition = geometry.camera_condition([0, 0, 0.9], [1, 0, 0, 0])
        np.testing.assert_array_equal(condition["lookahead_valid"], [True, False])
        np.testing.assert_allclose(condition["lookahead_camera"][:, 2], [0.05, 0.1])


class WriterTests(unittest.TestCase):
    def _fixture(self, root):
        centerline = Path(root) / "centerline.csv"
        centerline.write_text("x,y,z\n0,0,0\n0,0,100\n")
        camera_data = types.SimpleNamespace(
            pos_w=np.array([[0.0, 0.0, 0.2], [5.0, 0.0, 0.2]]),
            quat_w_ros=np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]),
            intrinsic_matrices=np.repeat(np.array([[[100.0, 0, 2], [0, 100.0, 2], [0, 0, 1]]]), 2, axis=0),
            output={"rgb": np.full((2, 4, 4, 4), [12, 34, 56, 255], dtype=np.uint8)},
        )
        base_env = types.SimpleNamespace(
            robot=types.SimpleNamespace(egocamera=types.SimpleNamespace(data=camera_data)),
            scene=types.SimpleNamespace(env_origins=np.array([[0, 0, 0], [5, 0, 0]], dtype=float)),
            colon=types.SimpleNamespace(
                colon_body=types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=np.array([[0, 0, 0], [5, 0, 0]], dtype=float)))
            ),
        )
        config = {
            "env_config": {"num_envs": 2, "colon_init_rot": (1, 0, 0, 0), "translation_action_scale": 0.001, "rotation_action_scale": 0.01},
            "robot_config": {"robot_type": "magnetic_endoscope", "front_camera_clipping_range": (0.01, 5.0)},
            "simulation_config": {"dt": 1 / 240, "render_interval": 2},
            "run_config": {"seed": 0, "colon_id": "c1", "task_id": "t1", "model_path": None},
        }
        writer = RGBWAMDatasetWriter(root, "result", Path(root) / "run", config, 6, base_env, centerline, 0.01, [0.1, 0.2])
        return writer, base_env

    def test_rgb_png_storage_without_depth(self):
        with tempfile.TemporaryDirectory() as root:
            writer, base = self._fixture(root)
            actions = np.zeros((2, 6))
            writer.write_step(
                0, actions, actions, actions, np.zeros(2), [False, False], [{}, {}], base, "c1", "t1"
            )
            writer.close()
            image = np.asarray(Image.open(Path(root) / "run/env_000/rgb/frame_00000000.png"))
            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(image.shape, (4, 4, 3))
            np.testing.assert_array_equal(image[0, 0], [12, 34, 56])
            self.assertFalse((Path(root) / "run/env_000/depth").exists())
            metadata = json.loads((Path(root) / "run/metadata.json").read_text())
            self.assertEqual(metadata["schema"]["name"], "arcgym_centerline_rgb_wam")
            self.assertNotIn("depth", metadata)

    def test_independent_reset_and_action_indexing(self):
        with tempfile.TemporaryDirectory() as root:
            writer, base = self._fixture(root)
            raw = np.full((2, 6), 0.5)
            pre = np.full((2, 6), 0.01)
            executed = pre.copy()
            kwargs = dict(policy_actions=raw, pre_constraint_actions=pre, executed_actions=executed, rewards=np.zeros(2), infos=[{}, {}], base_env=base, colon_id="c1", task_id="t1")
            writer.write_step(0, dones=[False, False], **kwargs)
            base.robot.egocamera.data.pos_w[:, 2] += 0.1
            writer.write_step(1, dones=[True, False], **kwargs)
            base.robot.egocamera.data.pos_w[:, 2] += 0.1
            executed[1, 0] = 0.0
            writer.write_step(2, dones=[False, False], executed_actions=executed, **{k: v for k, v in kwargs.items() if k != "executed_actions"})
            writer.close()
            with open(Path(root) / "run/env_000/samples.csv", newline="") as stream:
                rows0 = list(csv.DictReader(stream))
            with open(Path(root) / "run/env_001/samples.csv", newline="") as stream:
                rows1 = list(csv.DictReader(stream))
            self.assertEqual(rows0[2]["relative_pose_valid"], "False")
            self.assertEqual(rows0[2]["episode_id"], "1")
            self.assertEqual(rows1[2]["relative_pose_valid"], "True")
            self.assertEqual(rows1[1]["action_0"], "0.01")
            self.assertEqual(rows1[1]["policy_action_0"], "0.5")
            self.assertEqual(rows1[2]["action_constraint_active"], "True")

    def test_action_frame_index_contract(self):
        with tempfile.TemporaryDirectory() as root:
            writer, base = self._fixture(root)
            first_action = np.zeros((2, 6))
            second_action = np.full((2, 6), 0.02)
            common = dict(rewards=np.zeros(2), dones=[False, False], infos=[{}, {}], base_env=base, colon_id="c1", task_id="t1")
            writer.write_step(0, first_action, first_action, first_action, **common)
            base.robot.egocamera.data.pos_w[:, 2] += 0.1
            writer.write_step(1, second_action, second_action, second_action, **common)
            writer.close()
            with open(Path(root) / "run/env_000/samples.csv", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["relative_pose_valid"], "False")
            self.assertEqual(rows[1]["relative_pose_valid"], "True")
            self.assertEqual(float(rows[1]["action_0"]), 0.02)
            self.assertAlmostEqual(float(rows[1]["delta_camera_z"]), 0.1)


if __name__ == "__main__":
    unittest.main()
