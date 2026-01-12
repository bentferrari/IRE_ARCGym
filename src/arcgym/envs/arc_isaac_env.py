from __future__ import annotations

from collections.abc import Sequence
import logging

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg

from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
import isaacsim.core.utils.stage as stage_utils
import pdb

from arcgym.assets.colons.colon_isaac import ColonModelCfg, ColonModel
from arcgym.assets.robots.robot_factory import RobotFactory
from arcgym.rewards.rewards import (
    GoalReward,
    TestRewardAction,
    TestRewardPosition,
    PointToPointLineDistanceReward,
    DepthDistanceReward,
    FinalReward,
    )
from arcgym.utils.debug import create_simple_sphere_markers

import gymnasium as gym
from gymnasium.spaces import Dict, Box

class ARCEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: ARCEnvWindow, window_name: str = "ARCEnv"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        # with self.ui_window_elements["main_vstack"]:
        #     with self.ui_window_elements["debug_frame"]:
        #         with self.ui_window_elements["debug_vstack"]:
        #             # add command manager visualization
        #             self._create_debug_vis_ui_element("targets", self.env)

def make_isaac_env_cfg(config: dict, robot_factory) -> ARCIsaacEnvCfg:
    env_config = config["env_config"]
    simulation_config = config["simulation_config"]
    render_config = config["render_config"]
    physx_config = config["physx_config"]
    @configclass
    class ARCIsaacEnvCfg(DirectRLEnvCfg):
        decimation = simulation_config["render_interval"]
        episode_length_s = env_config["episode_length_s"]
        action_scale = env_config["action_scale"]
        debug_vis = env_config["debug_vis"]
        num_envs: int = env_config["num_envs"]
        #pdb.set_trace()
        action_space = robot_factory.action_space # Can the action_space also be determined by the env, if so, add it here. # previous value: 6
        #pdb.set_trace()
        observation_space = robot_factory.observation_space # Can the observation_space also be determined by the env, if so, add it here.


        ui_window_class_type = ARCEnvWindow

        # Ground plane removed - no terrain in scene
        # terrain = TerrainImporterCfg(
        #     prim_path="/World/ground",
        #     terrain_type="plane",
        #     collision_group=-1,
        #     physics_material=sim_utils.RigidBodyMaterialCfg(
        #         friction_combine_mode="average",
        #         restitution_combine_mode="average",
        #         static_friction=1.0,
        #         dynamic_friction=1.0,
        #         restitution=0.0,
        #     ),
        #     debug_vis=debug_vis,
        # )

        scene: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=env_config["num_envs"],
            env_spacing=env_config["env_spacing"],
            replicate_physics=env_config["replicate_physics"],
            )  #replicate_physics seems not supported for PhysicsMaterial and Mesh??

        render_cfg = sim_utils.RenderCfg(**render_config)

        sim: SimulationCfg = SimulationCfg(
            render=render_cfg,
            physx=sim_utils.PhysxCfg(**physx_config),
            **simulation_config,
            )
        
        # colon model
        colon_cfg = ColonModelCfg()
    
    return ARCIsaacEnvCfg()

class CameraController:
    def __init__(self, camera, render_interval):
        self.camera = camera
        self.render_interval = render_interval
        self.step_count = 0
        
    def physics_callback(self, step_size):
        # Enable camera only on render steps
        should_render = (self.step_count % self.render_interval) == 0
        
        # Method 1: Toggle camera visibility
        # self.camera._view.set_visibility(should_render)
        
        # # Method 2: Or toggle the sensor directly
        if hasattr(self.camera._sensor_prims, 'set_enabled'):
            for prim in self.camera._sensor_prims:
                prim.set_enabled(should_render)
                
        self.step_count += 1

class ARCIsaacEnv(DirectRLEnv):
    cfg: ARCIsaacEnvCfg

    def __init__(self, cfg: ARCIsaacEnvCfg, robot_factory: RobotFactory, config: dict, **kwargs):
        self.robot_factory = robot_factory
        self.robot_config = config["robot_config"]
        self.config = config
        self.env_config = config["env_config"]
        self.reward_config = config["reward_config"]
        self.render_config = config["render_config"]
        self.debug_config = config["debug_config"]

        self.discrete = self.env_config["discrete_action_space"]
        self.use_pose_in_obs = self.env_config["use_pose"]
        self.use_camera = self.env_config["use_camera"]
        self.render_mode = self.env_config["render_mode"]
        reward_type = self.reward_config["reward_type"]

        self.show_markers = self.debug_config["show_markers"]
            
        super().__init__(cfg, self.render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale

        self.observation_space = robot_factory.observation_space
        self.action_space = robot_factory.action_space

        self.set_debug_vis(self.cfg.debug_vis)

        if reward_type == "default":
            self.reward_function = GoalReward(**self.reward_config)
        elif reward_type == "test_reward_action":
            self.reward_function = TestRewardAction(**self.reward_config)
        elif reward_type == "test_reward_position":
            self.reward_function = TestRewardPosition(**self.reward_config)
        elif reward_type == "line_goal_points":
            self.reward_function = PointToPointLineDistanceReward(**self.reward_config)
        elif reward_type == "depth_goal":
            self.reward_function = DepthDistanceReward(**self.reward_config)
        elif reward_type == "final_reward":
            self.reward_function = FinalReward(**self.reward_config)
        else:
            raise ValueError(f"Unknown reward type: '{reward_type}'")

        self.tracer = None
        self.debug_resets_remaining = -1

        if "tracer" in kwargs and kwargs["tracer"] is not None:
            logging.info("Adding tracer")
            self.tracer = kwargs["tracer"]
            self.debug_resets_remaining = 12

        self.markers = None
        self.previous_positions = None
        self.previous_states = None
        self.latest_rewards = None
        self.latest_l2_norm = 1.0 # tmp fix since l2_norm is not currently updated

        self.goal_reached = torch.tensor([False]*self.num_envs, device=self.device)
        self.hit_wall = torch.tensor([False]*self.num_envs, device=self.device)
        self.truncate_now = torch.tensor([False]*self.num_envs, device=self.device)
        self.colon_invalid = torch.tensor([False]*self.num_envs, device=self.device)
        self.colon_invalid_prev = torch.tensor([False]*self.num_envs, device=self.device)
        self.poor_alignment = torch.tensor([False]*self.num_envs, device=self.device)

        # Stuck detection and backward motion tracking
        self.previous_positions = None
        self.poor_alignment_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.backward_steps_remaining = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.stuck_threshold = 15  # Number of steps with poor alignment to consider stuck
        self.backward_duration = 10  # Number of steps to move backward when stuck
        self.alignment_threshold = 0.7  # Center alignment below this is considered poor

        # Bright region avoidance tracking
        self.bright_region_directions = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.has_bright_regions = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.sim.set_camera_view(eye=[0.5, 1.5, 0.5], target=[0.0, 0.0, 0.5])

    def _setup_scene(self):
        self.stage = stage_utils.get_current_stage()
        robot_type = self.robot_config.get("robot_type", None)
        if robot_type == "capsule":
            init_rot = (1, 0, 0, 0)
        else:
            # Check if we should load rotation from CSV
            csv_init_file = self.config.get("env_config", {}).get("init_from_csv", None)
            if csv_init_file is not None and csv_init_file.endswith('.csv'):
                import pandas as pd
                import os

                if os.path.exists(csv_init_file):
                    df = pd.read_csv(csv_init_file)
                    # Get the first row's quaternion values (w, x, y, z)
                    first_row = df.iloc[0]
                    init_rot = (
                        first_row['root_quat_w'],
                        first_row['root_quat_x'],
                        first_row['root_quat_y'],
                        first_row['root_quat_z']
                    )
                    logging.info(f"Loaded initial rotation from CSV: {init_rot}")
                else:
                    logging.warning(f"CSV file not found: {csv_init_file}, using default rotation")
                    init_rot = (0, 1, 0, 1)
            else:
                init_rot = (0, 1, 0, 1)
        self.robot = self.robot_factory.build_robot(
            scene=self.scene,
            init_pos=(0.22, 0.16, 0.4),
            init_rot=init_rot,
            )
        # endoscope init position x=7.4029, y=0.5015, z=0.5500
        self.colon = ColonModel(self.scene, cfg=self.cfg.colon_cfg, cfg1=self.config,  init_pos=(0.5,0.5,0.1), #init_rot=(0.707, 0.0, -0.707, 0.0), init_rot=(1.0, 0.0, 0.0, 0.0)
        init_rot=(0.707, 0.0, -0.707, 0.0), is_rigid=False)

        # Pass colon reference to robot for stress calculation
        if hasattr(self.robot, 'colon'):
            self.robot.colon = self.colon

        # Note: Keeping this here to show how to add a physics callback.
        # Register the callback
        #camera_controller = CameraController(self.robot.egocamera, self.cfg.decimation)

        #if self.sim.physics_callback_exists("camera_control"):
        #    self.sim.remove_physics_callback("camera_control")
        #self.sim.add_physics_callback("camera_control", camera_controller.physics_callback)

        # Terrain setup commented out - no ground plane in scene
        # self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        # self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        # self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.scene.cfg.filter_collisions:
            self.scene.filter_collisions(self.scene._global_prim_paths)

    def _compute_movement_metrics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute center alignment, lumen visibility, and escape directions from current depth images.
        Returns:
            center_alignments: Tensor of shape (num_envs,) with values in [0, 1]
            lumen_visibilities: Tensor of shape (num_envs,) with visibility scores
            escape_directions: Tensor of shape (num_envs, 2) with [y_offset, z_offset] to escape bright regions
        """
        current_states = self.get_states()
        depth_images = current_states["depth"]
        center_alignments = []
        lumen_visibilities = []
        escape_directions = []

        for depth_img in depth_images:
            depth_img_np = depth_img[:, :, 0].cpu().numpy()
            img_height, img_width = depth_img_np.shape
            img_center_row, img_center_col = img_height / 2, img_width / 2

            # Depth statistics
            max_depth = np.max(depth_img_np)

            # Calculate ratio of pixels with "good depth"
            GOOD_DEPTH_THRESHOLD = 0.12
            high_depth_pixels = np.sum(depth_img_np > GOOD_DEPTH_THRESHOLD)
            total_pixels = depth_img_np.size
            high_depth_ratio = high_depth_pixels / total_pixels

            # Find the deepest point location (highest depth value = farthest point)
            max_depth_idx = np.unravel_index(np.argmax(depth_img_np), depth_img_np.shape)
            max_depth_row, max_depth_col = max_depth_idx

            # Calculate max possible distance from center
            max_center_distance = np.sqrt(img_center_row**2 + img_center_col**2)

            # Distance from deepest point to image center
            deepest_point_deviation = np.sqrt((max_depth_row - img_center_row)**2 +
                                             (max_depth_col - img_center_col)**2)
            deepest_point_normalized = deepest_point_deviation / max_center_distance

            # Center alignment: 1.0 = perfectly centered, 0.0 = at edge
            center_alignment = 1.0 - deepest_point_normalized
            center_alignments.append(center_alignment)

            # Compute lumen visibility (same logic as in FinalReward)
            LUMEN_MIN_DEPTH = 0.20
            WALL_MAX_DEPTH = 0.10
            MIN_HIGH_DEPTH_RATIO = 0.40

            if max_depth < WALL_MAX_DEPTH:
                lumen_visibility = -1.0
            elif max_depth < LUMEN_MIN_DEPTH:
                base_penalty = -0.5 * (LUMEN_MIN_DEPTH - max_depth) / (LUMEN_MIN_DEPTH - WALL_MAX_DEPTH)
                if high_depth_ratio < MIN_HIGH_DEPTH_RATIO:
                    ratio_penalty = -0.3 * (MIN_HIGH_DEPTH_RATIO - high_depth_ratio) / MIN_HIGH_DEPTH_RATIO
                    lumen_visibility = base_penalty + ratio_penalty
                else:
                    lumen_visibility = base_penalty
            else:
                if high_depth_ratio < MIN_HIGH_DEPTH_RATIO:
                    lumen_visibility = -0.4 * (MIN_HIGH_DEPTH_RATIO - high_depth_ratio) / MIN_HIGH_DEPTH_RATIO
                elif high_depth_ratio < 0.25:
                    lumen_visibility = 0.1
                else:
                    lumen_visibility = 0.2

            lumen_visibilities.append(lumen_visibility)

            # Compute escape direction: find centroid of bright (open) pixels
            # Bright pixels = high depth values = far away = open lumen space
            # We want to move TOWARD these (to find open path)
            depth_threshold = np.percentile(depth_img_np, 75)  # Top 25% brightest pixels
            bright_mask = depth_img_np > depth_threshold

            if np.sum(bright_mask) > 0:
                # Find centroid of bright pixels (open space)
                rows, cols = np.where(bright_mask)
                bright_center_row = np.mean(rows)
                bright_center_col = np.mean(cols)

                # Calculate offset from image center to bright center
                offset_row = bright_center_row - img_center_row  # Positive = bright region is below
                offset_col = bright_center_col - img_center_col  # Positive = bright region is right

                # Escape direction is TOWARD the bright center (open lumen)
                # Normalize to [-1, 1] range
                escape_y = offset_col / (img_width / 2)   # Left/Right: toward col offset
                escape_z = offset_row / (img_height / 2)  # Up/Down: toward row offset
            else:
                # No clear bright region, no escape direction needed
                escape_y = 0.0
                escape_z = 0.0

            escape_directions.append([escape_y, escape_z])

        return (torch.tensor(center_alignments, dtype=torch.float32, device=depth_images.device),
                torch.tensor(lumen_visibilities, dtype=torch.float32, device=depth_images.device),
                torch.tensor(escape_directions, dtype=torch.float32, device=depth_images.device))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Apply different scaling for translation vs orientation
        # Actions: [0-2] translation (left/right, up/down, forward/back)
        #          [3-5] orientation (pitch, yaw, roll)
        scaled_actions = actions.clone()

        # Scale translation actions (indices 0, 1, 2)
        scaled_actions[:, :3] = scaled_actions[:, :3] * self.action_scale

        # Scale orientation actions (indices 3, 4, 5) with higher scale
        orientation_scale = self.action_scale * 1 # 3x higher for orientation
        scaled_actions[:, 3:] = scaled_actions[:, 3:] * orientation_scale

        # Store the scaled actions BEFORE clipping for reward computation
        # This ensures PPO trains on the actions it actually outputs
        self.actions = scaled_actions

    def _apply_movement_constraints(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Apply movement constraints based on quadrant of lumen centroid.

        When lumen visibility is poor (< -0.15), uses the deepest point position.
        Otherwise, uses the threshold_high centroid from the reward function.

        ALWAYS forces forward motion (v_forward_back = 0.1) and clips other actions based on quadrant:
        - upper-left: clip to go up and left (v_left_right ≤ 0, v_up_down ≤ 0, ω_pitch ≤ 0, ω_yaw ≤ 0)
        - upper-right: clip to go up and right (v_left_right ≥ 0, v_up_down ≤ 0, ω_pitch ≤ 0, ω_yaw ≥ 0)
        - lower-left: clip to go down and left (v_left_right ≤ 0, v_up_down ≥ 0, ω_pitch ≥ 0, ω_yaw ≤ 0)
        - lower-right: clip to go down and right (v_left_right ≥ 0, v_up_down ≥ 0, ω_pitch ≥ 0, ω_yaw ≥ 0)

        Constraints are only applied after 10,000 steps to allow initial exploration.

        Args:
            actions: Action tensor of shape (num_envs, 6)
                    [0] v_left_right, [1] v_up_down, [2] v_forward_back (OVERRIDDEN to 0.1),
                    [3] ω_pitch, [4] ω_yaw, [5] ω_roll

        Returns:
            Modified actions with constraints applied
        """
        # Check if movement constraints are disabled (e.g., for teleoperation mode)
        if self.env_config.get("disable_movement_constraints", False):
            return actions

        # Only apply constraints after 10,000 steps
        if self.common_step_counter < 100:
            return actions

        # Clone to avoid modifying input
        constrained_actions = actions.clone()

        # Compute lumen visibility to determine quadrant source
        if hasattr(self.robot, 'get_depth'):
            try:
                _, lumen_visibilities, _ = self._compute_movement_metrics()
            except Exception as e:
                logging.warning(f"Failed to compute lumen visibility: {e}")
                lumen_visibilities = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        else:
            lumen_visibilities = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Get quadrant information from reward function (using threshold_high centroid)
        if hasattr(self.reward_function, 'high_quadrants_per_env') and self.reward_function.high_quadrants_per_env:
            quadrants = self.reward_function.high_quadrants_per_env

            for env_id, quadrant in enumerate(quadrants):
                # Store original actions for printing
                original_action = actions[env_id].clone()

                # Override quadrant with deepest point position if lumen visibility is poor
                quadrant_source = "threshold_high"
                if lumen_visibilities[env_id] < -0.10:
                    # Poor visibility: compute quadrant from deepest point
                    try:
                        depth_images = self.robot.get_depth()
                        depth_img = depth_images[env_id]
                        depth_img_np = depth_img[:, :, 0].cpu().numpy()

                        img_height, img_width = depth_img_np.shape
                        img_center_row, img_center_col = img_height / 2, img_width / 2

                        # Find deepest point (maximum depth value)
                        max_depth_idx = np.unravel_index(np.argmax(depth_img_np), depth_img_np.shape)
                        max_depth_row, max_depth_col = max_depth_idx

                        # Determine quadrant based on deepest point
                        if max_depth_row < img_center_row and max_depth_col < img_center_col:
                            quadrant = "upper-left"
                        elif max_depth_row < img_center_row and max_depth_col >= img_center_col:
                            quadrant = "upper-right"
                        elif max_depth_row >= img_center_row and max_depth_col < img_center_col:
                            quadrant = "lower-left"
                        else:
                            quadrant = "lower-right"

                        quadrant_source = "deepest_point"
                    except Exception as e:
                        logging.warning(f"Failed to compute deepest point quadrant: {e}")

                # Clip forward/back motion to range [0.01, 0.05]
                constrained_actions[env_id, 2] = torch.clamp(constrained_actions[env_id, 2], min=0.05, max=0.10)

                # Always disable roll (set to 0)
                constrained_actions[env_id, 5] = 0.0

                if quadrant == "upper-left":
                    # Clip to go up and left: translation left/up, yaw left, pitch up
                    constrained_actions[env_id, 0] = torch.clamp(constrained_actions[env_id, 0], max=0)  # v_left_right <= 0 (left)
                    constrained_actions[env_id, 1] = torch.clamp(constrained_actions[env_id, 1], max=0)  # v_up_down <= 0 (up)
                    constrained_actions[env_id, 3] = torch.clamp(constrained_actions[env_id, 3], max=0)  # ω_pitch <= 0 (up)
                    constrained_actions[env_id, 4] = torch.clamp(constrained_actions[env_id, 4], max=0)  # ω_yaw <= 0 (left)

                elif quadrant == "upper-right":
                    # Clip to go up and right: translation right/up, yaw right, pitch up
                    constrained_actions[env_id, 0] = torch.clamp(constrained_actions[env_id, 0], min=0)  # v_left_right >= 0 (right)
                    constrained_actions[env_id, 1] = torch.clamp(constrained_actions[env_id, 1], max=0)  # v_up_down <= 0 (up)
                    constrained_actions[env_id, 3] = torch.clamp(constrained_actions[env_id, 3], max=0)  # ω_pitch <= 0 (up)
                    constrained_actions[env_id, 4] = torch.clamp(constrained_actions[env_id, 4], min=0)  # ω_yaw >= 0 (right)

                elif quadrant == "lower-left":
                    # Clip to go down and left: translation left/down, yaw left, pitch down
                    constrained_actions[env_id, 0] = torch.clamp(constrained_actions[env_id, 0], max=0)  # v_left_right <= 0 (left)
                    constrained_actions[env_id, 1] = torch.clamp(constrained_actions[env_id, 1], min=0)  # v_up_down >= 0 (down)
                    constrained_actions[env_id, 3] = torch.clamp(constrained_actions[env_id, 3], min=0)  # ω_pitch >= 0 (down)
                    constrained_actions[env_id, 4] = torch.clamp(constrained_actions[env_id, 4], max=0)  # ω_yaw <= 0 (left)

                elif quadrant == "lower-right":
                    # Clip to go down and right: translation right/down, yaw right, pitch down
                    constrained_actions[env_id, 0] = torch.clamp(constrained_actions[env_id, 0], min=0)  # v_left_right >= 0 (right)
                    constrained_actions[env_id, 1] = torch.clamp(constrained_actions[env_id, 1], min=0)  # v_up_down >= 0 (down)
                    constrained_actions[env_id, 3] = torch.clamp(constrained_actions[env_id, 3], min=0)  # ω_pitch >= 0 (down)
                    constrained_actions[env_id, 4] = torch.clamp(constrained_actions[env_id, 4], min=0)  # ω_yaw >= 0 (right)

                # Print quadrant and actions for each environment
                clipped_action = constrained_actions[env_id]
                lumen_vis = lumen_visibilities[env_id].item()
                # print(f"Env {env_id} - Quadrant: {quadrant:12s} ({quadrant_source}) | LumenVis: {lumen_vis:6.3f} | "
                #     #   f"Original: [L/R:{original_action[0]:6.3f}, U/D:{original_action[1]:6.3f}, F/B:{original_action[2]:6.3f}, "
                #     #   f"Pitch:{original_action[3]:6.3f}, Yaw:{original_action[4]:6.3f}, Roll:{original_action[5]:6.3f}] | "
                #       f"Clipped: [L/R:{clipped_action[0]:6.3f}, U/D:{clipped_action[1]:6.3f}, F/B:{clipped_action[2]:6.3f}, "
                #       f"Pitch:{clipped_action[3]:6.3f}, Yaw:{clipped_action[4]:6.3f}, Roll:{clipped_action[5]:6.3f}]")

        return constrained_actions

    def _apply_action(self) -> None:
        #pdb.set_trace()
        # Compute center alignment for conditional movement control
        if not hasattr(self, 'center_alignment'):
            self.center_alignment = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

        # Update center alignment from depth images if available
        if hasattr(self.robot, 'get_depth'):
            try:
                center_alignments, _, _ = self._compute_movement_metrics()
                self.center_alignment = center_alignments
            except Exception as e:
                logging.warning(f"Failed to compute center alignment: {e}")

        # Apply conditional logic to actions BEFORE passing to robot
        # Store original PPO actions for debugging
        original_ppo_actions = self.actions.clone()

        # Apply movement constraints
        actions_to_apply = self._apply_movement_constraints(self.actions.clone())

        # CRITICAL: Update self.actions to the clipped version for reward computation
        # This ensures PPO learns about the actions that were actually executed
        self.actions = actions_to_apply

        # Pass to robot (actions are already constrained)
        self.robot.apply_action(actions_to_apply)

    def _check_colon_validity(self) -> None:
        """
        Check if the colon is still valid (not all nodes blown away).
        Mark environments with invalid colons for truncation.
        """
        if self.colon.is_rigid:
            # For rigid colons, check if the body has moved too far from initial position
            current_pos = self.colon.colon_body.data.body_state_w[:, :3]
            # Use the initial colon position stored in init_pos
            initial_pos = torch.tensor(self.colon.init_pos, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
            displacement = torch.norm(current_pos - initial_pos, dim=1)
            # Threshold for invalid colon (10 meters - very large displacement)
            INVALID_THRESHOLD = 10.0
            self.colon_invalid = displacement > INVALID_THRESHOLD
        else:
            # For deformable colons, check nodal positions
            default_nodal_state = self.colon.colon_body._data.default_nodal_state_w
            current_nodal_state = self.colon.colon_body._data.nodal_state_w

            # Get positions (first 3 components)
            default_nodal_pos = default_nodal_state[:, :, :3]  # (num_envs, num_nodes, 3)
            current_nodal_pos = current_nodal_state[:, :, :3]  # (num_envs, num_nodes, 3)

            # Calculate displacement for each node in each environment
            displacement = torch.norm(current_nodal_pos - default_nodal_pos, dim=2)  # (num_envs, num_nodes)

            # Threshold for a node being "blown away" (in meters)
            BLOWN_AWAY_THRESHOLD = 0.5  # Same as in colon reset

            # Count how many nodes are blown away per environment
            blown_away_nodes = (displacement > BLOWN_AWAY_THRESHOLD).sum(dim=1)  # (num_envs,)
            total_nodes = displacement.shape[1]

            # If more than 50% of nodes are blown away, mark colon as invalid
            blown_away_ratio = blown_away_nodes.float() / total_nodes
            self.colon_invalid = blown_away_ratio > 0.5

            # Log when colons become invalid
            newly_invalid = self.colon_invalid & ~self.colon_invalid_prev
            if newly_invalid.any():
                invalid_env_ids = torch.where(newly_invalid)[0]
                logging.info(f"Colon became invalid in environments: {invalid_env_ids.cpu().tolist()}")
                logging.info(f"Blown away ratios: {blown_away_ratio[newly_invalid].cpu().tolist()}")

            # Update previous state
            self.colon_invalid_prev = self.colon_invalid.clone()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check colon validity before computing dones
        self._check_colon_validity()

        # Update camera data to get fresh depth for alignment checking
        # This is necessary because _get_dones() is called before _get_observations()
        if self.use_camera and hasattr(self.robot, 'egocamera'):
            try:
                self.robot.egocamera.update(dt=self.cfg.sim.dt * self.cfg.decimation)
            except Exception as e:
                logging.warning(f"Camera update failed in _get_dones: {e}")

        # Compute center alignment FRESH for done checking
        # This ensures we have the most recent alignment values
        if hasattr(self.robot, 'get_depth'):
            try:
                center_alignments, _, _ = self._compute_movement_metrics()
                self.center_alignment = center_alignments
            except Exception as e:
                logging.warning(f"Failed to compute center alignment in _get_dones: {e}")
                # Initialize if failed
                if not hasattr(self, 'center_alignment'):
                    self.center_alignment = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        truncated = time_out | self.truncate_now | self.colon_invalid
        terminated = self.goal_reached | self.hit_wall
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        current_states = self.get_states()

        if self.previous_states is None:
            self.previous_state = current_states

        # Print which actions are being used for reward computation
        if self.common_step_counter % 50 == 0:  # Print every 50 steps
            print(f"[REWARD] Step {self.common_step_counter} - Computing rewards with actions (Env 0): {self.actions[0].cpu().numpy()}")

        rewards = self.reward_function(self.actions, current_states, self.previous_states)

        # Update goal_reached status from reward function if available
        if hasattr(self.reward_function, 'goal_reached_per_env') and self.reward_function.goal_reached_per_env is not None:
            self.goal_reached = self.reward_function.goal_reached_per_env.clone()

        # IMPORTANT: Save goal_reached to extras BEFORE _reset_idx() gets called
        # This ensures we capture the success state for environments that just finished
        self.extras["goal_reached"] = self.goal_reached.clone()

        # Also save center alignment for logging
        if hasattr(self, 'center_alignment'):
            self.extras["center_alignment"] = self.center_alignment.clone()

        self.previous_states = current_states
        self.latest_rewards = rewards
        return rewards
        
    def _get_observations(self) -> dict:
        """Get observations from robot."""
        # Note: Camera is already updated in _get_dones() for alignment checking
        # So we skip the update here to avoid doing it twice
        # The camera data is still fresh from _get_dones()

        robot_obs = self.robot.get_observation(use_pose_in_obs=self.use_pose_in_obs, use_camera=self.use_camera)

        obs = {"policy": robot_obs}

        self._camera_data = robot_obs["rgb"].detach().cpu()
        return obs

    def get_states(self):
        if hasattr(self, "obs_buf"):
            obs = self.obs_buf["policy"]
        else:
            obs = None
        #pdb.set_trace()
        pose=self.robot.get_pose()
        #print("robot_pose",pose)

        robot_positions = self.robot.get_pose()[:, :3]
        depth_data = self.robot.get_depth()

        states = {
            "obs" : obs,
            "robot_positions" : robot_positions,
            "depth" : depth_data,
        }
        return states

    def reset(self, seed=42, env_ids: torch.Tensor = None, options = None):
        self.goal_reached = torch.tensor([False]*self.num_envs, device=self.device)
        self.hit_wall = torch.tensor([False]*self.num_envs, device=self.device)
        self.truncate_now = torch.tensor([False]*self.num_envs, device=self.device)
        self.colon_invalid = torch.tensor([False]*self.num_envs, device=self.device)
        self.colon_invalid_prev = torch.tensor([False]*self.num_envs, device=self.device)
        self.poor_alignment = torch.tensor([False]*self.num_envs, device=self.device)

        # Reset stuck detection tracking
        self.previous_positions = None
        self.poor_alignment_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.backward_steps_remaining = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Reset bright region tracking
        self.bright_region_directions = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.has_bright_regions = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        print("reset called")

        obs, extras = super().reset(seed=seed)
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Reset colon first
        self.colon.reset(env_ids)

        # CRITICAL: Step simulation to update colon state
        self.sim.step()

        # Check if CSV file is provided for initialization
        csv_init_file = self.config.get("env_config", {}).get("init_from_csv", None)
        csv_init_endpose = self.config.get("env_config", {}).get("init_endpose_from_csv", None)
        # Get entry positions AFTER stepping - load from CSV if provided
        self.entry_positions = self.colon.get_entry_pos(env_ids, csv_filepath=csv_init_file)
        self.targets = self.colon.get_targets(env_ids, csv_filepath=csv_init_endpose)

        self.reward_function.reset(
            initial_positions=self.entry_positions,
            goals=self.targets,
        )

        # Set robot positions based on colon entry points
        self.robot.set_pos(self.entry_positions)

        # CRITICAL: Write data to sim and forward kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()

        # Generate initial joint configurations

        if csv_init_file is not None:
            # Load from CSV file
            initial_joint_positions = csv_init_file
            logging.info(f"Initializing from CSV: {csv_init_file}")
        else:
            # Check if random initialization is enabled in config
            use_random_init = self.config.get("env_config", {}).get("random_initial_configuration", True)

            if use_random_init:
                # Generate random initial joint configurations for diversity
                # This creates different starting poses for the robot instead of always straight
                max_angle = self.config.get("env_config", {}).get("init_max_joint_angle", 0.3)
                smoothness = self.config.get("env_config", {}).get("init_joint_smoothness", 0.85)

                initial_joint_positions = self.robot.generate_random_joint_positions(
                    num_configs=len(env_ids),
                    max_angle=max_angle,    # Moderate bending (~17 degrees per joint default)
                    smoothness=smoothness    # Smooth, natural curves
                )
            else:
                # Use straight configuration (all zeros)
                initial_joint_positions = None

        # Reset robot with the configuration (which will now have valid data)
        self.robot.reset(env_ids, joint_positions=initial_joint_positions)
        # Final step to ensure everything is synchronized
        self.sim.step()

        if self.show_markers and self.markers is None:
            start_position = self.entry_positions[0]
            target_position = self.targets[0]

            self.markers = []
            for env_id, (start_position, target_positions) in enumerate(zip(self.entry_positions, self.targets)):
                self.markers.append(create_simple_sphere_markers(start_position, target_positions[1:], env_id))

        self.debug_resets_remaining -= 1
        if self.debug_resets_remaining == 0:
            if self.tracer is not None:
                self.tracer.save("intermediate_trace.json")
            exit()

        return obs, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        # Call parent reset FIRST. This resets the scene and all actors to default.
        super()._reset_idx(env_ids)

        # Now, apply our custom logic ON TOP of the default state.

        # Reset colon (this re-applies the nodal attachments)
        self.colon.reset(env_ids)

        # CRITICAL: Step to update colon state after attachments
        self.sim.step()

        # Check if we should use CSV initialization
        csv_init_file = self.config.get("env_config", {}).get("init_from_csv", None)
        csv_init_endpose = self.config.get("env_config", {}).get("init_endpose_from_csv", None)

        # Get entry positions AFTER stepping - load from CSV if provided
        self.entry_positions = self.colon.get_entry_pos(env_ids, csv_filepath=csv_init_file)
        self.targets = self.colon.get_targets(env_ids, csv_filepath=csv_init_endpose)

        self.reward_function.reset(
            initial_positions=self.entry_positions,
            goals=self.targets,
        )

        # Set robot position based on colon entry point
        self.robot.set_pos(self.entry_positions)

        # Write and forward to update sim buffers
        self.scene.write_data_to_sim()
        self.sim.forward()

        # Reset robot (which now starts at the correct position)

        if csv_init_file is not None:
            # Use CSV file for reset
            self.robot.reset(env_ids, joint_positions=csv_init_file)
        else:
            # Use default behavior (random or straight)
            use_random_init = self.config.get("env_config", {}).get("random_initial_configuration", True)
            if use_random_init:
                self.robot.reset(env_ids, joint_positions='random')
            else:
                self.robot.reset(env_ids)

        # Reset our internal environment flags
        self.goal_reached[env_ids] = False
        self.hit_wall[env_ids] = False
        self.truncate_now[env_ids] = False
        self.colon_invalid[env_ids] = False
        self.colon_invalid_prev[env_ids] = False
        self.poor_alignment[env_ids] = False

        # Reset stuck detection for these environments
        self.poor_alignment_counter[env_ids] = 0
        self.backward_steps_remaining[env_ids] = 0

        # Reset bright region tracking for these environments
        self.bright_region_directions[env_ids] = 0.0
        self.has_bright_regions[env_ids] = False

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)  # See contents of super().step(action) below
        """
        def step(self, action: torch.Tensor) -> VecEnvStepReturn:
            '''Execute one time-step of the environment's dynamics.

            The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
            lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
            independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
            and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
            time-step is computed as the product of the two.

            This function performs the following steps:

            1. Pre-process the actions before stepping through the physics.
            2. Apply the actions to the simulator and step thobsrough the physics in a decimated manner.
            3. Compute the reward and done signals.
            4. Reset environments that have terminated or reached the maximum episode length.
            5. Apply interval events if they are enabled.
            6. Compute observations.

            Args:
                action: The actions to apply on the environment. Shape is (num_envs, action_dim).

            Returns:
                A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
            '''
            action = action.to(self.device)
            # add action noise
            if self.cfg.action_noise_model:
                action = self._action_noise_model(action)

            # process actions
            self._pre_physics_step(action)

            # check if we need to do rendering within the physics loop
            # note: checked here once to avoid multiple checks within the loop
            is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

            # perform physics stepping
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                # set actions into buffers
                self._apply_action()
                # set actions into simulator
                self.scene.write_data_to_sim()
                # simulate
                self.sim.step(render=False) # Hans: This is the the main time consumer, it is probably the soft body simulation steps
                # render between steps only if the GUI or an RTX sensor needs it
                # note: we assume the render interval to be the shortest accepted rendering interval.
                #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
                if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                    self.sim.render()
                # update buffers at sim dt
                self.scene.update(dt=self.physics_dt)

            # post-step:
            # -- update env counters (used for curriculum generation)
            self.episode_length_buf += 1  # step in current episode (per env)
            self.common_step_counter += 1  # total step (common for all envs)

            self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
            self.reset_buf = self.reset_terminated | self.reset_time_outs
            self.reward_buf = self._get_rewards()

            # -- reset envs that terminated/timed-out and log the episode information
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self._reset_idx(reset_env_ids)
                # update articulation kinematics
                self.scene.write_data_to_sim()
                self.sim.forward()
                # if sensors are added to the scene, make sure we render to reflect changes in reset
                if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                    self.sim.render()

            # post-step: step interval event
            if self.cfg.events:
                if "interval" in self.event_manager.available_modes:
                    self.event_manager.apply(mode="interval", dt=self.step_dt)

            # update observations
            self.obs_buf = self._get_observations()

            # add observation noise
            # note: we apply no noise to the state space (since it is used for critic networks)
            if self.cfg.observation_noise_model:
                self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

            # return observations, rewards, resets and extras
            return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
        """
        # Print contact forces every 100 steps
        if self.common_step_counter % 100 == 0:
            try:
                self.colon.print_contact_forces()
            except Exception as e:
                # Silently catch errors if contact sensor is not available
                pass

        # Get colon stress for logging
        try:
            colon_stress = self.colon.get_accumulated_stress()
        except Exception:
            colon_stress = None

        # Note: goal_reached is already in self.extras (set in _get_rewards)
        # and gets passed through super().step() into info
        info_extras = {
            "l2_norm" : self.latest_l2_norm,
            "colon_stress" : colon_stress,
            }
        info.update(info_extras)

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            if self._camera_data is None:
                logging.info("Camera data is None this step")
                return np.zeros((84, 84, 3), dtype=np.uint8)

            # Handle torch tensor
            rgb_data = self._camera_data.cpu().numpy()
            
            # Handle 4D tensor (batch, channels, height, width)
            if rgb_data.ndim == 4:
                rgb_data = rgb_data[0]  # Take first batch
            
            # Convert from (C, H, W) to (H, W, C)
            if rgb_data.shape[0] == 3:
                rgb_data = np.transpose(rgb_data, (1, 2, 0))
            
            # Convert to uint8 [0, 255] range
            if rgb_data.dtype != np.uint8:
                if rgb_data.max() <= 1.0:
                    rgb_data = (rgb_data * 255).astype(np.uint8)
                else:
                    rgb_data = np.clip(rgb_data, 0, 255).astype(np.uint8)
            
            # Check if we need to add text overlays
            has_text = (self.latest_rewards is not None or 
                    self.latest_l2_norm is not None)
            
            # Convert to PIL Image for processing
            img = Image.fromarray(rgb_data)
            
            # Upsample if image is too small for proper text rendering AND we have text
            if has_text and (img.width < 400 or img.height < 400):
                scale_factor = max(400 / img.width, 400 / img.height)
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Add text overlays
            if has_text:
                draw = ImageDraw.Draw(img)
                
                # Try to use a reasonable font, fallback to default
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    try:
                        font = ImageFont.load_default()
                    except:
                        font = None
                
                y_offset = 10
                line_height = 20
                
                # Add reward text
                if self.latest_rewards is not None:
                    # Handle torch tensor conversion
                    if isinstance(self.latest_rewards, torch.Tensor):
                        rewards_val = self.latest_rewards.cpu().numpy()
                    else:
                        rewards_val = self.latest_rewards
                    
                    if isinstance(rewards_val, (list, np.ndarray)) and np.array(rewards_val).size > 1:
                        reward_text = f"Rewards: {np.array(rewards_val)}"
                    else:
                        # Handle scalar case
                        if isinstance(rewards_val, (list, np.ndarray)):
                            rewards_val = float(rewards_val.item() if hasattr(rewards_val, 'item') else rewards_val[0])
                        else:
                            rewards_val = float(rewards_val)
                        reward_text = f"Reward: {rewards_val:.4f}"
                    
                    # Add black outline for better visibility
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx != 0 or dy != 0:
                                draw.text((10 + dx, y_offset + dy), reward_text, 
                                        fill=(0, 0, 0), font=font)
                    # White text on top
                    draw.text((10, y_offset), reward_text, 
                            fill=(255, 255, 255), font=font)
                    y_offset += line_height
                
                # Add L2 norm text
                if self.latest_l2_norm is not None:
                    # Handle torch tensor conversion
                    if isinstance(self.latest_l2_norm, torch.Tensor):
                        l2_val = float(self.latest_l2_norm.cpu().item())
                    else:
                        l2_val = float(self.latest_l2_norm)
                    
                    l2_text = f"L2 Norm: {l2_val:.4f}"
                    
                    # Add black outline for better visibility
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx != 0 or dy != 0:
                                draw.text((10 + dx, y_offset + dy), l2_text, 
                                        fill=(0, 0, 0), font=font)
                    # White text on top
                    draw.text((10, y_offset), l2_text, 
                            fill=(255, 255, 255), font=font)
        return np.array(img)
        
            
            