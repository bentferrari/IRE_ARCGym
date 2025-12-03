from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
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

        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=debug_vis,
        )

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

        self.sim.set_camera_view(eye=[0.5, 1.5, 0.5], target=[0.0, 0.0, 0.5])

    def _setup_scene(self):
        self.stage = stage_utils.get_current_stage()

        self.robot = self.robot_factory.build_robot(
            scene=self.scene,
            init_pos=(0.22, 0.16, 0.4),
            #init_rot=(0, 1, 0, 1),
            init_rot=(1, 0, 0, 0),
            )

        self.colon = ColonModel(self.scene, cfg=self.cfg.colon_cfg, init_pos=(0.5,0.5,0.1), init_rot=(0.5, 0.5, 0.5, 0.5), is_rigid=False)

        # Note: Keeping this here to show how to add a physics callback.
        # Register the callback
        #camera_controller = CameraController(self.robot.egocamera, self.cfg.decimation)

        #if self.sim.physics_callback_exists("camera_control"):
        #    self.sim.remove_physics_callback("camera_control")
        #self.sim.add_physics_callback("camera_control", camera_controller.physics_callback)
        
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.scene.cfg.filter_collisions:
            self.scene.filter_collisions(self.scene._global_prim_paths)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()
    
    def _apply_action(self) -> None:
        #pdb.set_trace()
        self.robot.apply_action(self.actions)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        truncated = time_out | self.truncate_now
        terminated = self.goal_reached | self.hit_wall
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        current_states = self.get_states()

        if self.previous_states is None:
            self.previous_state = current_states

        rewards = self.reward_function(self.actions, current_states, self.previous_states)

        self.previous_states = current_states
        self.latest_rewards = rewards
        return rewards
        
    def _get_observations(self) -> dict:
        """Get observations from robot."""
        # Update camera data before getting observations
        if self.use_camera and hasattr(self.robot, 'egocamera'):
            try:
                self.robot.egocamera.update(dt=self.cfg.sim.dt * self.cfg.decimation)
            except Exception as e:
                logging.exception(f"Camera update failed: {e}")
                raise
        
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
        print("robot_pose",pose)
        
        #robot_positions = self.robot.get_pose()[:, 0, :3]
        robot_positions = self.robot.get_pose()[:, :3]
        depth_data = self.robot.get_depth()

        states = {
            "obs" : obs,
            "robot_positions" : robot_positions,
            "depth" : depth_data,
        }
        return states

    # def reset(self, seed=None, env_ids: torch.Tensor = None, options = None):
    #     self.goal_reached = torch.tensor([False]*self.num_envs, device=self.device)
    #     self.hit_wall = torch.tensor([False]*self.num_envs, device=self.device)
    #     self.truncate_now = torch.tensor([False]*self.num_envs, device=self.device)

    #     print("reset called")

    #     obs, extras = super().reset(seed=seed)
        
    #     if env_ids is None: # I am not sure about this, did I write this? Chat GPT is that you?
    #         env_ids = torch.arange(self.num_envs, device=self.device)
        
    #     # Reset colon first
    #     self.colon.reset(env_ids)
    #     self.sim.step()
        
    #     # Get entry positions for the environments being reset
    #     self.entry_positions = self.colon.get_entry_pos(env_ids)
    #     self.targets = self.colon.get_targets(env_ids)

    #     self.reward_function.reset(
    #         initial_positions=self.entry_positions,
    #         goals=self.targets,
    #         )
        
    #     #if self.robot:
    #     # Set robot positions based on colon entry points
    #     self.robot.set_pos(self.entry_positions)  # Remove env_ids parameter
    #     # Then reset robot
    #     self.robot.reset(env_ids)

    #     if self.show_markers and self.markers is None:
    #         start_position = self.entry_positions[0]
    #         target_position = self.targets[0]

    #         self.markers = []
    #         for env_id, (start_position, target_positions) in enumerate(zip(self.entry_positions, self.targets)):
    #             self.markers.append(create_simple_sphere_markers(start_position, target_positions[1:], env_id))

    #     self.debug_resets_remaining -= 1
    #     if self.debug_resets_remaining == 0:
    #         # BENCHMARKING START
    #         if self.tracer is not None:
    #             self.tracer.save("intermediate_trace.json")
    #         # BENCHMARKING END
    #         exit()

    #     return obs, extras

    # def _reset_idx(self, env_ids: Sequence[int]):
    #     self.colon.reset(env_ids)
    #     self.sim.step()
    #     self.robot.reset(env_ids)
        
    #     # Get entry positions for the environments being reset
    #     self.entry_positions = self.colon.get_entry_pos(env_ids)
    #     self.targets = self.colon.get_targets(env_ids)

    #     self.reward_function.reset(
    #         initial_positions=self.entry_positions,
    #         goals=self.targets,
    #         )

    #     self.goal_reached[env_ids] = False
    #     self.hit_wall[env_ids] = False
    #     self.truncate_now[env_ids] = False

    #     super()._reset_idx(env_ids)  # See contents of super()._reset_idx(env_ids) below
    #     """
    #     def _reset_idx(self, env_ids: Sequence[int]):
    #         '''Reset environments based on specified indices.

    #         Args:
    #             env_ids: List of environment ids which must be reset
    #         '''
    #         self.scene.reset(env_ids)

    #         # apply events such as randomization for environments that need a reset
    #         if self.cfg.events:
    #             if "reset" in self.event_manager.available_modes:
    #                 env_step_count = self._sim_step_counter // self.cfg.decimation
    #                 self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

    #         # reset noise models
    #         if self.cfg.action_noise_model:
    #             self._action_noise_model.reset(env_ids)
    #         if self.cfg.observation_noise_model:
    #             self._observation_noise_model.reset(env_ids)

    #         # reset the episode length buffer
    #         self.episode_length_buf[env_ids] = 0
    #     """

    def reset(self, seed=None, env_ids: torch.Tensor = None, options = None):
        self.goal_reached = torch.tensor([False]*self.num_envs, device=self.device)
        self.hit_wall = torch.tensor([False]*self.num_envs, device=self.device)
        self.truncate_now = torch.tensor([False]*self.num_envs, device=self.device)

        print("reset called")

        obs, extras = super().reset(seed=seed)
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Reset colon first
        self.colon.reset(env_ids)
        
        # CRITICAL: Step simulation to update colon state
        self.sim.step()
        
        # Get entry positions AFTER stepping
        self.entry_positions = self.colon.get_entry_pos(env_ids)
        self.targets = self.colon.get_targets(env_ids)

        self.reward_function.reset(
            initial_positions=self.entry_positions,
            goals=self.targets,
        )
        
        # Set robot positions based on colon entry points
        self.robot.set_pos(self.entry_positions)
        
        # CRITICAL: Write data to sim and forward kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()
        
        # Then reset robot (which will now have valid data)
        self.robot.reset(env_ids)
        
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

    # def _reset_idx(self, env_ids: Sequence[int]):
    #     # Reset colon
    #     self.colon.reset(env_ids)
        
    #     # CRITICAL: Step to update colon state
    #     self.sim.step()
        
    #     # Get entry positions AFTER stepping
    #     self.entry_positions = self.colon.get_entry_pos(env_ids)
    #     self.targets = self.colon.get_targets(env_ids)

    #     self.reward_function.reset(
    #         initial_positions=self.entry_positions,
    #         goals=self.targets,
    #     )
        
    #     # Set robot position
    #     self.robot.set_pos(self.entry_positions)
        
    #     # Write and forward
    #     self.scene.write_data_to_sim()
    #     self.sim.forward()
        
    #     # Reset robot
    #     self.robot.reset(env_ids)

    #     self.goal_reached[env_ids] = False
    #     self.hit_wall[env_ids] = False
    #     self.truncate_now[env_ids] = False

    #     # Call parent reset
    #     super()._reset_idx(env_ids)

    def _reset_idx(self, env_ids: Sequence[int]):
        # Call parent reset FIRST. This resets the scene and all actors to default.
        super()._reset_idx(env_ids)

        # Now, apply our custom logic ON TOP of the default state.

        # Reset colon (this re-applies the nodal attachments)
        self.colon.reset(env_ids)
        
        # CRITICAL: Step to update colon state after attachments
        self.sim.step()
        
        # Get entry positions AFTER stepping
        self.entry_positions = self.colon.get_entry_pos(env_ids)
        self.targets = self.colon.get_targets(env_ids)

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
        self.robot.reset(env_ids)

        # Reset our internal environment flags
        self.goal_reached[env_ids] = False
        self.hit_wall[env_ids] = False
        self.truncate_now[env_ids] = False

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
        info_extras = {
            "goal_reached" : self.goal_reached,
            "l2_norm" : self.latest_l2_norm,
            }
        info.update(info_extras)

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            if self._camera_data is None:
                logging.info("Camera data is None this step")
                return np.zeros((128, 128, 3), dtype=np.uint8)

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
        
            
            