import logging
import math

import torch
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObject
from isaaclab.sensors import CameraCfg, Camera, TiledCameraCfg, TiledCamera
from isaaclab.sim.spawners.lights import SphereLightCfg
from isaaclab.sensors import ContactSensorCfg, ContactSensor

from isaaclab.utils.math import quat_apply

from arcgym.assets.robots.base_robot import BaseRobot

class RobotEndoscopeCapsule(BaseRobot):
    def __init__(self, scene, config, isaac_cfg, init_pos, init_rot, device):
        self.config = config
        self.isaac_cfg =  isaac_cfg
        self.scene = scene
        
        self.use_camera = config["env_config"]["use_camera"]
        
        logging.info(f"Robot camera usage: {self.use_camera}")
        logging.info(f"Setting capsule init position to: {init_pos}")

        self.capsule = RigidObject(self.isaac_cfg.capsule_cfg.replace(init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_rot)))
        
        logging.info("Creating camera...")
        self.egocamera = TiledCamera(self.isaac_cfg.camera_cfg)
        logging.info("✓ Camera created")
        
        logging.info("Creating light...")
        self.egolight = sim_utils.spawners.lights.spawn_light(
            prim_path="/World/envs/env_.*/Robot/front_light",
            cfg=isaac_cfg.light_cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(1, 0, 0, 0)
        )
        logging.info("✓ Light created")
        
        self.contactsensor = ContactSensor(cfg=self.isaac_cfg.contact_cfg)

        scene.rigid_objects['robot'] = self.capsule
        scene.sensors['egocamera'] = self.egocamera
        scene.sensors['contactsensor'] = self.contactsensor
        
        logging.info("✓ All components registered with scene")

        self.init_pos = init_pos
        self.init_rot = init_rot
        self.env_init_pos = None
        self.device = device

    @staticmethod
    def make_isaac_config(config : dict):
        env_config = config["env_config"]
        robot_config = config["robot_config"]
        simulation_config = config["simulation_config"]
        discrete_action_space = env_config["discrete_action_space"]
        camera_height, camera_width = robot_config["camera_resolution"] # TODO: Figure out if it is (h, w) or (w, h)
        camera_update_period = simulation_config["render_interval"] * simulation_config["dt"]

        ROBOT_CFG = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.CapsuleCfg(
                radius=robot_config["capsule_radius"],
                height=robot_config["capsule_height"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False, 
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=robot_config["collision_contact_offset"],
                    rest_offset=robot_config["collision_rest_offset"],
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.5, 0.0),
                    emissive_color=(1.0, 0.3, 0.0),
                    metallic=0.0,
                    roughness=0.5,
                    opacity=1.0
                ),
                axis='Z',
                activate_contact_sensors=True,             
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0))
        )

        CAMERA_CFG = TiledCameraCfg(
                prim_path="/World/envs/env_.*/Robot/front_cam",
                update_period=camera_update_period,
                history_length=0,
                height=camera_height,
                width=camera_width,
                data_types=["rgb", "depth"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=robot_config["front_camera_focal_length"], 
                    focus_distance=robot_config["front_camera_focus_distance"],  
                    horizontal_aperture=robot_config["front_camera_horizontal_aperture"],  
                    clipping_range=robot_config["front_camera_clipping_range"], 
                ),
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(0.0, 0.0, robot_config["capsule_height"]), # Position at top of capsule
                    rot=(1, 0, 0, 0), # Identity rotation - camera points in +Z direction
                ),
        )

        LIGHT_CFG = SphereLightCfg(
                color=robot_config["front_light_color"], 
                color_temperature=robot_config["front_light_color_temperature"], 
                intensity=robot_config["front_light_intensity"],
                radius=robot_config["front_light_radius"],
                exposure=robot_config["front_light_exposure"],
                visible=True,
                )

        CONTACT_SENSOR_CFG = ContactSensorCfg(
            track_pose = True,
            prim_path="/World/envs/env_.*/Robot",
            update_period=0.0,
            history_length=5,
            debug_vis=False,
        )

        class RobotEndoscopeCapsuleCfg:
            capsule_cfg = ROBOT_CFG
            camera_cfg = CAMERA_CFG
            light_cfg = LIGHT_CFG
            contact_cfg = CONTACT_SENSOR_CFG
            action_space = Discrete(6) if discrete_action_space else Box(low=-1.0, high=1.0, shape=(6, ))
            observation_space = Dict({
                "rgb": Box(0, 255, shape=(3, camera_height, camera_width), dtype=np.uint8)
                })
        
        return RobotEndoscopeCapsuleCfg()

    def apply_action(self, actions: torch.Tensor, action_scale: float = 1.0, **kwargs) -> None:
        """Apply 6DOF velocity actions with collision-safe scaling.

        Args:
            actions: Action tensor
            action_scale: Scaling factor for actions
            **kwargs: Additional parameters for compatibility
        """
        pose = self.get_pose()  # Shape: (num_envs, 13)
        if pose is None or pose.numel() == 0:
            logging.warning("Warning: Invalid pose data, skipping action application")
            return

        robot_orient = pose[:, 3:7]  # (num_envs, 4) - extract quaternion

        actions_scaled = actions * action_scale # (num_evns, 6) * (1,) -> (num_evns, 6)

        actions_scaled[:, :3] *= 0.05
        actions_scaled[:, 3:] *= 1.0

        twist_local_lin = quat_apply(robot_orient, actions_scaled[:, :3])
        twist_local_ang = quat_apply(robot_orient, actions_scaled[:, 3:])

        twist = torch.cat((twist_local_lin, twist_local_ang), dim=-1)

        self.capsule.write_root_velocity_to_sim(twist)

    def get_observation(self, use_pose_in_obs, use_camera) -> torch.Tensor:
        """
        Get Capsule Robot Observations
        """
        obs = {}

        # Add pose data
        if use_pose_in_obs:
            obs["pose"] = self.get_pose()
        
        # Add camera data
        if use_camera:
            # Check if camera data is available
            if (not hasattr(self.egocamera, 'data') or self.egocamera.data is None or not hasattr(self.egocamera.data, 'output') or "rgb" not in self.egocamera.data.output):
                raise ValueError("camera data not available")
            
            rgb_data = self.egocamera.data.output["rgb"]
            
            if rgb_data is None:
                raise RuntimeError("RGB data is None")

            obs["rgb"] = rgb_data.permute(0, 3, 1, 2) # channels last to channels first, e.g. (n_env, w, h, c) -> (n_env, c, w, h)

        if len(obs) == 0:
            raise ValueError("Capsule observations are empty, did you forget to specify the observations space correctly?")

        return obs

    def get_pose(self):
        """Get robot pose. Returns shape (num_envs, 13) for consistency with articulated robots.

        The capsule has only one body, so we squeeze out the body dimension.
        Returns: tensor of shape (num_envs, 13) with [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        """
        body_state = self.capsule.data.body_state_w  # Shape: (num_envs, num_bodies, 13)
        # For capsule robot, there's only 1 body, so squeeze out the body dimension
        return body_state[:, 0, :]  # Shape: (num_envs, 13) 

    def get_depth(self):
        if (not hasattr(self.egocamera, 'data') or self.egocamera.data is None or not hasattr(self.egocamera.data, 'output') or "depth" not in self.egocamera.data.output):
            raise ValueError("camera data not available")

        return self.egocamera.data.output["depth"]

    def set_pos(self, init_pos, init_rot=None):
        """Set positions for specific environments."""
        if isinstance(init_pos, torch.Tensor):
            # Handle per-environment positions
            self.env_init_pos = init_pos.clone()
        else:
            # Handle single position (convert to default)
            self.init_pos = torch.tensor(init_pos)
            self.env_init_pos = self.init_pos.clone()

        if init_rot is not None:
            raise NotImplementedError("Initializing rotation is not implemented yet")

    def generate_random_joint_positions(self, num_configs: int = None, max_angle: float = 0.5, smoothness: float = 0.8) -> None:
        """Capsule robot has no joints, so return None.

        This method exists for compatibility with the articulated robot interface,
        but returns None since the capsule is a rigid body with no joints.

        Args:
            num_configs: Number of configurations (ignored)
            max_angle: Maximum joint angle (ignored)
            smoothness: Smoothness factor (ignored)

        Returns:
            None
        """
        return None

    def reset(self, env_ids: torch.Tensor = None, joint_positions: torch.Tensor = None) -> None:
        """Reset robot to initial pose and zero velocities.

        Args:
            env_ids: Environment indices to reset
            joint_positions: Ignored for capsule robot (no joints)
        """
        # Get default root state
        root_state = self.capsule.data.default_root_state[env_ids].clone()

        # Set positions based on whether we have per-env positions
        if self.env_init_pos is not None:
            # Use per-environment positions (already in world coordinates!)
            if len(env_ids) == self.env_init_pos.shape[0]:
                root_state[:, :3] = self.env_init_pos  # Direct assignment, no adding origins
            else:
                # Handle subset of environments
                root_state[:, :3] = self.env_init_pos
        else:
            # Use default position for all environments (need to add origins)
            origins = self.scene.env_origins[env_ids]
            root_state[:, :3] = origins + torch.tensor(self.init_pos, device=self.device)

        base_rot = torch.tensor(self.init_rot, device=self.device).repeat(len(env_ids), 1)

        root_state[:, 3:7] = base_rot

        # Zero out velocities
        root_state[:, 7:] = 0.0

        # Write to simulation
        self.capsule.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.capsule.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        # Reset internal buffers
        self.capsule.reset(env_ids)
