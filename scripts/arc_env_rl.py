import argparse
from datetime import datetime
import logging
import os
import pdb

from sane_rich_logging import setup_logging

setup_logging()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Create an application to launch ARC environment")
parser.add_argument("--train", action="store_true", help="Run in training mode")
parser.add_argument("--benchmark", action="store_true", help="benchmark a couple of steps using viztracer")
parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel environments")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli = parser.parse_args()

# Enable cameras (needed for camera sensors)
args_cli.enable_cameras = True

# Launch omniverse app
app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app

import gymnasium as gym

from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import VecNormalize

from arcgym.utils.callbacks import PerEnvRewardCallback, TrajectoryDataSaver

from isaaclab.envs import (
    DirectRLEnvCfg,
)

from arcgym.envs.sb3_wrapper import Sb3VecEnvWrapper

from arcgym.envs.arc_isaac_env import make_isaac_env_cfg
from arcgym.envs.framestack import FrameStack
from arcgym.assets.robots.robot_factory import RobotFactory

from gymnasium.envs.registration import register

if args_cli.benchmark:
    from viztracer import VizTracer
    import atexit

    tracer = VizTracer()
    tracer.start()

    # Register cleanup function
    def save_trace():
        tracer.stop()
        tracer.save("trace_results.json")

    atexit.register(save_trace)
else:
    tracer = None

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./logs/ppo_arc_{timestamp}/"
tensorboard_log = f"./tensorboard_logs/ppo_arc_{timestamp}/"
model_save_path = f"./models/ppo_arc_{timestamp}/"
video_save_path = os.path.join(tensorboard_log, "videos", "train")
trajectory_save_path = f"./trajectory_data/ppo_arc_{timestamp}/"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(tensorboard_log, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(trajectory_save_path, exist_ok=True)

register(
    id="ArcIsaacEnv-v0",
    entry_point="arcgym.envs.arc_isaac_env:ARCIsaacEnv",
    disable_env_checker=True,
    #max_episode_steps = 2500, # This breaks IsaacLab do not set this.
)

use_pose_in_obs = False
use_camera = True

device = args_cli.device

learning_config = {
    "total_timesteps": 1000000,
    "learning_rate" : 1e-4,
    "batch_size" : 1024,
    "verbose" : True,
}
robot_config = {
    "robot_type" : "magnetic_endoscope", # "capsule" or "soft_endoscope" or "magnetic_endoscope"
    # Capsule config 
    "capsule_radius" : 0.004,
    "capsule_height" : 0.012,
    "collision_contact_offset" : 0.0001,
    "collision_rest_offset" : 0.0,
    # Soft endoscope specific parameters (from original soft_endoscope.py)
    "num_passive_links" : 5,
    "num_active_links" : 5,
    "num_links_total" : 40,
    "link_radius" : 0.01,
    "link_height" : 0.1,
    "passive_stiffness" : 1e1,
    "passive_damping" : 1e4,
    "active_stiffness" : 1e8,
    "active_damping" : 1e3,
    "max_linear_velocity": 1,
    "max_angular_velocity": 1,
    #"link_density" : 0.1,
    # Camera and light configuration
    "camera_resolution" : (128, 128),
    "front_camera_focal_length" : 5.0, 
    "front_camera_focus_distance" : 10.0, 
    "front_camera_horizontal_aperture" : 20, 
    "front_camera_clipping_range" : (0.001, 0.2),
    "front_light_color" : (0.7, 0.7, 0.7), 
    "front_light_color_temperature" : 2000, 
    "front_light_intensity" : 3000,
    "front_light_radius" : 0.005,
    "front_light_exposure" : 5,
    "num_segments": 20,
    "joint_stiffness": 1e4,  # Adjust for desired compliance
    "joint_damping": 1e3,
    "joint_friction": 0.1
}
if robot_config["robot_type"] == "capsule":
    env_spacing = 0.5
else:
    env_spacing = 5

env_config = {
    "discrete_action_space" : False, # currently unsupported TODO: Figure out if this is something we want to be determinable from the outside, or if it is a property of the robot implementation.
    "use_pose" : False,
    "use_camera" : True,
    "render_mode" : "rgb_array",
    "env_spacing" : env_spacing,
    "num_envs" : args_cli.num_envs,
    "replicate_physics" : False,
    "action_scale" : 1,
    "debug_vis" : False,
    "episode_length_s" : 1000.0,   # short episode for testing
    "constraint_point_A": 1,  # Distance from robot tip to constraint point A along the robot's local z-axis
    "init_from_csv": "./saved_states/robot_state_20251215_134107.csv" if args_cli.train else None,
    "random_initial_configuration": False  # Use straight configuration (especially for teleoperation mode)
}

reward_config = {
    "reward_type" : "depth_goal", #"test_reward_action",#"depth_goal",#"default",#,
    "reward_scale" : 1.0,
    "eps" : 0.025,
    "running_penalty" : -0.1,
    "goal_reward" : 50.0,
}

simulation_config = {
    "device" : device,
    # The device to run the simulation on. Default is ``"cuda:0"``.
    # Valid options are:
    # - ``"cpu"``: Use CPU.
    # - ``"cuda"``: Use GPU, where the device ID is inferred from :class:`~isaaclab.app.AppLauncher`'s config.
    # - ``"cuda:N"``: Use GPU, where N is the device ID. For example, "cuda:0".
    "dt" : 1.0 / 240.0,
    # The physics simulation time-step (in seconds). Default is 0.0167 seconds.
    "render_interval": 4,
    # The number of physics simulation steps per rendering step. Default is 1.
    "gravity" : (0.0, 0.0, 0),
    # The gravity vector (in m/s^2). Default is (0.0, 0.0, -9.81).
    # If set to (0.0, 0.0, 0.0), gravity is disabled.
    "enable_scene_query_support" : False,
    # Enable/disable scene query support for collision shapes. Default is False.
    # This flag allows performing collision queries (raycasts, sweeps, and overlaps) on actors and
    # attached shapes in the scene. This is useful for implementing custom collision detection logic
    # outside of the physics engine.
    # If set to False, the physics engine does not create the scene query manager and the scene query
    # functionality will not be available. However, this provides some performance speed-up.
    # Note:
    #     This flag is overridden to True inside the :class:`SimulationContext` class when running the simulation
    #     with the GUI enabled. This is to allow certain GUI features to work properly.
    "use_fabric" : True,
    # Enable/disable reading of physics buffers directly. Default is True.
    # When running the simulation, updates in the states in the scene is normally synchronized with USD.
    # This leads to an overhead in reading the data and does not scale well with massive parallelization.
    # This flag allows disabling the synchronization and reading the data directly from the physics buffers.
    # It is recommended to set this flag to :obj:`True` when running the simulation with a large number
    # of primitives in the scene.
    # Note:
    #     When enabled, the GUI will not update the physics parameters in real-time. To enable real-time
    #     updates, please set this flag to :obj:`False`.
    #     When using GPU simulation, it is required to enable Fabric to visualize updates in the renderer.
    #     Transform updates are propagated to the renderer through Fabric. If Fabric is disabled with GPU simulation,
    #     the renderer will not be able to render any updates in the simulation, although simulation will still be
    #     running under the hood.
}

render_config = {
    "enable_translucency" : False,
	# Bool. Enables translucency for specular transmissive surfaces such as glass at the cost of some performance.
    "enable_reflections" : False,
    # Bool. Enables reflections at the cost of some performance.
    "enable_global_illumination" : True,
    # Bool. Enables Diffused Global Illumination at the cost of some performance.
    "antialiasing_mode" : "FXAA",
	# Literal[“Off”, “FXAA”, “DLSS”, “TAA”, “DLAA”].
    # DLSS: Boosts performance by using AI to output higher resolution frames from a lower resolution input. DLSS samples multiple lower resolution images and uses motion data and feedback from prior frames to reconstruct native quality images. DLAA: Provides higher image quality with an AI-based anti-aliasing technique. DLAA uses the same Super Resolution technology developed for DLSS, reconstructing a native resolution image to maximize image quality.
    "enable_dlssg" : False,
    # Bool. Enables the use of DLSS-G. DLSS Frame Generation boosts performance by using AI to generate more frames. This feature requires an Ada Lovelace architecture GPU and can hurt performance due to additional thread-related activities.
    "enable_dl_denoiser" : False,
	# Bool. Enables the use of a DL denoiser, which improves the quality of renders at the cost of performance.
    "dlss_mode" : 1,
	# Literal[0, 1, 2, 3]. For DLSS anti-aliasing, selects the performance/ quality tradeoff mode. Valid values are 0 (Performance), 1 (Balanced), 2 (Quality), or 3 (Auto).
    "enable_direct_lighting" : True,
	# Bool. Enable direct light contributions from lights.
    "samples_per_pixel" : 1,
	# Int. Defines the Direct Lighting samples per pixel. Higher values increase the direct lighting quality at the cost of performance.
    "enable_shadows" : True,
	# Bool. Enables shadows at the cost of performance. When disabled, lights will not cast shadows.
    "enable_ambient_occlusion" : False,
	# Bool. Enables ambient occlusion at the cost of some performance.
}

physx_config = {
    "solver_type" : 1,
    # The type of solver to use.Default is 1 (TGS).
    # Available solvers:
    # * :obj:`0`: PGS (Projective Gauss-Seidel)
    # * :obj:`1`: TGS (Temporal Gauss-Seidel)
    "min_position_iteration_count" : 1,
    # Minimum number of solver position iterations (rigid bodies, cloth, particles etc.). Default is 1.
    # .. note::
    #
    #     Each physics actor in Omniverse specifies its own solver iteration count. The solver takes
    #     the number of iterations specified by the actor with the highest iteration and clamps it to
    #     the range ``[min_position_iteration_count, max_position_iteration_count]``.
    "max_position_iteration_count" : 32,
    # Maximum number of solver position iterations (rigid bodies, cloth, particles etc.). Default is 255.
    # .. note::
    #     Each physics actor in Omniverse specifies its own solver iteration count. The solver takes
    #     the number of iterations specified by the actor with the highest iteration and clamps it to
    #     the range ``[min_position_iteration_count, max_position_iteration_count]``.
    "min_velocity_iteration_count" : 0,
    # Minimum number of solver velocity iterations (rigid bodies, cloth, particles etc.). Default is 0.
    # .. note::
    #     Each physics actor in Omniverse specifies its own solver iteration count. The solver takes
    #     the number of iterations specified by the actor with the highest iteration and clamps it to
    #     the range ``[min_velocity_iteration_count, max_velocity_iteration_count]``.
    "max_velocity_iteration_count" : 32,
    # Maximum number of solver velocity iterations (rigid bodies, cloth, particles etc.). Default is 255.
    # .. note::
    #     Each physics actor in Omniverse specifies its own solver iteration count. The solver takes
    #     the number of iterations specified by the actor with the highest iteration and clamps it to
    #     the range ``[min_velocity_iteration_count, max_velocity_iteration_count]``.
    "enable_ccd" : False,
    # Enable a second broad-phase pass that makes it possible to prevent objects from tunneling through each other.
    # Default is False.
    "enable_stabilization" : False,
    # Enable/disable additional stabilization pass in solver. Default is False.
    # .. note::
    #     We recommend setting this flag to true only when the simulation step size is large (i.e., less than 30 Hz or more than 0.0333 seconds).
    # .. warning::
    #     Enabling this flag may lead to incorrect contact forces report from the contact sensor.
    "enable_enhanced_determinism" : False,
    # Enable/disable improved determinism at the expense of performance. Defaults to False.
    # For more information on PhysX determinism, please check `here`_.
    # .. _here: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html#enhanced-determinism
    "bounce_threshold_velocity" : 0.5,
    # Relative velocity threshold for contacts to bounce (in m/s). Default is 0.5 m/s.
    "friction_offset_threshold" : 0.04,
    # Threshold for contact point to experience friction force (in m). Default is 0.04 m.
    "friction_correlation_distance" : 0.025,
    # Distance threshold for merging contacts into a single friction anchor point (in m). Default is 0.025 m.
    "gpu_max_rigid_contact_count" : 2**23,
    # Size of rigid contact stream buffer allocated in pinned host memory. Default is 2 ** 23.
    "gpu_max_rigid_patch_count" : 5 * 2**15,
    # Size of the rigid contact patch stream buffer allocated in pinned host memory. Default is 5 * 2 ** 15.
    "gpu_found_lost_pairs_capacity" : 2**21,
    # Capacity of found and lost buffers allocated in GPU global memory. Default is 2 ** 21.
    # This is used for the found/lost pair reports in the BP.
    "gpu_found_lost_aggregate_pairs_capacity" : 2**25,
    # Capacity of found and lost buffers in aggregate system allocated in GPU global memory.
    # Default is 2 ** 25.
    # This is used for the found/lost pair reports in AABB manager.
    "gpu_total_aggregate_pairs_capacity" : 2**21,
    # Capacity of total number of aggregate pairs allocated in GPU global memory. Default is 2 ** 21.
    "gpu_collision_stack_size" : 2**26,
    # Size of the collision stack buffer allocated in pinned host memory. Default is 2 ** 26.
    "gpu_heap_capacity" : 2**26,
    # Initial capacity of the GPU and pinned host memory heaps. Additional memory will be allocated
    # if more memory is required. Default is 2 ** 26.
    "gpu_temp_buffer_capacity" : 2**24,
    # Capacity of temp buffer allocated in pinned host memory. Default is 2 ** 24.
    "gpu_max_num_partitions" : 8,
    # Limitation for the partitions in the GPU dynamics pipeline. Default is 8.
    # This variable must be power of 2. A value greater than 32 is currently not supported. Range: (1, 32)
    "gpu_max_soft_body_contacts" : 2**20,
    # Size of soft body contacts stream buffer allocated in pinned host memory. Default is 2 ** 20.
    "gpu_max_particle_contacts" : 2**20,
    # Size of particle contacts stream buffer allocated in pinned host memory. Default is 2 ** 20.
}

debug_config = {
    "show_markers" : False,
}

config = {
    "learning_config" : learning_config,
    "env_config" : env_config,
    "reward_config" : reward_config,
    "robot_config" : robot_config,
    "simulation_config" : simulation_config,
    "render_config" : render_config,
    "physx_config" : physx_config,
    "debug_config" : debug_config,
}
#pdb.set_trace()
robot_factory = RobotFactory(config, device=device)
#pdb.set_trace()
env_cfg = make_isaac_env_cfg(config, robot_factory)

env = gym.make("ArcIsaacEnv-v0", cfg=env_cfg, robot_factory=robot_factory,
               config=config, tracer=tracer, disable_env_checker=True) # We need disable_env_checker=True because it fails due to the wrapper removing the 'policy' key
               #, render_mode = "rgb_array"

env = FrameStack(env, n_stack=1)

video_kwargs = {
    "video_folder": video_save_path,
    "step_trigger": lambda step: step % 25000 == 0,
    "video_length": 2500,
}
env = gym.wrappers.RecordVideo(env, **video_kwargs)

env = Sb3VecEnvWrapper(env)

# Set camera to view robot and colon properly
# The unwrapped environment has access to the sim
try:
    import omni.isaac.core.utils.viewports as vp_utils
    # Get the entry position from the environment to center the camera on it
    # Assuming the robot and colon are around entry_positions
    # Set camera position: above and behind the scene
    # Adjust these values based on your scene scale
    eye = [20.0, 10.0, 8.0]  # Camera position (x, y, z) - zoomed out to see larger area
    target = [5.0, 1.0, 1.0]  # Look at point - center of scene
    vp_utils.set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
    print(f"Camera set to eye={eye}, target={target}")
except Exception as e:
    print(f"Could not set camera view: {e}")

# env = VecVideoRecorder(
#     env,
#     video_folder=video_save_path,
#     record_video_trigger=lambda x: x % 10000 == 0,
#     video_length=2500,
#     name_prefix="training"
# )

# Debug your environment
import numpy as np
#pdb.set_trace()
print(f"Obs space: {env.observation_space}")
print(f"Action space: {env.action_space}")

if config["env_config"]["discrete_action_space"]:
    raise NotImplementedError("Discrete action space is not currently supported")
    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=learning_config["verbose"],
        policy_kwargs={"normalize_images": False},
        tensorboard_log=tensorboard_log,
        device=device,

        learning_rate=learning_config["learning_rate"],
        batch_size=learning_config["batch_size"],
        buffer_size=25000,
        learning_starts=100,

        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
    )
else:
    # model = SAC(
    #     "MultiInputPolicy",
    #     env,
    #     verbose=1,
    #     policy_kwargs={"normalize_images": False},
    #     tensorboard_log=tensorboard_log,
    #     device="cuda",
    #     ent_coef=0.5,
    #     learning_starts=5000,
    #     learning_rate=learning_config["learning_rate"],
    #     buffer_size=5000,
    #     batch_size=learning_config["batch_size"],
    #     tau=0.005,
    #     gamma=0.99,
    #     train_freq=10,
    #     gradient_steps=2,
    # )

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        policy_kwargs={"normalize_images": True},
        tensorboard_log=tensorboard_log,
        device="cuda",
        learning_rate=learning_config["learning_rate"],
        batch_size=learning_config["batch_size"],
        n_steps=2048,
        clip_range=0.2,
        ent_coef=0.1,
        n_epochs=15,
        gamma=0.99,
        gae_lambda=0.95,
    )

# Print the policy network structure
logging.debug("Policy architecture:")
logging.debug(model.policy)

# More detailed view of the policy networks
logging.debug("\nActor network:")
logging.debug(model.policy.action_net)

logging.debug("\nValue network:")  
logging.debug(model.policy.value_net)

logging.debug("\nFeature extractor:")
logging.debug(model.policy.features_extractor)

if args_cli.train:
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, # Save every 10k steps
        save_path=model_save_path,
        name_prefix=f"ppo_arc_checkpoint_{timestamp}"
    )

    log_callback = LogEveryNTimesteps(n_steps=500)

    per_env_reward_callback = PerEnvRewardCallback(verbose=1)

    # Create trajectory data saver callback (saves every 10 episodes by default)
    trajectory_callback = TrajectoryDataSaver(
        save_dir=trajectory_save_path,
        save_interval=2,  # Save every 10 episodes
        verbose=1
    )

    model.learn(learning_config["total_timesteps"], callback=[checkpoint_callback, log_callback, per_env_reward_callback, trajectory_callback])
    
    final_model_path = os.path.join(model_save_path, f"ppo_arc_final_{timestamp}")
    model.save(final_model_path)
    logging.info(f"Final model saved to: {final_model_path}")
else:
    from arcgym.utils.teleop_mode import run_teleoperation_mode
    run_teleoperation_mode(env, simulation_app, device=device)
