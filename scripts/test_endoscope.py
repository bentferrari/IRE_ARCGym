import time
import torch
import numpy as np
import carb
import omni.kit.app

from isaaclab.app import AppLauncher
from isaaclab.sim import SceneCfg, SimulationContext
from isaaclab.utils.assets import clone_prim
from isaaclab.utils.viewports import set_camera_view
from isaaclab.envs import DirectRLEnv

from arcgym.assets.robots.magnetic_endoscope import RobotEndoscopeChain

# Teleop key mapping
KEYMAP = {
    carb.keyboard.KeyboardEvent.KEY_W: np.array([0, 0,  1, 0, 0, 0]),   # Forward
    carb.keyboard.KeyboardEvent.KEY_S: np.array([0, 0, -1, 0, 0, 0]),   # Backward
    carb.keyboard.KeyboardEvent.KEY_A: np.array([ 1, 0, 0, 0, 0, 0]),   # Left
    carb.keyboard.KeyboardEvent.KEY_D: np.array([-1, 0, 0, 0, 0, 0]),   # Right
    carb.keyboard.KeyboardEvent.KEY_E: np.array([0,  1, 0, 0, 0, 0]),   # Up
    carb.keyboard.KeyboardEvent.KEY_Q: np.array([0, -1, 0, 0, 0, 0]),   # Down

    # Rotations
    carb.keyboard.KeyboardEvent.KEY_J: np.array([0, 0, 0, 0,  1, 0]),  # Pitch+
    carb.keyboard.KeyboardEvent.KEY_L: np.array([0, 0, 0, 0, -1, 0]),  # Pitch−
    carb.keyboard.KeyboardEvent.KEY_I: np.array([0, 0, 0, 1,  0, 0]),  # Roll+
    carb.keyboard.KeyboardEvent.KEY_K: np.array([0, 0, 0,-1,  0, 0]),  # Roll−
    carb.keyboard.KeyboardEvent.KEY_U: np.array([0, 0, 0, 0, 0,  1]),  # Yaw+
    carb.keyboard.KeyboardEvent.KEY_O: np.array([0, 0, 0, 0, 0, -1]),  # Yaw−
}

###############################################################################
# Main Teleop App
###############################################################################

def main():
    # Launch sim
    app_launcher = AppLauncher(headless=False)
    simulation_app = app_launcher.app

    # Simulation config
    sim_cfg = SimulationContext.SimCfg(dt=1/60, substeps=2)
    sim = SimulationContext(sim_cfg)

    # Create scene
    scene_cfg = SceneCfg(num_envs=1)
    scene = scene_cfg.create_scene(sim)

    # Robot config dictionary
    robot_cfg = {
        "env_config": { "use_camera": False },
        "simulation_config": {
            "render_interval": 1,
            "dt": 1/60,
        },
        "robot_config": {
            "num_links_total": 20,
            "link_radius": 0.01,
            "link_height": 0.02,
            "passive_stiffness": 3000,
            "passive_damping": 150,
            "camera_resolution": (64, 64),
            "front_light_color": (1.0, 1.0, 1.0),
            "front_light_intensity": 2000,
            "front_camera_focal_length": 18,
            "front_camera_focus_distance": 0.03,
            "front_camera_horizontal_aperture": 20.0,
            "front_camera_clipping_range": (0.001, 10.0),
        }
    }

    # Spawn robot
    isaac_cfg = RobotEndoscopeChain.make_isaac_config(robot_cfg)
    robot = RobotEndoscopeChain(
        scene=scene,
        config=robot_cfg,
        isaac_cfg=isaac_cfg,
        init_pos=(0.0, 0.0, 0.1),
        init_rot=(1, 0, 0, 0),
        device="cpu"
    )

    scene.init()  # build stage
    robot.reset() # initialize robot

    # Teleop control vector
    action = np.zeros(6, dtype=np.float32)

    # Setup viewport camera
    set_camera_view(
        eye=np.array([0.5, 0.0, 0.3]),
        target=np.array([0.0, 0.0, 0.0])
    )

    # Keyboard interface
    keyboard = omni.kit.app.get_app().get_keyboard()

    print("\n=== TELEOPERATION READY ===")
    print("W/S = forward/back")
    print("A/D = left/right")
    print("Q/E = down/up")
    print("I/K = roll")
    print("J/L = pitch")
    print("U/O = yaw")
    print("CTRL+C to quit")
    print("===========================\n")

    # Main loop
    while simulation_app.is_running():
        # Zero-out action each frame
        action[:] = 0.0

        # Read keyboard events
        for event in keyboard.consume_events():
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if event.key in KEYMAP:
                    action += KEYMAP[event.key]

        # Convert action → torch
        action_tensor = torch.tensor(action, dtype=torch.float32)[None, :]

        # Apply action
        robot.apply_action(action_tensor, action_scale=0.2)

        # Step simulation
        sim.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()
