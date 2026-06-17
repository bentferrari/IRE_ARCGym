import logging
import os

import numpy as np
import torch

from arcgym.utils.keyboard import FPVKeyboard
from arcgym.utils.robot_state_io import save_robot_state

def run_teleoperation_mode(env, simulation_app, device="cuda"):
    """Run the environment in teleoperation mode.

    Args:
        env: The environment to run teleoperation on.
        simulation_app: The Isaac Sim simulation application.
        device: Device to run on (default: "cuda").
    """
    logging.info("Starting teleoperation mode...")
    logging.info("Use keyboard to control the robot. Press 'C' to reset, 'G' to save state.")

    # Teleoperation-specific variables
    should_reset_recording_instance = False
    teleoperation_active = True

    # Create directory for saved states
    save_dir = os.path.join(os.getcwd(), "saved_states")
    os.makedirs(save_dir, exist_ok=True)

    def reset_recording_instance():
        """Reset the environment to its initial state."""
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        logging.info("Reset requested...")

    # Initialize keyboard interface
    teleop_interface = FPVKeyboard(lin_sensitivity=1, rot_sensitivity=1)
    teleop_interface.add_callback("C", reset_recording_instance)
    teleop_interface.add_callback("G", lambda: save_robot_state(env, save_dir=save_dir))

    env_id = 0

    # Reset environment and teleop interface
    env.reset()
    teleop_interface.reset()

    # Main teleoperation loop with Ctrl+C handling
    try:
        while simulation_app.is_running():
            # Run in inference mode for performance
            with torch.inference_mode():
                # Get keyboard input
                teleop_data = teleop_interface.advance()

                # Apply teleop commands when active
                if teleoperation_active:
                    # Convert teleop data to actions
                    teleop_actions = pre_process_actions(teleop_data, device)
                    actions = teleop_actions.repeat(env.num_envs, 1)

                    # Debug: Log non-zero actions
                    if torch.any(torch.abs(teleop_actions) > 0.01):
                        logging.info(f"Teleop actions: {teleop_actions}")

                    # Apply actions to environment
                    obs, reward, done, info = env.step(actions)

                    logging.info(f"{reward[env_id].item():.03f}, {done[env_id]}")

                # Always render in teleop mode
                #env.render()

                # Handle reset requests
                if should_reset_recording_instance:
                    env.reset()
                    should_reset_recording_instance = False
                    logging.info("Environment reset completed.")

    except KeyboardInterrupt:
        logging.info("\nCtrl+C detected. Exiting teleoperation mode.")

def pre_process_actions(
    teleop_data: tuple[np.ndarray, bool] | list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    device: str
) -> torch.Tensor:
    """Convert teleop data to the format expected by the environment action space."""
    delta_pose = teleop_data
    delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=device)
    return delta_pose
