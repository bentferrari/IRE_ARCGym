import logging
import os
from datetime import datetime
import csv

import numpy as np
import torch

from arcgym.utils.keyboard import FPVKeyboard

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

    def save_robot_state():
        """Save current joint positions and root_state pose to CSV."""
        try:
            # Get robot from environment - handle wrapped environments
            if hasattr(env, 'robot'):
                robot_wrapper = env.robot
            elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'robot'):
                robot_wrapper = env.unwrapped.robot
            elif hasattr(env, 'env') and hasattr(env.env, 'robot'):
                robot_wrapper = env.env.robot
            else:
                raise AttributeError("Cannot find robot in environment")

            # Get the articulation
            if hasattr(robot_wrapper, 'robot'):
                robot = robot_wrapper.robot  # Access the articulation
            else:
                robot = robot_wrapper  # It might be the articulation directly

            # Get joint positions and root state
            joint_positions = robot.data.joint_pos.clone().cpu().numpy()  # Shape: (num_envs, num_joints)
            root_state = robot.data.root_state_w.clone().cpu().numpy()    # Shape: (num_envs, 13)

            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"robot_state_{timestamp}.csv")

            num_envs = joint_positions.shape[0]
            num_joints = joint_positions.shape[1]

            # Write to CSV
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                header = ['env_id', 'timestamp']

                # Joint position columns
                for i in range(num_joints):
                    header.append(f'joint_{i}_pos')

                # Root state columns
                header.extend([
                    'root_pos_x', 'root_pos_y', 'root_pos_z',
                    'root_quat_w', 'root_quat_x', 'root_quat_y', 'root_quat_z',
                    'root_lin_vel_x', 'root_lin_vel_y', 'root_lin_vel_z',
                    'root_ang_vel_x', 'root_ang_vel_y', 'root_ang_vel_z'
                ])

                writer.writerow(header)

                # Write data for each environment
                for env_id in range(num_envs):
                    row = [env_id, timestamp]

                    # Add joint positions
                    row.extend(joint_positions[env_id].tolist())

                    # Add root state
                    row.extend(root_state[env_id].tolist())

                    writer.writerow(row)

            logging.info(f"Saved robot state to {filename}")
            logging.info(f"  - Number of environments: {num_envs}")
            logging.info(f"  - Number of joints: {num_joints}")

        except Exception as e:
            logging.error(f"Failed to save robot state: {e}")

    # Initialize keyboard interface
    teleop_interface = FPVKeyboard(lin_sensitivity=1, rot_sensitivity=1)
    teleop_interface.add_callback("C", reset_recording_instance)
    teleop_interface.add_callback("G", save_robot_state)

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
