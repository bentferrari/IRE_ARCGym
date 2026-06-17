import csv
import logging
import os
from datetime import datetime


def save_robot_state(env, save_dir=None, filename_prefix="robot_state"):
    """Save current robot joint positions and root state to a CSV file."""
    try:
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), "saved_states")
        os.makedirs(save_dir, exist_ok=True)

        if hasattr(env, "robot"):
            robot_wrapper = env.robot
        elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "robot"):
            robot_wrapper = env.unwrapped.robot
        elif hasattr(env, "env") and hasattr(env.env, "robot"):
            robot_wrapper = env.env.robot
        else:
            raise AttributeError("Cannot find robot in environment")

        if hasattr(robot_wrapper, "robot"):
            robot = robot_wrapper.robot
        else:
            robot = robot_wrapper

        joint_positions = robot.data.joint_pos.clone().cpu().numpy()
        root_state = robot.data.root_state_w.clone().cpu().numpy()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"{filename_prefix}_{timestamp}.csv")

        num_envs = joint_positions.shape[0]
        num_joints = joint_positions.shape[1]

        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = ["env_id", "timestamp"]
            header.extend(f"joint_{idx}_pos" for idx in range(num_joints))
            header.extend([
                "root_pos_x", "root_pos_y", "root_pos_z",
                "root_quat_w", "root_quat_x", "root_quat_y", "root_quat_z",
                "root_lin_vel_x", "root_lin_vel_y", "root_lin_vel_z",
                "root_ang_vel_x", "root_ang_vel_y", "root_ang_vel_z",
            ])
            writer.writerow(header)

            for env_id in range(num_envs):
                row = [env_id, timestamp]
                row.extend(joint_positions[env_id].tolist())
                row.extend(root_state[env_id].tolist())
                writer.writerow(row)

        logging.info("Saved robot state to %s", filename)
        logging.info("  - Number of environments: %s", num_envs)
        logging.info("  - Number of joints: %s", num_joints)
        return filename
    except Exception as exc:
        logging.error("Failed to save robot state: %s", exc)
        return None
