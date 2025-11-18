import logging

import numpy as np
import torch

from arcgym.utils.keyboard import FPVKeyboard

def run_teleoperation_mode(env, simulation_app, device="cuda"):
    """Run the environment in teleoperation mode."""
    logging.info("Starting teleoperation mode...")
    logging.info("Use keyboard to control the robot. Press 'C' to reset.")
    
    # Teleoperation-specific variables
    should_reset_recording_instance = False
    teleoperation_active = True

    def reset_recording_instance():
        """Reset the environment to its initial state."""
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        logging.info("Reset requested...")
        
    # Initialize keyboard interface
    teleop_interface = FPVKeyboard(lin_sensitivity=1, rot_sensitivity=1)
    teleop_interface.add_callback("C", reset_recording_instance)

    env_id = 0
    
    # Reset environment and teleop interface
    env.reset()
    teleop_interface.reset()
    
    # Main teleoperation loop
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
                
                # Apply actions to environment
                obs, reward, done, info = env.step(actions)
                #from PIL import Image
                #imsave = lambda path, i: Image.fromarray(np.clip((obs[i*3:(i+1)*3]*255), 0, 255).astype(np.uint8).transpose(1,2,0)).save(f"debug_image_{i:02}.png")
                #import IPython
                #IPython.embed()
                # if len(obs) == 12:
                #     imsave(obs, 0)
                #     imsave(obs, 1)
                #     imsave(obs, 2)
                #     imsave(obs, 3)
                #goal_reached = info[env_id]["goal_reached"]
                #l2_norm = info[env_id]["l2_norm"]
                logging.info(f"{reward[env_id].item():.03f}, {done[env_id]}")#, goal_reached, f"{l2_norm:.03f}")
            
            # Always render in teleop mode
            #env.render()

            # Handle reset requests
            if should_reset_recording_instance:
                env.reset()
                should_reset_recording_instance = False
                logging.info("Environment reset completed.")

def pre_process_actions(
    teleop_data: tuple[np.ndarray, bool] | list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    device: str
) -> torch.Tensor:
    """Convert teleop data to the format expected by the environment action space."""
    delta_pose = teleop_data
    delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=device)
    return delta_pose

