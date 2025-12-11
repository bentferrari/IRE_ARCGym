"""Custom callbacks for Stable Baselines3 training."""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch
import os
import json
from pathlib import Path
from PIL import Image


class PerEnvRewardCallback(BaseCallback):
    """
    Callback for logging per-environment rewards and episode statistics to TensorBoard.

    This callback logs individual environment rewards instead of just the mean,
    allowing you to track performance of each parallel environment separately.
    """

    def __init__(self, verbose=0):
        """
        Initialize the callback.

        Args:
            verbose: Verbosity level (0: not verbose, 1: info, 2: debug)
        """
        super().__init__(verbose)
        self.episode_rewards = {}
        self.episode_lengths = {}
        self.episode_counts = {}
        self.episode_stresses = {}  # Track colon stress per episode
        self.episode_successes = {}  # Track success/failure for each episode

    def _on_step(self) -> bool:
        """
        Called at each step of the training loop.

        Logs per-environment rewards and statistics when episodes complete.

        Returns:
            bool: Always returns True to continue training
        """
        # Get access to the unwrapped environment to access colon stress
        try:
            # Navigate through the wrappers to get to the base environment
            env = self.training_env.envs[0]
            # Unwrap to get to ARCIsaacEnv
            while hasattr(env, 'env'):
                env = env.env

            # Get current colon stress for all environments
            if hasattr(env, 'colon') and hasattr(env.colon, 'get_accumulated_stress'):
                current_stress = env.colon.get_accumulated_stress()
            else:
                current_stress = None
        except Exception as e:
            if self.verbose > 1:
                print(f"Could not access colon stress: {e}")
            current_stress = None

        # Check if any episodes finished in any environment
        if 'dones' in self.locals and 'infos' in self.locals:
            dones = self.locals['dones']
            infos = self.locals['infos']

            for env_idx, (done, info) in enumerate(zip(dones, infos)):
                if done and 'episode' in info:
                    # Log individual environment reward
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']

                    # Track statistics for each environment
                    if env_idx not in self.episode_rewards:
                        self.episode_rewards[env_idx] = []
                        self.episode_lengths[env_idx] = []
                        self.episode_stresses[env_idx] = []
                        self.episode_successes[env_idx] = []
                        self.episode_counts[env_idx] = 0

                    self.episode_rewards[env_idx].append(episode_reward)
                    self.episode_lengths[env_idx].append(episode_length)
                    self.episode_counts[env_idx] += 1

                    # Check if episode was successful (goal_reached)
                    is_success = False

                    # Debug: Print what's in info for goal_reached
                    if self.verbose > 0 and self.episode_counts[env_idx] <= 3:
                        print(f"Debug - Env {env_idx}: 'goal_reached' in info: {'goal_reached' in info}")
                        if 'goal_reached' in info:
                            print(f"  goal_reached type: {type(info['goal_reached'])}")
                            print(f"  goal_reached value: {info['goal_reached']}")

                    if 'goal_reached' in info:
                        is_success = bool(info['goal_reached'])
                        if self.verbose > 0 and is_success:
                            print(f"SUCCESS detected in env {env_idx}!")

                    self.episode_successes[env_idx].append(1 if is_success else 0)

                    # Get and log colon stress if available
                    # First try to get it from info dict (most reliable)
                    stress_value = None

                    # Debug: Print what's in info
                    if self.verbose > 0 and self.episode_counts[env_idx] == 1:
                        print(f"Info keys for env {env_idx}: {info.keys()}")
                        if 'colon_stress' in info:
                            print(f"colon_stress type: {type(info['colon_stress'])}, value: {info['colon_stress']}")

                    if 'colon_stress' in info and info['colon_stress'] is not None:
                        try:
                            stress_value = float(info['colon_stress'])
                            if self.verbose > 0:
                                print(f"Got stress from info: {stress_value}")
                        except Exception as e:
                            print(f"Error extracting stress from info: {e}")
                            import traceback
                            traceback.print_exc()

                    # Fallback to direct environment access
                    if stress_value is None and current_stress is not None and env_idx < len(current_stress):
                        try:
                            stress_value = float(current_stress[env_idx].cpu().item())
                        except Exception:
                            pass

                    # Log stress if we got it
                    if stress_value is not None:
                        self.episode_stresses[env_idx].append(stress_value)
                        self.logger.record(f'colon_stress/env_{env_idx}_stress', stress_value)
                        # Also log under rollout_per_env for convenience
                        self.logger.record(f'rollout_per_env/env_{env_idx}_colon_stress', stress_value)

                    # Log to TensorBoard
                    self.logger.record(f'rollout_per_env/env_{env_idx}_reward', episode_reward)
                    self.logger.record(f'rollout_per_env/env_{env_idx}_length', episode_length)
                    self.logger.record(f'rollout_per_env/env_{env_idx}_episode_count', self.episode_counts[env_idx])

                    # Calculate success rate over last 10 episodes
                    if len(self.episode_successes[env_idx]) >= 10:
                        recent_successes_10 = self.episode_successes[env_idx][-10:]
                    else:
                        recent_successes_10 = self.episode_successes[env_idx]

                    success_rate_10 = np.mean(recent_successes_10) if recent_successes_10 else 0.0

                    # Log success rate
                    self.logger.record(f'rollout_per_env/env_{env_idx}_success_rate_10ep', success_rate_10)
                    self.logger.record(f'rollout_per_env/env_{env_idx}_is_success', 1 if is_success else 0)

                    # Log running statistics (mean over last 100 episodes per env)
                    if len(self.episode_rewards[env_idx]) >= 100:
                        recent_rewards = self.episode_rewards[env_idx][-100:]
                        recent_lengths = self.episode_lengths[env_idx][-100:]
                        recent_stresses = self.episode_stresses[env_idx][-100:] if self.episode_stresses[env_idx] else []
                    else:
                        recent_rewards = self.episode_rewards[env_idx]
                        recent_lengths = self.episode_lengths[env_idx]
                        recent_stresses = self.episode_stresses[env_idx]

                    self.logger.record(f'rollout_per_env/env_{env_idx}_mean_reward_100ep', np.mean(recent_rewards))
                    self.logger.record(f'rollout_per_env/env_{env_idx}_mean_length_100ep', np.mean(recent_lengths))

                    if recent_stresses:
                        self.logger.record(f'colon_stress/env_{env_idx}_mean_stress_100ep', np.mean(recent_stresses))
                        # Also log under rollout_per_env for convenience
                        self.logger.record(f'rollout_per_env/env_{env_idx}_mean_colon_stress_100ep', np.mean(recent_stresses))

                    if self.verbose > 0:
                        stress_str = f", Stress={stress_value:.2f}" if stress_value is not None else ""
                        success_str = f", Success={is_success}, SR10={success_rate_10:.1%}"
                        print(f"Env {env_idx} - Episode {self.episode_counts[env_idx]}: "
                              f"Reward={episode_reward:.2f}, Length={episode_length}{stress_str}{success_str}")

        return True

    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        if self.verbose > 0:
            print(f"Starting training with per-environment reward logging")

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.verbose > 0:
            print("\nTraining complete. Per-environment statistics:")
            for env_idx in sorted(self.episode_counts.keys()):
                mean_reward = np.mean(self.episode_rewards[env_idx])
                mean_length = np.mean(self.episode_lengths[env_idx])

                # Calculate overall success rate
                if env_idx in self.episode_successes and self.episode_successes[env_idx]:
                    overall_success_rate = np.mean(self.episode_successes[env_idx])
                    success_str = f", Success Rate={overall_success_rate:.1%}"
                else:
                    success_str = ""

                stress_str = ""
                if env_idx in self.episode_stresses and self.episode_stresses[env_idx]:
                    mean_stress = np.mean(self.episode_stresses[env_idx])
                    stress_str = f", Mean Stress={mean_stress:.2f}"

                print(f"  Env {env_idx}: {self.episode_counts[env_idx]} episodes, "
                      f"Mean Reward={mean_reward:.2f}, Mean Length={mean_length:.2f}{success_str}{stress_str}")


class TrainingMetricsCallback(BaseCallback):
    """
    Callback for logging training metrics including value loss to TensorBoard.

    This ensures value_loss and other training metrics are properly logged.
    """

    def __init__(self, verbose=0):
        """
        Initialize the callback.

        Args:
            verbose: Verbosity level (0: not verbose, 1: info, 2: debug)
        """
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """
        Called at each step. Logs training metrics when available.

        Returns:
            bool: Always returns True to continue training
        """
        # Training metrics are available after a training update
        # Check if the model has just performed a training update
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            # The logger already records these automatically, but we can ensure they're there
            # SB3 PPO logs: value_loss, policy_gradient_loss, approx_kl, etc.
            pass

        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of a rollout (collection phase).
        This is where PPO performs its training updates and logs metrics.
        """
        # After rollout, PPO trains and logs metrics
        # The metrics should already be logged by PPO.train()
        # But we can add custom logging here if needed
        if self.verbose > 1:
            print("Rollout ended, training metrics should be logged")


class TrajectoryDataSaver(BaseCallback):
    """
    Callback for saving trajectory data (root_state) and corresponding images every N episodes.

    This callback:
    - Tracks the root_state trajectory for each environment during episodes
    - Captures corresponding RGB images at each timestep
    - Saves trajectory and images to disk every N episodes
    - Organizes data in a structured folder hierarchy

    Data is saved in the following structure:
        save_dir/
            env_0/
                episode_0000/
                    trajectory.npz       # Contains root_states array and metadata
                    images/
                        frame_0000.png
                        frame_0001.png
                        ...
                episode_0010/
                    ...
            env_1/
                ...
    """

    def __init__(self, save_dir="trajectory_data", save_interval=10, verbose=0):
        """
        Initialize the trajectory data saver callback.

        Args:
            save_dir: Root directory for saving trajectory data
            save_interval: Save trajectory data every N episodes (default: 10)
            verbose: Verbosity level (0: not verbose, 1: info, 2: debug)
        """
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_interval = save_interval

        # Track current episode data for each environment
        # Format: {env_idx: {'root_states': [], 'images': [], 'episode_num': int}}
        self.current_episode_data = {}

        # Track episode counts per environment
        self.episode_counts = {}

        # Create base save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose > 0:
            print(f"TrajectoryDataSaver initialized: saving to '{save_dir}' every {save_interval} episodes")

    def _on_step(self) -> bool:
        """
        Called at each step. Captures root_state and image data.

        Returns:
            bool: Always returns True to continue training
        """
        try:
            # Get access to the unwrapped environment
            # Sb3VecEnvWrapper has 'env' attribute, not 'envs'
            if hasattr(self.training_env, 'env'):
                env = self.training_env.env
            else:
                env = self.training_env

            # Unwrap further if needed (but stop when we find the robot)
            while hasattr(env, 'env') and not hasattr(env, 'robot'):
                env = env.env

            # Get root_state data for all environments
            # root_state shape: (num_envs, 13) with [pos(3), quat(4), lin_vel(3), ang_vel(3)]
            if hasattr(env, 'robot') and hasattr(env.robot, 'get_pose'):
                root_states = env.robot.get_pose()  # Tensor of shape (num_envs, 13)

                if self.verbose > 1:
                    print(f"[TrajectoryDataSaver] Captured root_states, shape: {root_states.shape}")

                # Get RGB images for all environments
                rgb_images = None
                if hasattr(env, '_camera_data') and env._camera_data is not None:
                    rgb_images = env._camera_data  # Should be (num_envs, C, H, W)

                # Process each environment
                num_envs = root_states.shape[0]
                for env_idx in range(num_envs):
                    # Initialize tracking for this environment if needed
                    if env_idx not in self.current_episode_data:
                        self.current_episode_data[env_idx] = {
                            'root_states': [],
                            'images': [],
                            'episode_num': self.episode_counts.get(env_idx, 0)
                        }

                    # Store root_state for this environment
                    root_state = root_states[env_idx].cpu().numpy()
                    self.current_episode_data[env_idx]['root_states'].append(root_state)

                    # Debug: Print first and periodic positions to check if they're changing
                    step_num = len(self.current_episode_data[env_idx]['root_states'])
                    if self.verbose > 1 and env_idx == 0:
                        if step_num <= 3 or step_num % 20 == 0:
                            pos = root_state[:3]
                            vel = root_state[7:10]
                            print(f"[TrajectoryDataSaver] Env {env_idx} step {step_num}: pos=[{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}], vel=[{vel[0]:.4f}, {vel[1]:.4f}, {vel[2]:.4f}]")

                    # Store image if available
                    if rgb_images is not None and env_idx < rgb_images.shape[0]:
                        # Convert from (C, H, W) to (H, W, C) for PIL
                        img = rgb_images[env_idx].cpu().numpy()
                        if img.shape[0] == 3:  # If channels first
                            img = np.transpose(img, (1, 2, 0))

                        # Ensure uint8 format
                        if img.dtype != np.uint8:
                            if img.max() <= 1.0:
                                img = (img * 255).astype(np.uint8)
                            else:
                                img = np.clip(img, 0, 255).astype(np.uint8)

                        self.current_episode_data[env_idx]['images'].append(img)

            # Check if any episodes finished
            if 'dones' in self.locals and 'infos' in self.locals:
                dones = self.locals['dones']
                infos = self.locals['infos']

                if self.verbose > 1:
                    print(f"[TrajectoryDataSaver] Checking dones: {dones}")

                for env_idx, (done, info) in enumerate(zip(dones, infos)):
                    if done:
                        if self.verbose > 1:
                            print(f"[TrajectoryDataSaver] Env {env_idx} done. 'episode' in info: {'episode' in info}")
                            if 'episode' not in info:
                                print(f"[TrajectoryDataSaver] Info keys: {list(info.keys())}")

                    if done and 'episode' in info:
                        # Episode finished for this environment
                        if env_idx not in self.episode_counts:
                            self.episode_counts[env_idx] = 0

                        self.episode_counts[env_idx] += 1
                        episode_num = self.episode_counts[env_idx]

                        if self.verbose > 1:
                            print(f"[TrajectoryDataSaver] Env {env_idx} completed episode {episode_num}")

                        # Check if we should save this episode
                        should_save = (episode_num % self.save_interval) == 0

                        if self.verbose > 1:
                            print(f"[TrajectoryDataSaver] Should save: {should_save} (episode {episode_num} % interval {self.save_interval})")

                        if should_save and env_idx in self.current_episode_data:
                            self._save_episode_data(env_idx, episode_num, info)

                            if self.verbose > 0:
                                print(f"✓ Saved trajectory data for env {env_idx}, episode {episode_num}")
                        elif should_save:
                            if self.verbose > 1:
                                print(f"[TrajectoryDataSaver] WARNING: Should save but env {env_idx} not in current_episode_data")

                        # Reset tracking for next episode
                        self.current_episode_data[env_idx] = {
                            'root_states': [],
                            'images': [],
                            'episode_num': episode_num
                        }

        except Exception as e:
            print(f"ERROR in TrajectoryDataSaver._on_step: {e}")
            import traceback
            traceback.print_exc()

        return True

    def _save_episode_data(self, env_idx, episode_num, info):
        """
        Save trajectory and image data for a completed episode.

        Args:
            env_idx: Environment index
            episode_num: Episode number
            info: Info dict from the environment containing episode metadata
        """
        try:
            episode_data = self.current_episode_data[env_idx]
            root_states = episode_data['root_states']
            images = episode_data['images']

            if not root_states:
                if self.verbose > 1:
                    print(f"No trajectory data to save for env {env_idx}, episode {episode_num}")
                return

            # Create episode directory
            episode_dir = self.save_dir / f"env_{env_idx}" / f"episode_{episode_num:06d}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            # Save trajectory data as numpy compressed array
            root_states_array = np.array(root_states)  # Shape: (num_steps, 13)

            # Prepare metadata
            metadata = {
                'env_idx': env_idx,
                'episode_num': episode_num,
                'num_steps': len(root_states),
                'episode_reward': float(info['episode']['r']) if 'episode' in info else None,
                'episode_length': int(info['episode']['l']) if 'episode' in info else None,
                'goal_reached': bool(info.get('goal_reached', False)),
                'colon_stress': float(info['colon_stress']) if 'colon_stress' in info and info['colon_stress'] is not None else None,
            }

            # Save trajectory and metadata
            trajectory_file = episode_dir / "trajectory.npz"
            np.savez_compressed(
                trajectory_file,
                root_states=root_states_array,
                metadata=json.dumps(metadata)
            )

            # Save metadata as JSON for easy reading
            metadata_file = episode_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Save images if available
            if images:
                images_dir = episode_dir / "images"
                images_dir.mkdir(exist_ok=True)

                for step_idx, img in enumerate(images):
                    img_path = images_dir / f"frame_{step_idx:06d}.png"
                    Image.fromarray(img).save(img_path)

                if self.verbose > 1:
                    print(f"  Saved {len(images)} images for env {env_idx}, episode {episode_num}")

            if self.verbose > 0:
                print(f"  Trajectory shape: {root_states_array.shape}, "
                      f"Reward: {metadata.get('episode_reward', 'N/A'):.2f}, "
                      f"Success: {metadata.get('goal_reached', False)}")

        except Exception as e:
            print(f"Error saving episode data for env {env_idx}, episode {episode_num}: {e}")
            import traceback
            traceback.print_exc()

    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        if self.verbose > 0:
            print(f"TrajectoryDataSaver: Starting trajectory collection")
            print(f"  Save directory: {self.save_dir.absolute()}")
            print(f"  Save interval: every {self.save_interval} episodes")

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.verbose > 0:
            print("\nTrajectoryDataSaver: Training complete")
            total_saved = sum(1 for env_idx in self.episode_counts
                            for ep in range(1, self.episode_counts[env_idx] + 1)
                            if ep % self.save_interval == 0)
            print(f"  Total episodes saved: {total_saved}")
            print(f"  Data location: {self.save_dir.absolute()}")
