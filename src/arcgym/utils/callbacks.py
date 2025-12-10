"""Custom callbacks for Stable Baselines3 training."""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch


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
