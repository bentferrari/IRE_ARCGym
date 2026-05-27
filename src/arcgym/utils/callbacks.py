"""Custom callbacks for Stable Baselines3 training."""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch
import os
import json
import csv
from pathlib import Path
from PIL import Image


class PerEnvRewardCallback(BaseCallback):
    """
    Callback for logging per-environment rewards and episode statistics to TensorBoard.

    This callback logs individual environment rewards instead of just the mean,
    allowing you to track performance of each parallel environment separately.
    """

    def __init__(self, verbose=0, save_plot_data=True, plot_data_file="episode_rewards.npz"):
        """
        Initialize the callback.

        Args:
            verbose: Verbosity level (0: not verbose, 1: info, 2: debug)
            save_plot_data: Whether to save episode rewards for plotting (default: True)
            plot_data_file: Filename to save episode reward data (default: "episode_rewards.npz")
        """
        super().__init__(verbose)
        self.episode_rewards = {}
        self.episode_lengths = {}
        self.episode_counts = {}
        self.episode_stresses = {}  # Track colon stress per episode
        self.episode_successes = {}  # Track success/failure for each episode

        # Track overall statistics across all environments
        self.total_episodes = 0
        self.total_successes = 0

        # For plotting: store all rewards in order
        self.save_plot_data = save_plot_data
        self.plot_data_file = plot_data_file
        self.all_episode_rewards = []  # List of (episode_num, env_idx, reward)
        self.global_episode_counter = 0

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

                    # Store for plotting
                    if self.save_plot_data:
                        self.all_episode_rewards.append({
                            'global_episode': self.global_episode_counter,
                            'env_idx': env_idx,
                            'reward': episode_reward,
                            'length': episode_length
                        })
                        self.global_episode_counter += 1

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

                    # Update overall statistics
                    self.total_episodes += 1
                    if is_success:
                        self.total_successes += 1

                    # Get center alignment if available
                    center_alignment = None
                    if 'center_alignment' in info:
                        try:
                            center_alignment = float(info['center_alignment'])
                        except Exception:
                            pass

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

                    # Calculate success rate over last 10 episodes (rolling window)
                    if len(self.episode_successes[env_idx]) >= 10:
                        recent_successes_10 = self.episode_successes[env_idx][-10:]
                    else:
                        recent_successes_10 = self.episode_successes[env_idx]

                    success_rate_10 = np.mean(recent_successes_10) if recent_successes_10 else 0.0

                    # Calculate cumulative success rate for this environment
                    cumulative_success_rate = np.mean(self.episode_successes[env_idx])

                    # Log success rates
                    self.logger.record(f'rollout_per_env/env_{env_idx}_success_rate_10ep', success_rate_10)
                    self.logger.record(f'rollout_per_env/env_{env_idx}_success_rate_cumulative', cumulative_success_rate)
                    self.logger.record(f'rollout_per_env/env_{env_idx}_is_success', 1 if is_success else 0)
                    self.logger.record(f'rollout_per_env/env_{env_idx}_total_successes', sum(self.episode_successes[env_idx]))
                    self.logger.record(f'rollout_per_env/env_{env_idx}_total_episodes', len(self.episode_successes[env_idx]))

                    # Log center alignment if available
                    if center_alignment is not None:
                        self.logger.record(f'rollout_per_env/env_{env_idx}_final_center_alignment', center_alignment)

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
                        success_str = f", Success={is_success}, SR10={success_rate_10:.1%}, Cumulative={cumulative_success_rate:.1%}"
                        alignment_str = f", Alignment={center_alignment:.2f}" if center_alignment is not None else ""
                        print(f"Env {env_idx} - Episode {self.episode_counts[env_idx]}: "
                              f"Reward={episode_reward:.2f}, Length={episode_length}{stress_str}{success_str}{alignment_str}")

        # Log overall success rate across all environments (after processing all dones)
        if self.total_episodes > 0:
            overall_success_rate = self.total_successes / self.total_episodes
            self.logger.record('success/overall_success_rate', overall_success_rate)
            self.logger.record('success/total_successes', self.total_successes)
            self.logger.record('success/total_episodes', self.total_episodes)

            # Calculate success rate over last 100 episodes across all environments
            all_recent_successes = []
            for env_idx in self.episode_successes:
                if len(self.episode_successes[env_idx]) >= 100:
                    all_recent_successes.extend(self.episode_successes[env_idx][-100:])
                else:
                    all_recent_successes.extend(self.episode_successes[env_idx])

            if all_recent_successes:
                # Take only the most recent 100 across all environments
                recent_100 = all_recent_successes[-100:] if len(all_recent_successes) >= 100 else all_recent_successes
                success_rate_recent_100 = np.mean(recent_100)
                self.logger.record('success/success_rate_recent_100ep', success_rate_recent_100)

        return True

    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        if self.verbose > 0:
            print(f"Starting training with per-environment reward logging")

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.verbose > 0:
            print("\n" + "="*80)
            print("TRAINING COMPLETE - FINAL STATISTICS")
            print("="*80)

            # Overall statistics across all environments
            if self.total_episodes > 0:
                overall_success_rate = self.total_successes / self.total_episodes
                print(f"\nOVERALL SUCCESS RATE: {overall_success_rate:.2%} ({self.total_successes}/{self.total_episodes})")
            else:
                print("\nNo episodes completed.")

            print("\nPer-environment statistics:")
            for env_idx in sorted(self.episode_counts.keys()):
                mean_reward = np.mean(self.episode_rewards[env_idx])
                mean_length = np.mean(self.episode_lengths[env_idx])

                # Calculate success rate for this environment
                if env_idx in self.episode_successes and self.episode_successes[env_idx]:
                    env_success_rate = np.mean(self.episode_successes[env_idx])
                    env_successes = sum(self.episode_successes[env_idx])
                    env_episodes = len(self.episode_successes[env_idx])
                    success_str = f", Success Rate={env_success_rate:.1%} ({env_successes}/{env_episodes})"
                else:
                    success_str = ""

                stress_str = ""
                if env_idx in self.episode_stresses and self.episode_stresses[env_idx]:
                    mean_stress = np.mean(self.episode_stresses[env_idx])
                    stress_str = f", Mean Stress={mean_stress:.2f}"

                print(f"  Env {env_idx}: {self.episode_counts[env_idx]} episodes, "
                      f"Mean Reward={mean_reward:.2f}, Mean Length={mean_length:.2f}{success_str}{stress_str}")

            print("="*80)

        # Save episode rewards for plotting
        if self.save_plot_data and self.all_episode_rewards:
            # Convert to arrays for easy plotting
            data = {
                'episodes': np.array([d['global_episode'] for d in self.all_episode_rewards]),
                'env_indices': np.array([d['env_idx'] for d in self.all_episode_rewards]),
                'rewards': np.array([d['reward'] for d in self.all_episode_rewards]),
                'lengths': np.array([d['length'] for d in self.all_episode_rewards]),
            }
            np.savez(self.plot_data_file, **data)
            if self.verbose > 0:
                print(f"\nSaved episode reward data to {self.plot_data_file} for plotting")


class RewardAblationMetricsCallback(BaseCallback):
    """
    Log reward-ablation components and normalized mean step reward.

    The ablation metrics are emitted by FinalReward through the info dict:
    s_c is the center-alignment score, s_1 the depth-existence cue, s_2 the
    deep-region area cue, s_3 the depth-contrast/geometric-confidence cue, and
    s_o the full lumen-visibility score. This callback records both partial
    reward designs and the full reward in one reproducible CSV stream.
    """

    METRIC_KEYS = [
        "raw_step_reward",
        "normalized_step_reward",
        "s_c",
        "s_1",
        "s_2",
        "s_3",
        "s_o",
        "normalized_progress",
        "roi_aligned",
        "lumen_visible",
    ]

    def __init__(self, save_dir, reward_variant, constrained, log_interval_steps=500, verbose=0):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.reward_variant = reward_variant
        self.constrained = constrained
        self.log_interval_steps = max(1, int(log_interval_steps))
        self.csv_path = self.save_dir / "training_log.csv"
        self.summary_path = self.save_dir / "training_metrics_summary.json"
        self.rows = []
        self.total_episodes = 0
        self.total_successes = 0
        self._csv_file = None
        self._writer = None
        self._latest_row = None
        self._last_written_timestep = None

    def _on_training_start(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "timestep",
            "reward_variant",
            "constrained",
            "train_reward_mean",
            "raw_mean_step_reward",
            "normalized_mean_step_reward",
            "mean_s_c",
            "mean_s_1",
            "mean_s_2",
            "mean_s_3",
            "mean_s_o",
            "normalized_progress",
            "roi_alignment_rate",
            "lumen_visible_ratio",
            "success_rate",
            "episodes",
        ]
        self._csv_file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._csv_file.flush()

    def _extract_metric(self, infos, key):
        values = []
        for info in infos:
            if key not in info or info[key] is None:
                continue
            try:
                value = info[key]
                if hasattr(value, "item"):
                    value = value.item()
                values.append(float(value))
            except Exception:
                continue
        return values

    def _build_row(self, rewards, metric_values):
        success_rate = self.total_successes / self.total_episodes if self.total_episodes else 0.0
        return {
            "timestep": int(self.num_timesteps),
            "reward_variant": self.reward_variant,
            "constrained": bool(self.constrained),
            "train_reward_mean": float(np.mean(rewards)) if rewards is not None else None,
            "raw_mean_step_reward": float(np.mean(metric_values["raw_step_reward"])) if metric_values["raw_step_reward"] else None,
            "normalized_mean_step_reward": float(np.mean(metric_values["normalized_step_reward"])) if metric_values["normalized_step_reward"] else None,
            "mean_s_c": float(np.mean(metric_values["s_c"])) if metric_values["s_c"] else None,
            "mean_s_1": float(np.mean(metric_values["s_1"])) if metric_values["s_1"] else None,
            "mean_s_2": float(np.mean(metric_values["s_2"])) if metric_values["s_2"] else None,
            "mean_s_3": float(np.mean(metric_values["s_3"])) if metric_values["s_3"] else None,
            "mean_s_o": float(np.mean(metric_values["s_o"])) if metric_values["s_o"] else None,
            "normalized_progress": float(np.mean(metric_values["normalized_progress"])) if metric_values["normalized_progress"] else None,
            "roi_alignment_rate": float(np.mean(metric_values["roi_aligned"])) if metric_values["roi_aligned"] else None,
            "lumen_visible_ratio": float(np.mean(metric_values["lumen_visible"])) if metric_values["lumen_visible"] else None,
            "success_rate": success_rate,
            "episodes": int(self.total_episodes),
        }

    def _write_row(self, row):
        if row is None:
            return
        self.rows.append(row)
        self._last_written_timestep = row["timestep"]
        if self._writer is not None:
            self._writer.writerow(row)
            self._csv_file.flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if done and isinstance(info, dict) and "episode" in info:
                self.total_episodes += 1
                is_success = bool(info.get("goal_reached", False))
                if is_success:
                    self.total_successes += 1

        metric_values = {key: self._extract_metric(infos, key) for key in self.METRIC_KEYS}

        for key, values in metric_values.items():
            if not values:
                continue
            mean_value = float(np.mean(values))
            self.logger.record(f"reward_ablation/{key}_mean", mean_value)
            for env_idx, value in enumerate(values):
                self.logger.record(f"reward_ablation_per_env/env_{env_idx}_{key}", value)

        if rewards is not None:
            self.logger.record("reward_ablation/train_reward_mean", float(np.mean(rewards)))

        self._latest_row = self._build_row(rewards, metric_values)
        should_write = self.n_calls == 1 or self.num_timesteps % self.log_interval_steps == 0
        if should_write:
            self._write_row(self._latest_row)
        return True

    def _on_training_end(self) -> None:
        if (
            self._latest_row is not None
            and self._latest_row["timestep"] != self._last_written_timestep
        ):
            self._write_row(self._latest_row)

        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

        summary = {
            "reward_variant": self.reward_variant,
            "constrained": bool(self.constrained),
            "total_episodes": int(self.total_episodes),
            "total_successes": int(self.total_successes),
            "success_rate": self.total_successes / self.total_episodes if self.total_episodes else 0.0,
        }
        if self.rows:
            final_row = self.rows[-1]
            summary.update({f"final_{key}": value for key, value in final_row.items()})
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)


class SuccessRateDataSaver(BaseCallback):
    """
    Callback for saving success rate data for all environments to a JSON file.

    This callback tracks and periodically saves:
    - Per-environment success rates (rolling window and cumulative)
    - Overall success rate across all environments
    - Episode counts and success counts per environment
    - Timestamped history of success rates

    Data is saved to: trajectory_data/success_rate_data.json
    """

    def __init__(self, save_dir="trajectory_data", save_interval_episodes=10, verbose=0):
        """
        Initialize the success rate data saver callback.

        Args:
            save_dir: Directory to save success rate data (default: "trajectory_data")
            save_interval_episodes: Save data every N episodes across all envs (default: 10)
            verbose: Verbosity level (0: not verbose, 1: info, 2: debug)
        """
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_interval_episodes = save_interval_episodes

        # Track success/failure for each environment
        # Format: {env_idx: [1, 0, 1, ...]} where 1=success, 0=failure
        self.episode_successes = {}
        self.episode_rewards = {}
        self.episode_lengths = {}

        # Track overall statistics
        self.total_episodes = 0
        self.total_successes = 0

        # History of success rates over time (for plotting)
        # Each entry: {'timestep': int, 'episode': int, 'env_success_rates': {env_idx: rate}, 'overall_rate': float}
        self.success_rate_history = []

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose > 0:
            print(f"SuccessRateDataSaver initialized: saving to '{save_dir}' every {save_interval_episodes} episodes")

    def _on_step(self) -> bool:
        """
        Called at each step. Track episode completions and save data periodically.

        Returns:
            bool: Always returns True to continue training
        """
        # Check if any episodes finished
        if 'dones' in self.locals and 'infos' in self.locals:
            dones = self.locals['dones']
            infos = self.locals['infos']

            for env_idx, (done, info) in enumerate(zip(dones, infos)):
                if done and 'episode' in info:
                    # Initialize tracking for this environment if needed
                    if env_idx not in self.episode_successes:
                        self.episode_successes[env_idx] = []
                        self.episode_rewards[env_idx] = []
                        self.episode_lengths[env_idx] = []

                    # Check if episode was successful (goal_reached)
                    is_success = False
                    if 'goal_reached' in info:
                        is_success = bool(info['goal_reached'])

                    # Record episode data
                    self.episode_successes[env_idx].append(1 if is_success else 0)
                    self.episode_rewards[env_idx].append(float(info['episode']['r']))
                    self.episode_lengths[env_idx].append(int(info['episode']['l']))

                    # Update overall statistics
                    self.total_episodes += 1
                    if is_success:
                        self.total_successes += 1

                    # Save data at specified interval
                    if self.total_episodes % self.save_interval_episodes == 0:
                        self._save_success_rate_data()

                        if self.verbose > 0:
                            overall_rate = self.total_successes / self.total_episodes if self.total_episodes > 0 else 0
                            print(f"✓ Saved success rate data at episode {self.total_episodes} "
                                  f"(Overall: {overall_rate:.1%})")

        return True

    def _compute_success_rates(self):
        """
        Compute success rates for all environments.

        Returns:
            dict: Dictionary containing various success rate metrics
        """
        env_success_rates = {}
        env_success_rates_10ep = {}
        env_success_rates_100ep = {}

        for env_idx in sorted(self.episode_successes.keys()):
            successes = self.episode_successes[env_idx]

            # Cumulative success rate
            env_success_rates[env_idx] = np.mean(successes) if successes else 0.0

            # Rolling window success rates
            if len(successes) >= 10:
                env_success_rates_10ep[env_idx] = np.mean(successes[-10:])
            else:
                env_success_rates_10ep[env_idx] = np.mean(successes) if successes else 0.0

            if len(successes) >= 100:
                env_success_rates_100ep[env_idx] = np.mean(successes[-100:])
            else:
                env_success_rates_100ep[env_idx] = np.mean(successes) if successes else 0.0

        # Overall success rate
        overall_rate = self.total_successes / self.total_episodes if self.total_episodes > 0 else 0.0

        return {
            'env_success_rates_cumulative': env_success_rates,
            'env_success_rates_10ep': env_success_rates_10ep,
            'env_success_rates_100ep': env_success_rates_100ep,
            'overall_success_rate': overall_rate,
        }

    def _save_success_rate_data(self):
        """
        Save success rate data to JSON file.
        """
        try:
            # Compute current success rates
            rates = self._compute_success_rates()

            # Record in history
            history_entry = {
                'timestep': self.num_timesteps,
                'total_episodes': self.total_episodes,
                'total_successes': self.total_successes,
                'overall_success_rate': rates['overall_success_rate'],
                'env_success_rates_cumulative': {str(k): v for k, v in rates['env_success_rates_cumulative'].items()},
                'env_success_rates_10ep': {str(k): v for k, v in rates['env_success_rates_10ep'].items()},
                'env_success_rates_100ep': {str(k): v for k, v in rates['env_success_rates_100ep'].items()},
            }
            self.success_rate_history.append(history_entry)

            # Prepare full data structure
            data = {
                'summary': {
                    'total_episodes': self.total_episodes,
                    'total_successes': self.total_successes,
                    'overall_success_rate': rates['overall_success_rate'],
                    'num_environments': len(self.episode_successes),
                },
                'per_environment': {},
                'history': self.success_rate_history,
            }

            # Add per-environment detailed data
            for env_idx in sorted(self.episode_successes.keys()):
                env_data = {
                    'total_episodes': len(self.episode_successes[env_idx]),
                    'total_successes': sum(self.episode_successes[env_idx]),
                    'success_rate_cumulative': rates['env_success_rates_cumulative'].get(env_idx, 0.0),
                    'success_rate_10ep': rates['env_success_rates_10ep'].get(env_idx, 0.0),
                    'success_rate_100ep': rates['env_success_rates_100ep'].get(env_idx, 0.0),
                    'mean_reward': np.mean(self.episode_rewards[env_idx]) if self.episode_rewards[env_idx] else 0.0,
                    'mean_length': np.mean(self.episode_lengths[env_idx]) if self.episode_lengths[env_idx] else 0.0,
                    'episode_successes': self.episode_successes[env_idx],
                    'episode_rewards': self.episode_rewards[env_idx],
                    'episode_lengths': self.episode_lengths[env_idx],
                }
                data['per_environment'][str(env_idx)] = env_data

            # Save to JSON file
            save_path = self.save_dir / "success_rate_data.json"
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)

            if self.verbose > 1:
                print(f"  Saved to {save_path}")

        except Exception as e:
            print(f"Error saving success rate data: {e}")
            import traceback
            traceback.print_exc()

    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        if self.verbose > 0:
            print(f"SuccessRateDataSaver: Starting success rate tracking")
            print(f"  Save directory: {self.save_dir.absolute()}")
            print(f"  Save interval: every {self.save_interval_episodes} episodes")

    def _on_training_end(self) -> None:
        """Called at the end of training. Save final data."""
        # Save final data
        self._save_success_rate_data()

        if self.verbose > 0:
            print("\n" + "="*80)
            print("SUCCESS RATE DATA SAVER - FINAL SUMMARY")
            print("="*80)

            overall_rate = self.total_successes / self.total_episodes if self.total_episodes > 0 else 0
            print(f"\nOverall Success Rate: {overall_rate:.2%} ({self.total_successes}/{self.total_episodes})")

            print("\nPer-Environment Success Rates:")
            for env_idx in sorted(self.episode_successes.keys()):
                successes = self.episode_successes[env_idx]
                total = len(successes)
                success_count = sum(successes)
                rate = np.mean(successes) if successes else 0
                print(f"  Env {env_idx}: {rate:.2%} ({success_count}/{total} episodes)")

            save_path = self.save_dir / "success_rate_data.json"
            print(f"\nData saved to: {save_path.absolute()}")
            print("="*80)


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


class FreezeLearningUntilFirstEpisodeCallback(BaseCallback):
    """
    Temporarily freeze optimizer learning rates until the first completed episode.

    This prevents parameter updates during the first episode while still allowing
    data collection, replay-buffer population, and callback-based recording.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._optimizer_states = []
        self._restored = False
        self._frozen = False

    def _iter_model_optimizers(self):
        """Collect known optimizers from SB3 on-policy and off-policy models."""
        candidates = []

        for attr_name in ("policy", "actor", "critic"):
            obj = getattr(self.model, attr_name, None)
            if obj is not None and hasattr(obj, "optimizer"):
                candidates.append(getattr(obj, "optimizer"))

        for attr_name in ("ent_coef_optimizer",):
            opt = getattr(self.model, attr_name, None)
            if opt is not None:
                candidates.append(opt)

        # Some models may expose extra optimizers in lists/tuples.
        for attr_name in ("optimizers",):
            maybe_opts = getattr(self.model, attr_name, None)
            if isinstance(maybe_opts, (list, tuple)):
                candidates.extend(maybe_opts)

        seen = set()
        optimizers = []
        for opt in candidates:
            if opt is None or not hasattr(opt, "param_groups"):
                continue
            if id(opt) in seen:
                continue
            seen.add(id(opt))
            optimizers.append(opt)
        return optimizers

    def _freeze_optimizers(self):
        if self._frozen:
            return

        self._optimizer_states = []
        for opt in self._iter_model_optimizers():
            original_lrs = [group.get("lr", 0.0) for group in opt.param_groups]
            self._optimizer_states.append((opt, original_lrs))
            for group in opt.param_groups:
                group["lr"] = 0.0

        self._frozen = True
        if self.verbose > 0:
            print(
                "FreezeLearningUntilFirstEpisodeCallback: froze "
                f"{len(self._optimizer_states)} optimizer(s) until first episode completes"
            )

    def _restore_optimizers(self):
        if self._restored:
            return

        for opt, original_lrs in self._optimizer_states:
            for group, lr in zip(opt.param_groups, original_lrs):
                group["lr"] = lr

        self._restored = True
        if self.verbose > 0:
            print("FreezeLearningUntilFirstEpisodeCallback: restored optimizer learning rates")

    def _on_training_start(self) -> None:
        self._freeze_optimizers()

    def _on_step(self) -> bool:
        if self._restored:
            return True

        if 'dones' not in self.locals or 'infos' not in self.locals:
            return True

        dones = self.locals['dones']
        infos = self.locals['infos']

        for done, info in zip(dones, infos):
            if done and 'episode' in info:
                self._restore_optimizers()
                break

        return True

    def _on_training_end(self) -> None:
        # Ensure we don't leak zero LRs if training stops early.
        self._restore_optimizers()


class FreezeLearningUntilAllEnvsNEpisodesCallback(BaseCallback):
    """
    Freeze optimizer learning rates until every env completes the target episode count.

    This is a guardrail on top of rollout/learning-start thresholds so continual_training
    cannot update parameters during the initial data-collection window.
    """

    def __init__(self, target_episodes=1, verbose=0):
        super().__init__(verbose)
        self.target_episodes = int(target_episodes)
        self.completed_counts = {}
        self._optimizer_states = []
        self._restored = False
        self._frozen = False

    def _iter_model_optimizers(self):
        candidates = []

        for attr_name in ("policy", "actor", "critic"):
            obj = getattr(self.model, attr_name, None)
            if obj is not None and hasattr(obj, "optimizer"):
                candidates.append(getattr(obj, "optimizer"))

        for attr_name in ("ent_coef_optimizer",):
            opt = getattr(self.model, attr_name, None)
            if opt is not None:
                candidates.append(opt)

        for attr_name in ("optimizers",):
            maybe_opts = getattr(self.model, attr_name, None)
            if isinstance(maybe_opts, (list, tuple)):
                candidates.extend(maybe_opts)

        seen = set()
        optimizers = []
        for opt in candidates:
            if opt is None or not hasattr(opt, "param_groups"):
                continue
            if id(opt) in seen:
                continue
            seen.add(id(opt))
            optimizers.append(opt)
        return optimizers

    def _freeze_optimizers(self):
        if self._frozen:
            return

        self._optimizer_states = []
        for opt in self._iter_model_optimizers():
            original_lrs = [group.get("lr", 0.0) for group in opt.param_groups]
            self._optimizer_states.append((opt, original_lrs))
            for group in opt.param_groups:
                group["lr"] = 0.0

        self._frozen = True
        if self.verbose > 0:
            print(
                "FreezeLearningUntilAllEnvsNEpisodesCallback: froze "
                f"{len(self._optimizer_states)} optimizer(s) until all envs complete "
                f"{self.target_episodes} episodes"
            )

    def _restore_optimizers(self):
        if self._restored:
            return

        for opt, original_lrs in self._optimizer_states:
            for group, lr in zip(opt.param_groups, original_lrs):
                group["lr"] = lr

        self._restored = True
        if self.verbose > 0:
            print("FreezeLearningUntilAllEnvsNEpisodesCallback: restored optimizer learning rates")

    def _on_training_start(self) -> None:
        self._freeze_optimizers()

    def _on_step(self) -> bool:
        if self._restored:
            return True

        if 'dones' not in self.locals or 'infos' not in self.locals:
            return True

        dones = self.locals['dones']
        infos = self.locals['infos']
        num_envs = getattr(self.training_env, "num_envs", len(dones))

        for env_idx, (done, info) in enumerate(zip(dones, infos)):
            if done and 'episode' in info:
                self.completed_counts[env_idx] = self.completed_counts.get(env_idx, 0) + 1

        if all(self.completed_counts.get(env_idx, 0) >= self.target_episodes for env_idx in range(num_envs)):
            self._restore_optimizers()

        return True

    def _on_training_end(self) -> None:
        self._restore_optimizers()


class StopAfterFirstEpisodeCallback(BaseCallback):
    """
    Stop training after the first completed episode from any parallel environment.

    This is useful when trajectory/success callbacks are configured with save interval 1
    and we want a single recorded episode before terminating.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.stopped_on_env_idx = None
        self.stopped_on_timestep = None

    def _on_step(self) -> bool:
        if 'dones' not in self.locals or 'infos' not in self.locals:
            return True

        dones = self.locals['dones']
        infos = self.locals['infos']

        for env_idx, (done, info) in enumerate(zip(dones, infos)):
            if done and 'episode' in info:
                self.stopped_on_env_idx = env_idx
                self.stopped_on_timestep = self.num_timesteps
                if self.verbose > 0:
                    ep = info.get('episode', {})
                    print(
                        "StopAfterFirstEpisodeCallback: stopping training after first completed "
                        f"episode (env={env_idx}, reward={ep.get('r')}, length={ep.get('l')}, "
                        f"timesteps={self.num_timesteps})"
                    )
                return False

        return True


class StopAfterAllEnvsNEpisodesCallback(BaseCallback):
    """
    Stop training after every parallel environment has completed a target number of episodes.

    Place this callback after data-recording callbacks so episode data is saved
    before training terminates.
    """

    def __init__(self, target_episodes=1, verbose=0):
        super().__init__(verbose)
        self.target_episodes = int(target_episodes)
        self.completed_counts = {}

    def _on_step(self) -> bool:
        if 'dones' not in self.locals or 'infos' not in self.locals:
            return True

        dones = self.locals['dones']
        infos = self.locals['infos']
        num_envs = getattr(self.training_env, "num_envs", len(dones))

        for env_idx, (done, info) in enumerate(zip(dones, infos)):
            if not done or 'episode' not in info:
                continue

            self.completed_counts[env_idx] = self.completed_counts.get(env_idx, 0) + 1
            if self.verbose > 0:
                ep = info.get('episode', {})
                print(
                    "StopAfterAllEnvsNEpisodesCallback: env completed episode "
                    f"{self.completed_counts[env_idx]}/{self.target_episodes} "
                    f"(env={env_idx}, reward={ep.get('r')}, length={ep.get('l')})"
                )

        if all(self.completed_counts.get(env_idx, 0) >= self.target_episodes for env_idx in range(num_envs)):
            if self.verbose > 0:
                print(
                    "StopAfterAllEnvsNEpisodesCallback: stopping training after "
                    f"all {num_envs} envs completed {self.target_episodes} episodes "
                    f"at timesteps={self.num_timesteps}"
                )
            return False

        return True


class StopAfterTotalEpisodesCallback(BaseCallback):
    """
    Stop training after a total number of completed episodes across all parallel environments.

    This is used by reward-ablation launchers so each reward design is trained
    for the same number of completed episodes before switching to the next one.
    Place this callback after logging/data-saving callbacks so the final episode
    data is recorded before training terminates.
    """

    def __init__(self, target_episodes=300, verbose=0):
        super().__init__(verbose)
        self.target_episodes = int(target_episodes)
        self.total_episodes = 0

    def _on_step(self) -> bool:
        if 'dones' not in self.locals or 'infos' not in self.locals:
            return True

        dones = self.locals['dones']
        infos = self.locals['infos']

        for env_idx, (done, info) in enumerate(zip(dones, infos)):
            if not done or 'episode' not in info:
                continue

            self.total_episodes += 1
            if self.verbose > 0:
                ep = info.get('episode', {})
                print(
                    "StopAfterTotalEpisodesCallback: completed episode "
                    f"{self.total_episodes}/{self.target_episodes} "
                    f"(env={env_idx}, reward={ep.get('r')}, length={ep.get('l')})"
                )

        if self.total_episodes >= self.target_episodes:
            if self.verbose > 0:
                print(
                    "StopAfterTotalEpisodesCallback: stopping training after "
                    f"{self.total_episodes} completed episodes at timesteps={self.num_timesteps}"
                )
            return False

        return True


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

    def __init__(self, save_dir="trajectory_data", save_interval=10, save_images=True, verbose=0):
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
        self.save_images = save_images

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
                if self.save_images and hasattr(env, '_camera_data') and env._camera_data is not None:
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
