"""
Plot episode rewards with shaded confidence intervals, similar to the MADDPG/MADQN plot style.

This script reads episode reward data saved by PerEnvRewardCallback and creates
a plot with smoothed mean rewards and shaded standard deviation areas.

Usage:
    python plot_rewards.py --data episode_rewards.npz --window 10 --output reward_plot.png
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy.ndimage import uniform_filter1d


def smooth_rewards(rewards, window=10):
    """
    Apply moving average smoothing to rewards.

    Args:
        rewards: Array of rewards
        window: Window size for smoothing

    Returns:
        Smoothed rewards array
    """
    if len(rewards) < window:
        return rewards
    return uniform_filter1d(rewards, size=window, mode='nearest')


def compute_rolling_statistics(episodes, rewards, window=10):
    """
    Compute rolling mean and std for plotting with shaded areas.

    Args:
        episodes: Array of episode numbers
        rewards: Array of rewards
        window: Window size for computing statistics

    Returns:
        episode_bins, mean_rewards, std_rewards
    """
    if len(episodes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Create bins for episodes
    max_episode = int(np.max(episodes))
    episode_bins = np.arange(0, max_episode + 1)

    mean_rewards = []
    std_rewards = []

    for ep in episode_bins:
        # Get rewards in window around this episode
        start_ep = max(0, ep - window // 2)
        end_ep = ep + window // 2

        mask = (episodes >= start_ep) & (episodes <= end_ep)
        window_rewards = rewards[mask]

        if len(window_rewards) > 0:
            mean_rewards.append(np.mean(window_rewards))
            std_rewards.append(np.std(window_rewards))
        else:
            # Use previous value or 0
            mean_rewards.append(mean_rewards[-1] if mean_rewards else 0)
            std_rewards.append(std_rewards[-1] if std_rewards else 0)

    return episode_bins, np.array(mean_rewards), np.array(std_rewards)


def plot_rewards_with_shading(data_file, window=10, output_file=None,
                              title="Training Rewards", figsize=(10, 6)):
    """
    Create a plot similar to the MADDPG/MADQN style with shaded variance.

    Args:
        data_file: Path to .npz file containing episode reward data
        window: Window size for smoothing (default: 10)
        output_file: Path to save the plot (default: None, shows plot instead)
        title: Plot title
        figsize: Figure size tuple
    """
    # Load data
    data = np.load(data_file)
    episodes = data['episodes']
    env_indices = data['env_indices']
    rewards = data['rewards']

    print(f"Loaded {len(episodes)} episode records")
    print(f"Environments: {np.unique(env_indices)}")
    print(f"Episode range: {np.min(episodes)} to {np.max(episodes)}")
    print(f"Reward range: {np.min(rewards):.2f} to {np.max(rewards):.2f}")

    # Create figure
    plt.figure(figsize=figsize)

    # Plot per environment with different colors
    unique_envs = np.unique(env_indices)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_envs)))

    for env_idx, color in zip(unique_envs, colors):
        # Get data for this environment
        mask = env_indices == env_idx
        env_episodes = episodes[mask]
        env_rewards = rewards[mask]

        # Sort by episode number
        sort_idx = np.argsort(env_episodes)
        env_episodes = env_episodes[sort_idx]
        env_rewards = env_rewards[sort_idx]

        # Compute rolling statistics
        ep_bins, mean_rew, std_rew = compute_rolling_statistics(
            env_episodes, env_rewards, window=window
        )

        # Plot mean line
        plt.plot(ep_bins, mean_rew, color=color, linewidth=2,
                label=f'Env {env_idx}', alpha=0.9)

        # Plot shaded standard deviation
        plt.fill_between(ep_bins,
                        mean_rew - std_rew,
                        mean_rew + std_rew,
                        color=color, alpha=0.2)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_all_environments_combined(data_file, window=10, output_file=None,
                                   title="Training Rewards (All Environments)",
                                   figsize=(10, 6)):
    """
    Create a single plot combining all environments.

    Args:
        data_file: Path to .npz file containing episode reward data
        window: Window size for smoothing (default: 10)
        output_file: Path to save the plot (default: None, shows plot instead)
        title: Plot title
        figsize: Figure size tuple
    """
    # Load data
    data = np.load(data_file)
    episodes = data['episodes']
    rewards = data['rewards']

    print(f"Loaded {len(episodes)} episode records")
    print(f"Episode range: {np.min(episodes)} to {np.max(episodes)}")
    print(f"Reward range: {np.min(rewards):.2f} to {np.max(rewards):.2f}")

    # Create figure
    plt.figure(figsize=figsize)

    # Compute rolling statistics across all environments
    ep_bins, mean_rew, std_rew = compute_rolling_statistics(
        episodes, rewards, window=window
    )

    # Plot mean line
    plt.plot(ep_bins, mean_rew, color='blue', linewidth=2,
            label='Mean Reward', alpha=0.9)

    # Plot shaded standard deviation
    plt.fill_between(ep_bins,
                    mean_rew - std_rew,
                    mean_rew + std_rew,
                    color='blue', alpha=0.3, label='± 1 std')

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot episode rewards with shaded confidence intervals')
    parser.add_argument('--data', type=str, default='episode_rewards.npz',
                       help='Path to episode rewards data file (default: episode_rewards.npz)')
    parser.add_argument('--window', type=int, default=10,
                       help='Window size for smoothing (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: None, shows plot)')
    parser.add_argument('--title', type=str, default='Training Rewards',
                       help='Plot title (default: "Training Rewards")')
    parser.add_argument('--combined', action='store_true',
                       help='Plot all environments combined instead of separately')
    parser.add_argument('--figsize', type=float, nargs=2, default=[10, 6],
                       help='Figure size as width height (default: 10 6)')

    args = parser.parse_args()

    # Check if data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file '{args.data}' not found!")
        print("Make sure to run training with PerEnvRewardCallback(save_plot_data=True)")
        return

    # Create plot
    if args.combined:
        plot_all_environments_combined(
            args.data,
            window=args.window,
            output_file=args.output,
            title=args.title,
            figsize=tuple(args.figsize)
        )
    else:
        plot_rewards_with_shading(
            args.data,
            window=args.window,
            output_file=args.output,
            title=args.title,
            figsize=tuple(args.figsize)
        )


if __name__ == '__main__':
    main()
