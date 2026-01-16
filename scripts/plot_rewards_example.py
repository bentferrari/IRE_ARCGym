"""
Example script demonstrating how to create reward plots with shaded areas.

This shows how to:
1. Load saved episode reward data
2. Create plots with different styles
3. Customize the visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def demo_plot_from_data(data_file='episode_rewards.npz'):
    """
    Simple example showing how to create the plot.
    """
    # Load the data saved by PerEnvRewardCallback
    data = np.load(data_file)
    episodes = data['episodes']
    env_indices = data['env_indices']
    rewards = data['rewards']

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique environments
    unique_envs = np.unique(env_indices)
    colors = ['blue', 'orange', 'green', 'red', 'purple']

    # Plot each environment
    for i, env_idx in enumerate(unique_envs):
        # Filter data for this environment
        mask = env_indices == env_idx
        env_eps = episodes[mask]
        env_rews = rewards[mask]

        # Sort by episode
        sort_idx = np.argsort(env_eps)
        env_eps = env_eps[sort_idx]
        env_rews = env_rews[sort_idx]

        # Compute rolling mean and std (window=10 episodes)
        window = 10
        max_ep = int(np.max(env_eps))

        mean_rewards = []
        std_rewards = []
        episode_bins = []

        for ep in range(0, max_ep + 1):
            # Get rewards in window around this episode
            start = max(0, ep - window // 2)
            end = ep + window // 2
            window_mask = (env_eps >= start) & (env_eps <= end)
            window_rews = env_rews[window_mask]

            if len(window_rews) > 0:
                episode_bins.append(ep)
                mean_rewards.append(np.mean(window_rews))
                std_rewards.append(np.std(window_rews))

        episode_bins = np.array(episode_bins)
        mean_rewards = np.array(mean_rewards)
        std_rewards = np.array(std_rewards)

        # Plot mean line
        color = colors[i % len(colors)]
        ax.plot(episode_bins, mean_rewards, color=color,
               linewidth=2, label=f'Env {env_idx}', alpha=0.9)

        # Plot shaded area (mean ± std)
        ax.fill_between(episode_bins,
                       mean_rewards - std_rewards,
                       mean_rewards + std_rewards,
                       color=color, alpha=0.2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Training Rewards per Environment', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reward_plot.png', dpi=150, bbox_inches='tight')
    print("Saved plot to reward_plot.png")
    plt.show()


def demo_multiple_runs():
    """
    Example showing how to plot multiple training runs (like MADDPG vs MADQN).
    """
    # This assumes you have multiple .npz files from different runs
    runs = {
        'Run 1': 'episode_rewards_run1.npz',
        'Run 2': 'episode_rewards_run2.npz',
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue', 'orange', 'green', 'red']

    for i, (run_name, data_file) in enumerate(runs.items()):
        if not Path(data_file).exists():
            print(f"Skipping {run_name}: file not found")
            continue

        # Load data
        data = np.load(data_file)
        episodes = data['episodes']
        rewards = data['rewards']

        # Compute statistics across all environments
        window = 10
        max_ep = int(np.max(episodes))

        mean_rewards = []
        std_rewards = []
        episode_bins = []

        for ep in range(0, max_ep + 1):
            start = max(0, ep - window // 2)
            end = ep + window // 2
            mask = (episodes >= start) & (episodes <= end)
            window_rews = rewards[mask]

            if len(window_rews) > 0:
                episode_bins.append(ep)
                mean_rewards.append(np.mean(window_rews))
                std_rewards.append(np.std(window_rews))

        episode_bins = np.array(episode_bins)
        mean_rewards = np.array(mean_rewards)
        std_rewards = np.array(std_rewards)

        # Plot
        color = colors[i % len(colors)]
        ax.plot(episode_bins, mean_rewards, color=color,
               linewidth=2, label=run_name, alpha=0.9)
        ax.fill_between(episode_bins,
                       mean_rewards - std_rewards,
                       mean_rewards + std_rewards,
                       color=color, alpha=0.2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Comparison of Training Runs', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=150, bbox_inches='tight')
    print("Saved comparison plot to comparison_plot.png")
    plt.show()


if __name__ == '__main__':
    # Run the demo
    data_file = 'episode_rewards.npz'

    if Path(data_file).exists():
        demo_plot_from_data(data_file)
    else:
        print(f"Data file '{data_file}' not found!")
        print("\nTo generate the data:")
        print("1. Train with: callback = PerEnvRewardCallback(save_plot_data=True)")
        print("2. After training, run this script")
        print("\nOr use the full plotting script:")
        print("  python plot_rewards.py --data episode_rewards.npz --window 10 --output plot.png")
