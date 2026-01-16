#!/usr/bin/env python3
"""
Example script to load and analyze saved trajectory data.

Usage:
    # Analyze a specific episode
    python analyze_trajectories.py --episode trajectory_data/ppo_arc_20231215_120000/env_0/episode_000010

    # Find and analyze all successful episodes
    python analyze_trajectories.py --save-dir trajectory_data/ppo_arc_20231215_120000 --successful-only

    # List all episodes for environment 0
    python analyze_trajectories.py --save-dir trajectory_data/ppo_arc_20231215_120000 --env 0 --list-only
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import arcgym modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcgym.utils.trajectory_loader import (
    load_trajectory,
    load_metadata,
    find_episodes,
    plot_trajectory_analysis,
    plot_trajectory_3d,
    export_trajectory_to_csv
)

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Analyze saved trajectory data")
    parser.add_argument("--episode", type=str, help="Path to specific episode directory to analyze")
    parser.add_argument("--save-dir", type=str, help="Root directory containing trajectory data")
    parser.add_argument("--env", type=int, help="Environment index to analyze")
    parser.add_argument("--successful-only", action="store_true", help="Only analyze successful episodes")
    parser.add_argument("--list-only", action="store_true", help="Only list episodes, don't plot")
    parser.add_argument("--save-plots", action="store_true", help="Save plots as PNG files")
    parser.add_argument("--export-csv", action="store_true", help="Export trajectories to CSV")
    parser.add_argument("--max-episodes", type=int, default=5, help="Maximum number of episodes to analyze")
    parser.add_argument("--num-frames", type=int, default=6, help="Number of endoscope frames to display (default: 6)")
    parser.add_argument("--max-steps", type=int, default=4000, help="Maximum number of steps to plot (default: 4000)")

    args = parser.parse_args()

    if args.episode:
        # Analyze a specific episode
        print(f"Loading trajectory from: {args.episode}")
        analyze_episode(args.episode, save_plot=args.save_plots, export_csv=args.export_csv,
                       num_frames=args.num_frames, max_steps=args.max_steps)

    elif args.save_dir:
        # Find and analyze multiple episodes
        print(f"Searching for episodes in: {args.save_dir}")
        episodes = find_episodes(
            args.save_dir,
            env_idx=args.env,
            successful_only=args.successful_only
        )

        if not episodes:
            print("No episodes found!")
            return

        print(f"Found {len(episodes)} episodes")

        if args.list_only:
            # Just list the episodes with metadata
            for ep_dir in episodes:
                metadata = load_metadata(ep_dir)
                success_str = "✓" if metadata.get('goal_reached', False) else "✗"
                reward = metadata.get('episode_reward', 'N/A')
                steps = metadata.get('num_steps', 'N/A')
                print(f"{success_str} {ep_dir}")
                print(f"   Reward: {reward:.2f}, Steps: {steps}")
        else:
            # Analyze the episodes
            num_to_analyze = min(len(episodes), args.max_episodes)
            print(f"Analyzing {num_to_analyze} episodes...")

            for i, ep_dir in enumerate(episodes[:num_to_analyze]):
                print(f"\n[{i+1}/{num_to_analyze}] Analyzing: {ep_dir}")
                analyze_episode(ep_dir, save_plot=args.save_plots, export_csv=args.export_csv,
                               num_frames=args.num_frames, max_steps=args.max_steps)

            plt.show()  # Show all plots at once

    else:
        parser.print_help()
        print("\nError: Must specify either --episode or --save-dir")


def analyze_episode(episode_dir, save_plot=False, export_csv=False, num_frames=6, max_steps=4000):
    """Analyze a single episode."""
    try:
        # Load metadata first
        metadata = load_metadata(episode_dir)
        print(f"  Episode {metadata['episode_num']} (Env {metadata['env_idx']})")
        print(f"  Reward: {metadata.get('episode_reward', 'N/A'):.2f}")
        print(f"  Steps: {metadata.get('num_steps', 'N/A')}")
        print(f"  Goal reached: {metadata.get('goal_reached', False)}")
        print(f"  Colon stress: {metadata.get('colon_stress', 'N/A')}")

        # Create analysis plot
        fig = plot_trajectory_analysis(episode_dir, save_plot=save_plot, num_frames=num_frames, max_steps=max_steps)

        # Export to CSV if requested
        if export_csv:
            export_trajectory_to_csv(episode_dir)

    except Exception as e:
        print(f"  Error analyzing episode: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
