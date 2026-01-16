"""Utility functions for loading and visualizing saved trajectory data."""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image


def load_trajectory(trajectory_dir: str) -> Dict:
    """
    Load trajectory data from a saved episode directory.

    Args:
        trajectory_dir: Path to the episode directory containing trajectory.npz

    Returns:
        Dictionary containing:
            - 'root_states': numpy array of shape (num_steps, 13)
            - 'metadata': dictionary with episode information
            - 'images': list of image file paths (if available) - use Image.open() to load

    Example:
        >>> data = load_trajectory("trajectory_data/env_0/episode_000010")
        >>> print(f"Trajectory shape: {data['root_states'].shape}")
        >>> print(f"Episode reward: {data['metadata']['episode_reward']}")
    """
    trajectory_dir = Path(trajectory_dir)

    # Load trajectory data
    trajectory_file = trajectory_dir / "trajectory.npz"
    if not trajectory_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")

    npz_data = np.load(trajectory_file, allow_pickle=True)
    root_states = npz_data['root_states']

    # Load metadata
    metadata_str = str(npz_data['metadata'])
    metadata = json.loads(metadata_str)

    # Load images if available
    # Note: Images are loaded lazily to avoid "too many open files" error
    # Store file paths instead of loaded images
    images = []
    images_dir = trajectory_dir / "images"
    if images_dir.exists():
        image_files = sorted(images_dir.glob("frame_*.png"))
        # Store just the paths, we'll load them when needed
        images = [str(f) for f in image_files]

    return {
        'root_states': root_states,
        'metadata': metadata,
        'images': images,
        'trajectory_dir': str(trajectory_dir)
    }


def load_metadata(trajectory_dir: str) -> Dict:
    """
    Load only the metadata for a trajectory (faster than loading full data).

    Args:
        trajectory_dir: Path to the episode directory

    Returns:
        Dictionary with episode metadata
    """
    metadata_file = Path(trajectory_dir) / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, 'r') as f:
        return json.load(f)


def find_episodes(save_dir: str, env_idx: Optional[int] = None,
                  successful_only: bool = False) -> List[str]:
    """
    Find all saved episode directories.

    Args:
        save_dir: Root directory where trajectory data is saved
        env_idx: If specified, only return episodes for this environment
        successful_only: If True, only return episodes where goal was reached

    Returns:
        List of episode directory paths

    Example:
        >>> episodes = find_episodes("trajectory_data", env_idx=0, successful_only=True)
        >>> print(f"Found {len(episodes)} successful episodes for env 0")
    """
    save_dir = Path(save_dir)
    episode_dirs = []

    # Find environment directories
    if env_idx is not None:
        env_dirs = [save_dir / f"env_{env_idx}"]
    else:
        env_dirs = sorted(save_dir.glob("env_*"))

    # Find episode directories
    for env_dir in env_dirs:
        if env_dir.is_dir():
            for episode_dir in sorted(env_dir.glob("episode_*")):
                if successful_only:
                    try:
                        metadata = load_metadata(episode_dir)
                        if metadata.get('goal_reached', False):
                            episode_dirs.append(str(episode_dir))
                    except:
                        continue
                else:
                    episode_dirs.append(str(episode_dir))

    return episode_dirs


def plot_trajectory_3d(root_states: np.ndarray, title: str = "3D Trajectory",
                       figsize: Tuple[int, int] = (14.52, 9.68)) -> plt.Figure:
    """
    Plot 3D trajectory of the capsule from root_states.

    Args:
        root_states: Array of shape (num_steps, 13) containing root states
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure object

    Example:
        >>> data = load_trajectory("trajectory_data/env_0/episode_000010")
        >>> fig = plot_trajectory_3d(data['root_states'])
        >>> plt.show()
    """
    positions = root_states[:, :3]  # Extract x, y, z positions

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'b-', linewidth=2, label='Trajectory')

    # Mark start and end
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
               c='green', s=100, marker='o', label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
               c='red', s=100, marker='X', label='End')

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    return fig


def plot_trajectory_3d_standalone(trajectory_dir: str, save_plot: bool = False,
                                   max_steps: int = 4000,
                                   figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot standalone 3D trajectory of the capsule.

    Args:
        trajectory_dir: Path to the episode directory
        save_plot: If True, save the plot as a PNG file
        max_steps: Maximum number of steps to plot (default: 4000)
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    data = load_trajectory(trajectory_dir)
    root_states = data['root_states']
    metadata = data['metadata']

    # Limit trajectory to max_steps
    total_steps = root_states.shape[0]
    if total_steps > max_steps:
        root_states = root_states[:max_steps]
        print(f"  Note: Trajectory truncated from {total_steps} to {max_steps} steps for plotting")

    positions = root_states[:, :3]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
               c='green', s=100, marker='o', label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
               c='red', s=100, marker='X', label='End')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    ax.grid(True)

    success_str = "SUCCESS" if metadata.get('goal_reached', False) else "INCOMPLETE"
    reward = metadata.get('episode_reward', 'N/A')
    fig.suptitle(
        f"3D Trajectory - Episode {metadata['episode_num']} (Env {metadata['env_idx']}) - {success_str}\n"
        f"Reward: {reward:.2f}, Steps: {total_steps}",
        fontsize=12, fontweight='bold'
    )

    if save_plot:
        plot_path = Path(trajectory_dir) / "trajectory_3d.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")

    return fig


def plot_position_over_time(trajectory_dir: str, save_plot: bool = False,
                            max_steps: int = 4000,
                            figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot position (x, y, z) over time.

    Args:
        trajectory_dir: Path to the episode directory
        save_plot: If True, save the plot as a PNG file
        max_steps: Maximum number of steps to plot (default: 4000)
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    data = load_trajectory(trajectory_dir)
    root_states = data['root_states']
    metadata = data['metadata']

    # Limit trajectory to max_steps
    total_steps = root_states.shape[0]
    if total_steps > max_steps:
        root_states = root_states[:max_steps]
        print(f"  Note: Trajectory truncated from {total_steps} to {max_steps} steps for plotting")

    num_steps = root_states.shape[0]
    time_steps = np.arange(num_steps)
    positions = root_states[:, :3]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(time_steps, positions[:, 0], label='X', linewidth=2)
    ax.plot(time_steps, positions[:, 1], label='Y', linewidth=2)
    ax.plot(time_steps, positions[:, 2], label='Z', linewidth=2)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Position (m)')
    ax.legend()
    ax.grid(True)

    success_str = "SUCCESS" if metadata.get('goal_reached', False) else "INCOMPLETE"
    reward = metadata.get('episode_reward', 'N/A')
    fig.suptitle(
        f"Position over Time - Episode {metadata['episode_num']} (Env {metadata['env_idx']}) - {success_str}\n"
        f"Reward: {reward:.2f}, Steps: {total_steps}",
        fontsize=12, fontweight='bold'
    )

    if save_plot:
        plot_path = Path(trajectory_dir) / "position_over_time.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")

    return fig


def plot_trajectory_analysis(trajectory_dir: str, save_plot: bool = False, num_frames: int = 6, max_steps: int = 4000) -> plt.Figure:
    """
    Create a comprehensive analysis plot for a trajectory.

    Includes:
    - 3D trajectory plot
    - Position over time (x, y, z)
    - Sample endoscope images from the trajectory

    Args:
        trajectory_dir: Path to the episode directory
        save_plot: If True, save the plot as a PNG file in the trajectory directory
        num_frames: Number of endoscope frames to display (default: 6)
        max_steps: Maximum number of steps to plot (default: 4000)

    Returns:
        Matplotlib figure object

    Example:
        >>> fig = plot_trajectory_analysis("trajectory_data/env_0/episode_000010", save_plot=True, num_frames=9)
        >>> plt.show()
    """
    data = load_trajectory(trajectory_dir)
    root_states = data['root_states']
    metadata = data['metadata']
    images = data['images']

    # Limit trajectory to max_steps
    total_steps = root_states.shape[0]
    if total_steps > max_steps:
        root_states = root_states[:max_steps]
        # Also adjust images to match the truncated trajectory
        if images:
            images = images[:max_steps]
        print(f"  Note: Trajectory truncated from {total_steps} to {max_steps} steps for plotting")

    num_steps = root_states.shape[0]
    time_steps = np.arange(num_steps)

    # Extract positions
    positions = root_states[:, :3]

    # Calculate grid layout for frames
    frames_per_row = 6
    num_frame_rows = 1

    # Create figure with subplots
    # Layout: 1 row for trajectory plots + 1 row for frames
    total_rows = 2
    total_cols = 6
    fig = plt.figure(figsize=(12.96, 9.5))
    height_ratios = [2.6, 0.6]
    gs = fig.add_gridspec(
        total_rows,
        total_cols,
        hspace=0.25,
        wspace=0.8,
        left=0.04,
        right=0.99,
        top=0.90,
        bottom=0.06,
        height_ratios=height_ratios
    )

    # Title
    success_str = "SUCCESS" if metadata.get('goal_reached', False) else "INCOMPLETE"
    reward = metadata.get('episode_reward', 'N/A')
    truncated_str = f" (showing first {num_steps})" if total_steps > max_steps else ""
    fig.suptitle(
        f"Episode {metadata['episode_num']} (Env {metadata['env_idx']}) - {success_str}\n"
        f"Reward: {reward:.2f}, Steps: {total_steps}{truncated_str}",
        fontsize=14,
        fontweight='bold',
        y=0.97
    )

    # 3D Trajectory
    ax1 = fig.add_subplot(gs[0, 0:2], projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                c='green', s=100, marker='o', label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                c='red', s=100, marker='X', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('')
    ax1.legend()
    ax1.grid(True)

    # Position over time
    ax2 = fig.add_subplot(gs[0, 2:5])
    ax2.plot(time_steps, positions[:, 0], label='X', linewidth=2)
    ax2.plot(time_steps, positions[:, 1], label='Y', linewidth=2)
    ax2.plot(time_steps, positions[:, 2], label='Z', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('')
    ax2.legend()
    ax2.grid(True)

    # Align the top edges so titles land on the same horizontal line.
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    top = max(pos1.y1, pos2.y1)
    ax1.set_position([pos1.x0, pos1.y0, pos1.width, top - pos1.y0])
    ax2.set_position([pos2.x0, pos2.y0, pos2.width, top - pos2.y0])
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    title_y = min(0.94, max(pos1.y1, pos2.y1) + 0.01)
    fig.text(pos1.x0 + pos1.width / 2, title_y, '3D Trajectory', ha='center', va='bottom')
    fig.text(pos2.x0 + pos2.width / 2, title_y, 'Position over Time', ha='center', va='bottom')

    # Sample endoscope frames
    if images:
        num_sample_images = min(num_frames, len(images))
        sample_indices = np.linspace(0, len(images) - 1, num_sample_images, dtype=int)

        for i, idx in enumerate(sample_indices):
            row = 1
            col = i % frames_per_row
            ax = fig.add_subplot(gs[row, col])
            # Load image on-demand and close it after use
            img_path = images[idx]
            with Image.open(img_path) as img:
                # Convert to numpy array so we can close the file
                img_array = np.array(img)
            ax.imshow(img_array)
            ax.set_title(f'Frame {idx}/{len(images)-1}')
            ax.axis('off')

    if save_plot:
        plot_path = Path(trajectory_dir) / "trajectory_analysis.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")

    return fig


def export_trajectory_to_csv(trajectory_dir: str, output_file: Optional[str] = None):
    """
    Export trajectory data to CSV format for external analysis.

    Args:
        trajectory_dir: Path to the episode directory
        output_file: Output CSV file path. If None, saves as 'trajectory.csv' in the episode directory

    Example:
        >>> export_trajectory_to_csv("trajectory_data/env_0/episode_000010")
    """
    import csv

    data = load_trajectory(trajectory_dir)
    root_states = data['root_states']

    if output_file is None:
        output_file = Path(trajectory_dir) / "trajectory.csv"

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'pos_x', 'pos_y', 'pos_z',
                        'quat_w', 'quat_x', 'quat_y', 'quat_z',
                        'lin_vel_x', 'lin_vel_y', 'lin_vel_z',
                        'ang_vel_x', 'ang_vel_y', 'ang_vel_z'])

        for step, state in enumerate(root_states):
            writer.writerow([step] + state.tolist())

    print(f"Trajectory exported to: {output_file}")
