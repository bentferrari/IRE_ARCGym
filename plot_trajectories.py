#!/usr/bin/env python3
"""Plot 3D trajectories from test results."""

import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Find all trajectory files
test_results_dir = "/home/guanglin/arcgym/test_results"
trajectory_files = sorted(glob.glob(os.path.join(test_results_dir, "*", "first_episode_trajectory.csv")))

# Exclude specific trajectories
exclude_list = ["20260122_135050", "20260122_151107", "20260122_152147", "20260123_101600", "20260123_113915", "20260123_114105", "20260123_114242", "20260123_114325", "20260123_114639"]
trajectory_files = [f for f in trajectory_files if not any(ex in f for ex in exclude_list)]

print(f"Found {len(trajectory_files)} trajectory files")

# Calculate grid size for subplots
n_files = len(trajectory_files)
n_cols = 3
n_rows = int(np.ceil(n_files / n_cols))

# Create figure with subplots
fig = plt.figure(figsize=(16, 5 * n_rows))

# Color map for different trajectories
colors = plt.cm.tab10(range(n_files))

# Track global min/max for consistent axes
all_x, all_y, all_z = [], [], []

# First pass: load all data and find global bounds
trajectories = []
for traj_file in trajectory_files:
    df = pd.read_csv(traj_file)
    trajectories.append(df)
    all_x.extend(df['x'].tolist())
    all_y.extend(df['y'].tolist())
    all_z.extend(df['z'].tolist())

x_min, x_max = min(all_x), max(all_x)
y_min, y_max = min(all_y), max(all_y)
z_min, z_max = min(all_z), max(all_z)

for i, (traj_file, df) in enumerate(zip(trajectory_files, trajectories)):
    # Extract test name from path
    test_name = os.path.basename(os.path.dirname(traj_file))
    # Shorten name for display
    short_name = test_name.replace("ppo_test_", "")

    # Create 3D subplot for combined figure
    ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')

    # Plot trajectory
    ax.plot(df['x'], df['y'], df['z'],
            color=colors[i],
            linewidth=1.5,
            alpha=0.8)

    # Mark start point (green)
    ax.scatter(df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0],
               color='green', marker='o', s=80, label='Start')

    # Mark end point (red)
    ax.scatter(df['x'].iloc[-1], df['y'].iloc[-1], df['z'].iloc[-1],
               color='red', marker='x', s=80, label='End')

    # Set axis limits based on individual trajectory data with padding
    padding = 0.05
    x_range = df['x'].max() - df['x'].min()
    y_range = df['y'].max() - df['y'].min()
    z_range = df['z'].max() - df['z'].min()
    ax.set_xlim(df['x'].min() - padding * x_range, df['x'].max() + padding * x_range)
    ax.set_ylim(df['y'].min() - padding * y_range, df['y'].max() + padding * y_range)
    ax.set_zlim(df['z'].min() - padding * z_range, df['z'].max() + padding * z_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{short_name}\n({len(df)} points)', fontsize=10)
    ax.legend(loc='upper left', fontsize=7)

    print(f"Loaded {test_name}: {len(df)} points")

    # Create separate figure for this trajectory with smoothing
    fig_single = plt.figure(figsize=(10, 8))
    ax_single = fig_single.add_subplot(111, projection='3d')

    # Apply Gaussian smoothing to the trajectory
    sigma = 10  # Smoothing parameter
    x_smooth = gaussian_filter1d(df['x'].values, sigma=sigma)
    y_smooth = gaussian_filter1d(df['y'].values, sigma=sigma)
    z_smooth = gaussian_filter1d(df['z'].values, sigma=sigma)

    ax_single.plot(x_smooth, y_smooth, z_smooth,
                   color=colors[i],
                   linewidth=2,
                   alpha=0.8)

    ax_single.scatter(df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0],
                      color='green', marker='o', s=100, label='Start')
    ax_single.scatter(df['x'].iloc[-1], df['y'].iloc[-1], df['z'].iloc[-1],
                      color='red', marker='x', s=100, label='End')

    ax_single.set_xlim(df['x'].min() - padding * x_range, df['x'].max() + padding * x_range)
    ax_single.set_ylim(df['y'].min() - padding * y_range, df['y'].max() + padding * y_range)
    ax_single.set_zlim(df['z'].min() - padding * z_range, df['z'].max() + padding * z_range)

    ax_single.set_xlabel('X')
    ax_single.set_ylabel('Y')
    ax_single.set_zlabel('Z')
    ax_single.set_title(f'3D Trajectory: {short_name}\n({len(df)} points)', fontsize=12)
    ax_single.legend(loc='upper left', fontsize=9)

    single_path = f'/home/guanglin/arcgym/test_results/trajectory_{short_name}.png'
    fig_single.tight_layout()
    fig_single.savefig(single_path, dpi=150, bbox_inches='tight')
    plt.close(fig_single)
    print(f"  Saved individual plot to {single_path}")

plt.suptitle('3D Trajectories from Test Results', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('/home/guanglin/arcgym/test_results/trajectories_3d_subplots.png', dpi=150, bbox_inches='tight')
print(f"\nSaved combined plot to /home/guanglin/arcgym/test_results/trajectories_3d_subplots.png")
plt.show()
