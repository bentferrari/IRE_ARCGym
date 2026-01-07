from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import cv2
import torch
import numpy as np

# from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_configs = {
#     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
#     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
#     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
#     'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
# }

# encoder = 'vits' # or 'vits', 'vitb', 'vitg'

# depth_model = DepthAnythingV2(**model_configs[encoder])
# depth_model.load_state_dict(torch.load(f'/home/guanglin/arcgym/src/DepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
# depth_model = depth_model.to(DEVICE).eval()


def point_to_polyline_distance(p: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """
    Compute the shortest distance between point p and a polyline
    defined by consecutive vertices.

    Args:
        p (torch.Tensor): Point tensor of shape (D,)
        vertices (torch.Tensor): Tensor of shape (N, D) containing
                                 N vertices (v0, v1, ..., vN-1)

    Returns:
        torch.Tensor: Scalar tensor containing the shortest distance
    """
    # p shape: (D,)
    # vertices shape: (N, D)

    # Build consecutive segments: (vi, vi+1)
    seg_start = vertices[:-1]  # shape (N-1, D)
    seg_end = vertices[1:]     # shape (N-1, D)

    seg_vec = seg_end - seg_start               # (N-1, D)
    ap_vec = p.unsqueeze(0) - seg_start         # (N-1, D)

    # Squared length of each segment
    seg_len_sq = torch.sum(seg_vec * seg_vec, dim=1)  # (N-1,)

    # Handle degenerate segments (length = 0)
    # Avoid division by zero: set t=0 when degenerate
    t = torch.where(
        seg_len_sq > 0,
        torch.sum(ap_vec * seg_vec, dim=1) / seg_len_sq,
        torch.zeros_like(seg_len_sq),
    )

    # Clamp t to [0,1] for projection within segment
    t = torch.clamp(t, 0.0, 1.0)

    # Closest point on each segment
    closest_points = seg_start + t.unsqueeze(1) * seg_vec  # (N-1, D)

    # Distance from p to each closest point
    dists = torch.norm(p.unsqueeze(0) - closest_points, dim=1)  # (N-1,)

    # Return minimum distance
    i = torch.argmin(dists)
    return dists[i], i, t[i]

class RewardFunction(ABC):
    @abstractmethod
    def __call__(
        self,
        action: Any,
        current_states: Dict,
        previous_states: Dict,
    ) -> float:
        pass

class TestRewardAction(RewardFunction):
    def __init__(self, reward_scale, **kwargs):
        self.reward_scale = reward_scale
    
    def reset(self, *args, **kwargs):
        pass
    
    def __call__(self, action, current_states, _previous_states):
        return action[:, 0] * self.reward_scale

class TestRewardPosition(RewardFunction):
    def __init__(self, reward_scale, **kwargs):
        self.reward_scale = reward_scale
        self.initial_positions_x = None
    
    def reset(self, *args, **kwargs):
        pass
    
    def __call__(self, action, current_states, _previous_states):
        robot_positions_x = current_states["robot_positions"][:, 0]
        if self.initial_positions_x is None:
            self.initial_positions_x = robot_positions_x.clone()
        difference = (robot_positions_x - self.initial_positions_x) * self.reward_scale * 100.0
        self.initial_positions_x = robot_positions_x.clone()
        return difference

class GoalReward(RewardFunction):
    def __init__(self, running_penalty, goal_reward, eps, reward_scale, **kwargs):
        self.goals = None
        self.running_penalty = running_penalty
        self.goal_reward = goal_reward
        self.eps = eps
        self.reward_scale = reward_scale

    def reset(self, initial_positions, goals):
        self.initial_positions = initial_positions
        self.goals = goals

    def __call__(self, action, current_states, _previous_states):
        robot_positions = current_states["robot_positions"]
        dist = torch.norm(robot_positions - self.goals[:,2,:], dim=1)

        dist_mask = dist < self.eps
        rewards = torch.ones_like(dist) * self.running_penalty
        rewards[dist_mask] = self.goal_reward

        #print(robot_positions, dist, dist_mask, rewards)

        return rewards * self.reward_scale

class PointToPointLineDistanceReward(RewardFunction):
    def __init__(self, eps, reward_scale, **kwargs):
        self.goal = None
        self.initial_positions = None
        self.initial_distances = None
        self.eps = eps
        self.reward_scale = reward_scale
    
    def reset(self, initial_positions, goals):
        self.initial_positions = initial_positions.unsqueeze(1)
        self.goals = goals
        self.initial_distances = torch.norm(self.initial_positions - self.goals, dim=2)

    def __call__(self, action, current_states, _previous_states):
        robot_positions = current_states["robot_positions"]

        rewards = []

        for position, goal_positions in zip(robot_positions, self.goals):
            dist, segment_idx, t = point_to_polyline_distance(position, goal_positions)
            #print(dist, segment_idx, t, 15.0 * t * (segment_idx + 1), - dist**2, 15.0 * t * (segment_idx + 1) - dist**2)
            rewards.append(15.0 * t * (segment_idx + 1) - dist**2)

        return torch.stack(rewards) * self.reward_scale

class DepthDistanceReward(RewardFunction):
    def __init__(self, eps, reward_scale, **kwargs):
        self.reward_scale = reward_scale
        self.goal_reached_per_env = None  # Will store per-environment goal status

    def reset(self, initial_positions, goals):
        self.initial_positions = initial_positions
        self.goals = goals
        # Reset goal_reached status for all environments
        num_envs = len(initial_positions)
        self.goal_reached_per_env = torch.zeros(num_envs, dtype=torch.bool, device=initial_positions.device)

    def __call__(self, action, current_states, _previous_states):
        # obs = current_states["obs"] why cannot access obs?
        # print("obs shape:", obs)
        robot_positions = current_states["robot_positions"]
        # print("robot_pos:", robot_positions)
        depth_images = current_states["depth"]
        goals = self.goals
        entry_poss = self.initial_positions
        rewards = []

        # Reset goal_reached status for this step
        if self.goal_reached_per_env is None:
            num_envs = len(robot_positions)
            self.goal_reached_per_env = torch.zeros(num_envs, dtype=torch.bool, device=robot_positions.device)
        else:
            self.goal_reached_per_env.fill_(False)

        for i, (robot_position, depth_img, goal, entry_pos) in enumerate(zip(robot_positions, depth_images, goals, entry_poss)):
            # print("robot_positions:", robot_positions)
            print("goals:", self.goals)
            d2target = torch.norm(goal - robot_position)
            # print("entry_pos:", entry_pos)
            d_max = torch.norm(goal - entry_pos)
            print("d2target:", d2target, "d_max:", d_max)

            ###### operate depth model ######
            #obs_depth = np.transpose(obs, (1,2,0))
            depth_img = depth_img[:, :, 0].cpu().numpy()
            #depth_img = depth_model.infer_image(obs)
            ### Locate the deepest point (max depth value)
            max_depth_idx = np.unravel_index(np.argmin(depth_img), depth_img.shape)  # (row, col)
            #print(len(max_depth_idx), depth_img.shape)
            max_depth_row, max_depth_col = max_depth_idx
            #print("Deepest point (row, col):", max_depth_row, max_depth_col)
            ###### operate darkness ######
            #obs_gray = cv2.cvtColor(obs_depth, cv2.COLOR_RGB2GRAY)
            #print(np.min(depth_img), np.max(depth_img))
            max_val = np.max(depth_img) # depth_img is not normalized
            _, binary_image = cv2.threshold(depth_img, 0.43*max_val, 1.0*max_val, cv2.THRESH_BINARY)
            mask = (binary_image == 0).astype(np.uint8)
            #print("dark ratio:", np.sum(mask) / mask.size)
            #from PIL import Image
            #Image.fromarray(np.clip(depth_img*255, 0, 255).astype(np.uint8)).save("depth_img.png")
    
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                cx_dark, cy_dark = centroids[largest_label]
                #print("center of dark region (x, y):", cx_dark, cy_dark)
                r_a = 1 - (((max_depth_row - 64)**2 + ((max_depth_col - 64)**2))**0.5) / 64
                print("reward_a",r_a)
                r_b = 1 - (((cx_dark - 64)**2 + ((cy_dark - 64)**2))**0.5) / 64
                r_c = 1 - d2target/d_max
                print("reward_c",r_c)
                black_pixels =  np.sum(binary_image == 0)
                total_pixels = binary_image.size
                dark_ratio = black_pixels / total_pixels
                r_d = -3 / dark_ratio
                reward = 0.5*(r_a + r_c)
            else:
                ### No dark region found, punish the robot
                print("No dark region found!")
                dark_ratio = 0
                r_c = 1 - d2target/d_max
                reward = 0.5*r_c
            # slightly punish hitting wall
            if torch.abs(torch.tensor(np.max(depth_img) - 1.0)) < 0.001:
                reward = -1
            if dark_ratio < 0.1:
                reward = -1
            if d2target < 0.002:
                reward = 1
                self.goal_reached_per_env[i] = True
                print(f"Goal reached in environment {i}!")
            print("reward", reward)
            rewards.append(torch.tensor(reward,
                                 dtype=torch.float32,
                                 device='cuda' if torch.cuda.is_available() else 'cpu'))
            #rewards.append(reward)

        return torch.stack(rewards)

class FinalReward(RewardFunction):
    def __init__(self, eps, reward_scale, center_weight=0.3, goal_weight=0.4, obstruction_weight=0.3, **kwargs):
        """
        Final reward function combining multiple penalty components.

        Args:
            eps: Epsilon threshold for goal reaching
            reward_scale: Overall reward scaling factor
            center_weight: Weight for lumen center deviation penalty
            goal_weight: Weight for goal distance penalty
            obstruction_weight: Weight for obstruction penalty
        """
        self.reward_scale = reward_scale
        self.eps = eps
        self.center_weight = center_weight
        self.goal_weight = goal_weight
        self.obstruction_weight = obstruction_weight
        self.goal_reached_per_env = None

    def reset(self, initial_positions, goals):
        self.initial_positions = initial_positions
        self.goals = goals
        # Reset goal_reached status for all environments
        num_envs = len(initial_positions)
        self.goal_reached_per_env = torch.zeros(num_envs, dtype=torch.bool, device=initial_positions.device)

    def __call__(self, action, current_states, _previous_states):
        robot_positions = current_states["robot_positions"]
        depth_images = current_states["depth"]
        goals = self.goals
        entry_poss = self.initial_positions
        rewards = []

        # Reset goal_reached status for this step
        if self.goal_reached_per_env is None:
            num_envs = len(robot_positions)
            self.goal_reached_per_env = torch.zeros(num_envs, dtype=torch.bool, device=robot_positions.device)
        else:
            self.goal_reached_per_env.fill_(False)

        for i, (robot_position, depth_img, goal, entry_pos) in enumerate(zip(robot_positions, depth_images, goals, entry_poss)):
            # Calculate distance to goal
            d2target = torch.norm(goal - robot_position)
            d_max = torch.norm(goal - entry_pos)

            # Process depth image
            depth_img_np = depth_img[:, :, 0].cpu().numpy()
            img_height, img_width = depth_img_np.shape
            img_center_row, img_center_col = img_height / 2, img_width / 2

            # Depth statistics - KEY CRITERIA for lumen quality:
            # 1. max_depth: indicates if facing lumen or wall
            # 2. depth_variance & high_depth_ratio: indicates quality of lumen view
            max_depth = np.max(depth_img_np)
            min_depth = np.min(depth_img_np)
            mean_depth = np.mean(depth_img_np)
            depth_variance = np.var(depth_img_np)

            # Find the deepest point location (highest depth value = farthest point)
            max_depth_idx = np.unravel_index(np.argmax(depth_img_np), depth_img_np.shape)
            max_depth_row, max_depth_col = max_depth_idx  # (row, col)

            # Calculate ratio of pixels with "good depth" (indicates wide open lumen vs narrow view)
            GOOD_DEPTH_THRESHOLD = 0.12  # Pixels above this are considered "good depth"
            high_depth_pixels = np.sum(depth_img_np > GOOD_DEPTH_THRESHOLD)
            total_pixels = depth_img_np.size
            high_depth_ratio = high_depth_pixels / total_pixels

            # Calculate max possible distance from center (for normalization)
            max_center_distance = np.sqrt(img_center_row**2 + img_center_col**2)

            # Distance from deepest point to image center
            deepest_point_deviation = np.sqrt((max_depth_row - img_center_row)**2 +
                                             (max_depth_col - img_center_col)**2)
            deepest_point_normalized = deepest_point_deviation / max_center_distance

            # print(f"Depth stats - min: {min_depth:.3f}, max: {max_depth:.3f}, mean: {mean_depth:.3f}, "
            #       f"variance: {depth_variance:.6f}, high_depth_ratio: {high_depth_ratio:.3f}, "
            #       f"deepest_deviation: {deepest_point_normalized:.3f}")

            # Calculate individual reward components
            reward_components = {}

            # 1. Lumen alignment: reward when deepest point (farthest) is centered
            # When facing lumen center, the deepest point should be in the center of the image
            reward_components['center_alignment'] = 1.0 - deepest_point_normalized

            # 2. Lumen visibility quality: combines max_depth AND high_depth_ratio
            # Good lumen view requires BOTH deep point AND wide open view
            LUMEN_MIN_DEPTH = 0.20  # Minimum max depth for good lumen
            WALL_MAX_DEPTH = 0.10   # Maximum depth indicating wall
            MIN_HIGH_DEPTH_RATIO = 0.40  # Minimum ratio of "good depth" pixels for quality view

            # First check max_depth (is lumen visible at all?)
            if max_depth < WALL_MAX_DEPTH:
                # Very low max depth - definitely facing wall (BAD)
                lumen_visibility_penalty = -1.0
                reward_components['center_alignment'] = -1.0  # Override center alignment
            elif max_depth < LUMEN_MIN_DEPTH:
                # Low max depth - partially obstructed
                base_penalty = -0.5 * (LUMEN_MIN_DEPTH - max_depth) / (LUMEN_MIN_DEPTH - WALL_MAX_DEPTH)

                # Further penalize if high_depth_ratio is low (narrow view)
                if high_depth_ratio < MIN_HIGH_DEPTH_RATIO:
                    ratio_penalty = -0.3 * (MIN_HIGH_DEPTH_RATIO - high_depth_ratio) / MIN_HIGH_DEPTH_RATIO
                    lumen_visibility_penalty = base_penalty + ratio_penalty
                else:
                    lumen_visibility_penalty = base_penalty
            else:
                # Good max depth - but check if view is wide open or narrow
                if high_depth_ratio < MIN_HIGH_DEPTH_RATIO:
                    # Narrow view - can see lumen but not well positioned (YOUR CASE)
                    # Penalize proportionally to how narrow the view is
                    lumen_visibility_penalty = -0.4 * (MIN_HIGH_DEPTH_RATIO - high_depth_ratio) / MIN_HIGH_DEPTH_RATIO
                elif high_depth_ratio < 0.25:
                    # Decent view but not optimal
                    lumen_visibility_penalty = 0.1
                else:
                    # Wide open lumen view - excellent! (GOOD)
                    lumen_visibility_penalty = 0.2

            reward_components['lumen_visibility'] = lumen_visibility_penalty

            # print(f"max_depth: {max_depth:.3f}, high_depth_ratio: {high_depth_ratio:.3f}, "
            #       f"lumen_visibility: {lumen_visibility_penalty:.3f}")

            # 3. Penalty for deviation from robot position to goal position
            reward_components['goal_progress'] = 1.0 - (d2target / d_max).item()

            # Combine reward components with weights
            total_reward = (self.center_weight * reward_components['center_alignment'] +
                          self.goal_weight * reward_components['goal_progress'] +
                          self.obstruction_weight * reward_components['lumen_visibility'])

            # Special conditions override the weighted sum
            # Hitting wall - severe penalty (depth = 1.0 means collision)
            if torch.abs(torch.tensor(max_depth - 1.0)) < 0.001:
                total_reward = -1.0

            # Goal reached - maximum reward
            if d2target < self.eps:
                total_reward = 1.0
                self.goal_reached_per_env[i] = True
                print(f"Goal reached in environment {i}!")

            # Normalize the reward to be in range [-1, 1]
            # The weighted combination naturally stays in this range for normal operation
            # Special conditions are already normalized
            total_reward = np.clip(total_reward, -1.0, 1.0)

            # Debug output
            print(f"Env {i} - Center: {reward_components['center_alignment']:.3f}, "
                  f"Goal: {reward_components['goal_progress']:.3f}, "
                  f"Lumen_Visibility: {reward_components['lumen_visibility']:.3f}, "
                  f"Total: {total_reward:.3f}")

            rewards.append(torch.tensor(total_reward,
                                       dtype=torch.float32,
                                       device='cuda' if torch.cuda.is_available() else 'cpu'))

        return torch.stack(rewards) * self.reward_scale
