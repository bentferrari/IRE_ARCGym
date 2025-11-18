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

    def reset(self, initial_positions, goals):
        self.initial_positions = initial_positions
        self.goals = goals

    def __call__(self, action, current_states, _previous_states):
        # obs = current_states["obs"] why cannot access obs?
        # print("obs shape:", obs)
        robot_positions = current_states["robot_positions"]
        print("robot_pos:", robot_positions)
        depth_images = current_states["depth"]
        goals = self.goals
        entry_poss = self.initial_positions
        rewards = []

        for i, (robot_position, depth_img, goal, entry_pos) in enumerate(zip(robot_positions, depth_images, goals, entry_poss)):
            # print("robot_position:", robot_position)
            # print("goals:", self.goals)
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
                r_b = 1 - (((cx_dark - 64)**2 + ((cy_dark - 64)**2))**0.5) / 64
                r_c = 1 - d2target/d_max
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
            # if dark_ratio < 0.1:
            #     reward = -1
            if d2target < 0.02:
                reward = 1
                self.goal_reached = True
                print("Goal reached!")
            print("reward", reward) 
            rewards.append(torch.tensor(reward,
                                 dtype=torch.float32,
                                 device='cuda' if torch.cuda.is_available() else 'cpu')) 
            #rewards.append(reward)

        return torch.stack(rewards)
