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

    def reset(self, initial_positions, goals, env_ids=None):
        # Check if this is the first initialization
        is_first_init = not hasattr(self, 'initial_positions') or self.initial_positions is None

        # Check if env_ids contains indices larger than current tensor size
        needs_full_init = is_first_init
        if not is_first_init and env_ids is not None and len(env_ids) > 0:
            max_env_id = int(env_ids.max().item()) if isinstance(env_ids, torch.Tensor) else max(env_ids)
            if max_env_id >= len(self.initial_positions):
                needs_full_init = True

        if env_ids is None or needs_full_init:
            if env_ids is not None and len(env_ids) > 0:
                max_env_id = int(env_ids.max().item()) if isinstance(env_ids, torch.Tensor) else max(env_ids)
                num_envs = max_env_id + 1
                device = initial_positions.device
                self.initial_positions = torch.zeros((num_envs, initial_positions.shape[-1]), dtype=initial_positions.dtype, device=device)
                self.goals = torch.zeros((num_envs,) + goals.shape[1:], dtype=goals.dtype, device=device)
                self.initial_positions[env_ids] = initial_positions
                self.goals[env_ids] = goals
            else:
                self.initial_positions = initial_positions
                self.goals = goals
        else:
            # Partial reset for specific environments
            if len(env_ids) > 0:
                self.initial_positions[env_ids] = initial_positions
                self.goals[env_ids] = goals

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

    def reset(self, initial_positions, goals, env_ids=None):
        # Check if this is the first initialization
        is_first_init = not hasattr(self, 'initial_positions') or self.initial_positions is None

        # Check if env_ids contains indices larger than current tensor size
        needs_full_init = is_first_init
        if not is_first_init and env_ids is not None and len(env_ids) > 0:
            max_env_id = int(env_ids.max().item()) if isinstance(env_ids, torch.Tensor) else max(env_ids)
            if max_env_id >= len(self.initial_positions):
                needs_full_init = True

        if env_ids is None or needs_full_init:
            if env_ids is not None and len(env_ids) > 0:
                max_env_id = int(env_ids.max().item()) if isinstance(env_ids, torch.Tensor) else max(env_ids)
                num_envs = max_env_id + 1
                device = initial_positions.device
                self.initial_positions = torch.zeros((num_envs, 1, initial_positions.shape[-1]), dtype=initial_positions.dtype, device=device)
                self.goals = torch.zeros((num_envs,) + goals.shape[1:], dtype=goals.dtype, device=device)
                self.initial_distances = torch.zeros((num_envs, goals.shape[1]), dtype=initial_positions.dtype, device=device)
                self.initial_positions[env_ids] = initial_positions.unsqueeze(1)
                self.goals[env_ids] = goals
                self.initial_distances[env_ids] = torch.norm(self.initial_positions[env_ids] - self.goals[env_ids], dim=2)
            else:
                self.initial_positions = initial_positions.unsqueeze(1)
                self.goals = goals
                self.initial_distances = torch.norm(self.initial_positions - self.goals, dim=2)
        else:
            # Partial reset for specific environments
            if len(env_ids) > 0:
                self.initial_positions[env_ids] = initial_positions.unsqueeze(1)
                self.goals[env_ids] = goals
                self.initial_distances[env_ids] = torch.norm(self.initial_positions[env_ids] - self.goals[env_ids], dim=1)

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

    def reset(self, initial_positions, goals, env_ids=None):
        # Check if this is the first initialization
        is_first_init = not hasattr(self, 'initial_positions') or self.initial_positions is None

        # Check if env_ids contains indices larger than current tensor size
        needs_full_init = is_first_init
        if not is_first_init and env_ids is not None and len(env_ids) > 0:
            max_env_id = int(env_ids.max().item()) if isinstance(env_ids, torch.Tensor) else max(env_ids)
            if max_env_id >= len(self.goal_reached_per_env):
                needs_full_init = True

        if env_ids is None or needs_full_init:
            if env_ids is not None and len(env_ids) > 0:
                max_env_id = int(env_ids.max().item()) if isinstance(env_ids, torch.Tensor) else max(env_ids)
                num_envs = max_env_id + 1
            else:
                num_envs = len(initial_positions)

            device = initial_positions.device
            self.initial_positions = torch.zeros((num_envs, initial_positions.shape[-1]), dtype=initial_positions.dtype, device=device)
            self.goals = torch.zeros((num_envs,) + goals.shape[1:], dtype=goals.dtype, device=device)
            self.goal_reached_per_env = torch.zeros(num_envs, dtype=torch.bool, device=device)

            if env_ids is not None and len(env_ids) > 0:
                self.initial_positions[env_ids] = initial_positions
                self.goals[env_ids] = goals
            else:
                self.initial_positions = initial_positions
                self.goals = goals
        else:
            # Partial reset for specific environments
            if len(env_ids) > 0:
                self.initial_positions[env_ids] = initial_positions
                self.goals[env_ids] = goals
                self.goal_reached_per_env[env_ids] = False

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
    ABLATION_VARIANTS = {
        "center_only",
        "depth_existence",
        "deep_area",
        "lumen_evidence",
        "full_reward",
    }

    def __init__(self, eps, reward_scale, center_weight=0.5, goal_weight=0.0, obstruction_weight=0.5,
                 alignment_threshold=0.8, consecutive_negative_threshold=200, reset_penalty=-100.0,
                 episode_length_s=20.0, decimation=2, dt=1.0/240.0,
                 success_alignment_ratio=0.8, success_distance_threshold=0.1,
                 reward_variant=None, reward_wc=None, reward_wo=None, reward_lambda_o=1.0,
                 reward_beta=0.6, train_with_normalized_reward=False,
                 reset_on_low_lumen_visibility=False, low_lumen_visibility_threshold=-0.4,
                 low_lumen_reset_penalty=-1.0, **kwargs):
        """
        Final reward function combining multiple penalty components.

        Args:
            eps: Epsilon threshold for goal reaching
            reward_scale: Overall reward scaling factor
            center_weight: Weight for lumen center deviation penalty
            goal_weight: Weight for goal distance penalty
            obstruction_weight: Weight for obstruction penalty
            alignment_threshold: Center alignment threshold for goal reaching (default 0.95)
            consecutive_negative_threshold: Number of consecutive -1 rewards before reset (default 50)
            reset_penalty: Penalty applied when consecutive negative threshold is reached (default -100.0)
            episode_length_s: Episode length in seconds (default 40.0, used in training mode)
            decimation: Number of physics steps per environment step (default 2)
            dt: Physics time step in seconds (default 1/240)
            success_alignment_ratio: Required ratio of steps with high alignment for success (default 0.90)
            success_distance_threshold: Distance threshold to target for success (default 0.05 meters)
        """
        self.reward_scale = reward_scale
        self.eps = eps
        self.center_weight = center_weight
        self.goal_weight = goal_weight
        self.obstruction_weight = obstruction_weight
        self.reward_variant = reward_variant
        self.reward_wc = center_weight if reward_wc is None else reward_wc
        self.reward_wo = obstruction_weight if reward_wo is None else reward_wo
        self.reward_lambda_o = reward_lambda_o
        self.reward_beta = reward_beta
        self.train_with_normalized_reward = train_with_normalized_reward
        self.reset_on_low_lumen_visibility = reset_on_low_lumen_visibility
        self.low_lumen_visibility_threshold = low_lumen_visibility_threshold
        self.low_lumen_reset_penalty = low_lumen_reset_penalty
        self.ablation_enabled = reward_variant is not None
        if self.ablation_enabled and reward_variant not in self.ABLATION_VARIANTS:
            raise ValueError(f"Unknown reward_variant: {reward_variant}")
        self.alignment_threshold = alignment_threshold
        self.consecutive_negative_threshold = consecutive_negative_threshold
        self.reset_penalty = reset_penalty
        self.success_alignment_ratio = success_alignment_ratio
        self.success_distance_threshold = success_distance_threshold
        # Calculate max_episode_length from episode_length_s, decimation, and dt
        # Formula: max_episode_length = episode_length_s / (decimation * dt)
        self.max_episode_length = int(episode_length_s / (decimation * dt))
        self.goal_reached_per_env = None
        self.center_alignment_per_env = None  # Track center alignment for each environment
        self.alignment_step_counter = None  # Count cumulative steps with alignment > threshold
        self.consecutive_negative_counter = None  # Count consecutive -1 rewards per environment
        self.should_reset_env = None  # Flag to indicate which environments should reset
        # Track high-alignment steps for percentage-based success
        self.high_alignment_step_counter = None  # Steps with center_alignment > 0.95
        self.latest_reward_metrics = {}

    def _reward_range(self, variant):
        """Return the theoretical raw-reward range for normalized logging.

        Ablation variants share one common range so normalized_mean_step_reward
        has the same absolute scale across center_only, partial rewards, and
        full_reward. With default coefficients this common range is [-0.5, 2.0].
        """
        if variant in self.ABLATION_VARIANTS:
            ranges = [
                (0.0, 1.0),  # center_only: s_c
                (0.0, 1.0 + float(self.reward_lambda_o)),  # partial cue variants
                (-float(self.reward_wo), float(self.reward_wc) + float(self.reward_wo)),  # full_reward
            ]
            return min(r[0] for r in ranges), max(r[1] for r in ranges)
        # Legacy reward is already clipped to [-1, 1] and is not part of the ablation scale.
        return -1.0, 1.0

    def _normalize_reward(self, raw_reward, variant):
        reward_min, reward_max = self._reward_range(variant)
        denom = max(reward_max - reward_min, 1e-6)
        normalized = 2.0 * (raw_reward - reward_min) / denom - 1.0
        return float(np.clip(normalized, -1.0, 1.0))

    def _compute_ablation_reward(self, s_c, s_1, s_2, s_3, s_o):
        """
        Compute the selected ablation reward.

        s_c is the center-alignment score.
        s_1 is the depth-existence cue.
        s_2 is the deep-region area cue.
        s_3 is the depth-contrast / geometric-confidence cue.
        s_o is the full lumen-visibility score.

        The ablation compares partial reward designs and the full reward.
        """
        variant = self.reward_variant
        if variant == "center_only":
            return s_c
        if variant == "depth_existence":
            return s_c + self.reward_lambda_o * s_1
        if variant == "deep_area":
            return s_c + self.reward_lambda_o * s_2
        if variant == "lumen_evidence":
            evidence = self.reward_beta * s_1 + (1.0 - self.reward_beta) * s_2
            return s_c + self.reward_lambda_o * evidence
        if variant == "full_reward":
            return self.reward_wc * s_c + self.reward_wo * s_o
        raise ValueError(f"Unknown reward_variant: {variant}")

    def reset(self, initial_positions, goals, env_ids=None):
        """
        Reset reward function state.

        Args:
            initial_positions: Entry positions for environments
            goals: Target positions for environments
            env_ids: Optional tensor of environment indices to reset. If None, resets all environments
                     and reinitializes all tensors. If provided, only resets specified environments.
        """
        # Check if this is the first initialization (no tensors exist yet)
        is_first_init = not hasattr(self, 'initial_positions') or self.initial_positions is None

        # Check if env_ids contains indices larger than current tensor size
        # This means we need to do a full initialization with the correct size
        needs_full_init = is_first_init
        if not is_first_init and env_ids is not None and len(env_ids) > 0:
            max_env_id = int(env_ids.max().item()) if isinstance(env_ids, torch.Tensor) else max(env_ids)
            if max_env_id >= len(self.goal_reached_per_env):
                needs_full_init = True

        if env_ids is None or needs_full_init:
            # Full reset - determine the correct number of environments
            if env_ids is not None and len(env_ids) > 0:
                max_env_id = int(env_ids.max().item()) if isinstance(env_ids, torch.Tensor) else max(env_ids)
                num_envs = max_env_id + 1
            else:
                num_envs = len(initial_positions)

            # Initialize all tensors with correct size
            device = initial_positions.device
            self.initial_positions = torch.zeros((num_envs, initial_positions.shape[-1]), dtype=initial_positions.dtype, device=device)
            self.goals = torch.zeros((num_envs,) + goals.shape[1:], dtype=goals.dtype, device=device)
            self.goal_reached_per_env = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self.center_alignment_per_env = torch.zeros(num_envs, dtype=torch.float32, device=device)
            self.alignment_step_counter = torch.zeros(num_envs, dtype=torch.int32, device=device)
            self.consecutive_negative_counter = torch.zeros(num_envs, dtype=torch.int32, device=device)
            self.should_reset_env = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self.high_alignment_step_counter = torch.zeros(num_envs, dtype=torch.int32, device=device)

            # Set values for the environments being reset
            if env_ids is not None and len(env_ids) > 0:
                self.initial_positions[env_ids] = initial_positions
                self.goals[env_ids] = goals
            else:
                self.initial_positions = initial_positions
                self.goals = goals
        else:
            # Partial reset - only reset specified environments without resizing tensors
            # Update positions and goals for specific environments
            if len(env_ids) > 0:
                self.initial_positions[env_ids] = initial_positions
                self.goals[env_ids] = goals
                # Reset counters for specific environments
                self.goal_reached_per_env[env_ids] = False
                self.center_alignment_per_env[env_ids] = 0.0
                self.alignment_step_counter[env_ids] = 0
                self.consecutive_negative_counter[env_ids] = 0
                self.should_reset_env[env_ids] = False
                self.high_alignment_step_counter[env_ids] = 0

    def __call__(self, action, current_states, _previous_states):
        robot_positions = current_states["robot_positions"]
        depth_images = current_states["depth"]
        goals = self.goals
        entry_poss = self.initial_positions
        rewards = []
        metric_accumulators = {
            "s_c": [],
            "s_1": [],
            "s_2": [],
            "s_3": [],
            "s_o": [],
            "raw_step_reward": [],
            "normalized_step_reward": [],
            "normalized_progress": [],
            "roi_aligned": [],
            "lumen_visible": [],
        }

        # Reset goal_reached status for this step
        if self.goal_reached_per_env is None:
            num_envs = len(robot_positions)
            self.goal_reached_per_env = torch.zeros(num_envs, dtype=torch.bool, device=robot_positions.device)
            self.center_alignment_per_env = torch.zeros(num_envs, dtype=torch.float32, device=robot_positions.device)
            self.alignment_step_counter = torch.zeros(num_envs, dtype=torch.int32, device=robot_positions.device)
            self.consecutive_negative_counter = torch.zeros(num_envs, dtype=torch.int32, device=robot_positions.device)
            self.should_reset_env = torch.zeros(num_envs, dtype=torch.bool, device=robot_positions.device)
            self.high_quadrants_per_env = []  # Store quadrant for each environment (using threshold_high)
            self.high_alignment_step_counter = torch.zeros(num_envs, dtype=torch.int32, device=robot_positions.device)
        else:
            self.goal_reached_per_env.fill_(False)
            self.should_reset_env.fill_(False)
            self.high_quadrants_per_env = []

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

            #Print depth statistics for each environment at each timestep
            print(f"Env {i} - min_depth: {min_depth:.4f}, max_depth: {max_depth:.4f}")

            # Find the centroid of the region with different thresholds based on proximity to wall
            # When close to wall (min_depth < 0.05), use lower thresholds to find the opening
            # Otherwise, use higher thresholds to focus on deeper lumen regions
            depth_threshold_low = 0.6 * max_depth
            depth_threshold_high = 0.7 * max_depth
            #print(f"using thresholds 0.6 and 0.7")
            if min_depth < 0.02 and max_depth <0.9 and max_depth >0.2:
                depth_threshold_low = 0.0 * max_depth
                depth_threshold_high = 0.2 * max_depth

                #print(f"Env {i} - close to wall, using thresholds 0.05 and 0.1")
            # Compute centroids for threshold_low and threshold_high regions separately
            # Region 1: pixels >= threshold_low
            low_region_mask = depth_img_np >= depth_threshold_low
            low_region_coords = np.argwhere(low_region_mask)

            # Region 2: pixels >= threshold_high
            high_region_mask = depth_img_np >= depth_threshold_high
            high_region_coords = np.argwhere(high_region_mask)

            # Helper function to determine quadrant
            def get_quadrant(row, col, center_row, center_col):
                if row < center_row and col < center_col:
                    return "upper-left"
                elif row < center_row and col >= center_col:
                    return "upper-right"
                elif row >= center_row and col < center_col:
                    return "lower-left"
                else:
                    return "lower-right"

            # Compute centroid for threshold_low region
            if len(low_region_coords) > 0:
                low_centroid_row = np.mean(low_region_coords[:, 0])
                low_centroid_col = np.mean(low_region_coords[:, 1])
                low_quadrant = get_quadrant(low_centroid_row, low_centroid_col, img_center_row, img_center_col)
                #print(f"Env {i} - threshold_low centroid: ({low_centroid_row:.1f}, {low_centroid_col:.1f}), quadrant: {low_quadrant}")
            # else:
            #     print(f"Env {i} - threshold_low region: No pixels found")

            # Compute centroid for threshold_high region (THIS IS USED FOR CENTER ALIGNMENT AND ACTION CLIPPING)
            if len(high_region_coords) > 0:
                high_centroid_row = np.mean(high_region_coords[:, 0])
                high_centroid_col = np.mean(high_region_coords[:, 1])
                high_quadrant = get_quadrant(high_centroid_row, high_centroid_col, img_center_row, img_center_col)
                #print(f"Env {i} - threshold_high centroid: ({high_centroid_row:.1f}, {high_centroid_col:.1f}), quadrant: {high_quadrant}")
                self.high_quadrants_per_env.append(high_quadrant)
            else:
                # Fallback to absolute deepest point if no pixels found
                max_depth_idx = np.unravel_index(np.argmax(depth_img_np), depth_img_np.shape)
                high_centroid_row, high_centroid_col = max_depth_idx
                print(f"Env {i} - threshold_high region: No pixels found, using deepest point")
                self.high_quadrants_per_env.append("center")  # Default when no pixels found

            # Calculate ratio of far-depth pixels using a dynamic threshold over the depth range
            depth_range = max_depth - min_depth
            total_pixels = depth_img_np.size
            FAR_DEPTH_FRACTION = 0.7  # Top fraction of depth range treated as "far"
            far_threshold = min_depth + FAR_DEPTH_FRACTION * depth_range
            if depth_range > 1e-6:
                high_depth_pixels = np.sum(depth_img_np >= far_threshold)
                high_depth_ratio = high_depth_pixels / total_pixels
            else:
                high_depth_ratio = 0.0

            # Check if too many pixels are in the shallow depth range (very close to camera)
            # This indicates facing a wall or obstruction
            # Use an absolute threshold instead of relative to avoid false positives
            SHALLOW_DEPTH_ABSOLUTE = 0.05  # Pixels closer than this are "very shallow"
            shallow_pixels = np.sum(depth_img_np < SHALLOW_DEPTH_ABSOLUTE)
            shallow_ratio = shallow_pixels / total_pixels

            # Calculate max possible distance from center (for normalization)
            max_center_distance = np.sqrt(img_center_row**2 + img_center_col**2)

            # Distance from high_centroid (threshold_high region centroid) to image center
            # This aligns the camera with the deepest part of the lumen
            deepest_point_deviation = np.sqrt((high_centroid_row - img_center_row)**2 +
                                             (high_centroid_col - img_center_col)**2)
            deepest_point_normalized = deepest_point_deviation / max_center_distance

            # print(f"Depth stats - min: {min_depth:.3f}, max: {max_depth:.3f}, mean: {mean_depth:.3f}, "
            #       f"variance: {depth_variance:.6f}, high_depth_ratio: {high_depth_ratio:.3f}, "
            #       f"deepest_deviation: {deepest_point_normalized:.3f}")

            # Calculate individual reward components
            reward_components = {}

            # 1. Lumen alignment: reward when deepest point (farthest) is centered
            # When facing lumen center, the deepest point should be in the center of the image
            center_alignment = 1.0 - deepest_point_normalized
            reward_components['center_alignment'] = center_alignment
            self.center_alignment_per_env[i] = center_alignment

            # 2. Lumen visibility quality. The ablation components are:
            # s_1: depth-existence cue, large when max depth indicates open lumen.
            # s_2: deep-region area cue, large when enough pixels belong to the far-depth region.
            # s_3: depth-contrast / geometric-confidence cue, large when the image has useful depth range.
            # s_o: full lumen-visibility score in [-1, 1].
            LUMEN_MIN_DEPTH = 0.20  # Minimum max depth for good lumen
            WALL_MAX_DEPTH = 0.05   # Maximum depth indicating wall
            MIN_HIGH_DEPTH_RATIO = 0.20  # Minimum ratio of far-depth pixels for quality view
            DEPTH_RANGE_MIN = 0.04  # Minimum depth range to consider lumen present
            DEPTH_RANGE_GOOD = 0.12  # Depth range considered strong lumen

            depth_span = max(LUMEN_MIN_DEPTH - WALL_MAX_DEPTH, 1e-6)
            s_1 = np.clip((max_depth - WALL_MAX_DEPTH) / depth_span, 0.0, 1.0)
            s_2 = np.clip((high_depth_ratio - MIN_HIGH_DEPTH_RATIO) / (1.0 - MIN_HIGH_DEPTH_RATIO), 0.0, 1.0)
            s_3 = np.clip((depth_range - DEPTH_RANGE_MIN) / max(DEPTH_RANGE_GOOD - DEPTH_RANGE_MIN, 1e-6), 0.0, 1.0)
            s_o = 2.0 * (self.reward_beta * s_1 + (1.0 - self.reward_beta) * s_2) * s_3 - 1.0

            if max_depth < WALL_MAX_DEPTH or depth_range < DEPTH_RANGE_MIN or shallow_ratio > 0.6:
                lumen_reward = -1.0
                if not self.ablation_enabled:
                    reward_components['center_alignment'] = -1.0  # Preserve legacy behavior.
            else:
                lumen_reward = s_o

            reward_components['lumen_visibility'] = lumen_reward

            # print(f"max_depth: {max_depth:.3f}, high_depth_ratio: {high_depth_ratio:.3f}, "
            #       f"lumen_visibility: {lumen_reward:.3f}")

            # 3. Penalty for deviation from robot position to goal position
            reward_components['goal_progress'] = 1.0 - (d2target / d_max).item()

            # Combine reward components with weights. If no ablation variant is selected,
            # preserve the legacy full reward exactly as the default behavior.
            if self.ablation_enabled:
                raw_reward = self._compute_ablation_reward(center_alignment, s_1, s_2, s_3, s_o)
                normalized_reward = self._normalize_reward(raw_reward, self.reward_variant)
                total_reward = normalized_reward if self.train_with_normalized_reward else raw_reward
            else:
                total_reward = (self.center_weight * reward_components['center_alignment'] +
                              #self.goal_weight * reward_components['goal_progress'] +
                              self.obstruction_weight * reward_components['lumen_visibility'])
                raw_reward = total_reward

            # Poor center alignment penalty - penalize when not well-aligned with lumen center
            # if center_alignment < 0.8:
            #     total_reward = -1.0

            # # Too many shallow pixels - indicates facing wall/obstruction
            # if shallow_ratio > 0.6:
            #     total_reward = -1.0

            # Hitting wall - severe penalty (very close collision)
            if not self.ablation_enabled and min_depth < 0.01:
                total_reward = -1.0
                raw_reward = total_reward

            # Normalize the reward to be in range [-1, 1]
            # The weighted combination naturally stays in this range for normal operation
            # Special conditions are already normalized
            if not self.ablation_enabled:
                total_reward = np.clip(total_reward, -1.0, 1.0)
                raw_reward = total_reward
                normalized_reward = total_reward

            # Vanilla PPO ablation safety rule: when action clipping is disabled,
            # a low lumen-visibility score means the robot has lost a useful lumen
            # view. Reset that env on the next done check and assign an immediate
            # bounded penalty to the agent.
            if self.reset_on_low_lumen_visibility and lumen_reward <= self.low_lumen_visibility_threshold:
                total_reward = self.low_lumen_reset_penalty
                raw_reward = self.low_lumen_reset_penalty
                normalized_reward = -1.0
                self.should_reset_env[i] = True
                self.consecutive_negative_counter[i] = 0
                print(
                    f"Env {i} - Low lumen visibility reset triggered "
                    f"(s_o={s_o:.3f}, lumen_reward={lumen_reward:.3f}, "
                    f"threshold={self.low_lumen_visibility_threshold:.3f})."
                )

            # Track consecutive -1 rewards and trigger reset if threshold is reached
            if abs(total_reward - (-1.0)) < 1e-6:  # Check if reward is -1
                self.consecutive_negative_counter[i] += 1
                #print(f"Env {i} - Consecutive -1 rewards: {self.consecutive_negative_counter[i].item()}/{self.consecutive_negative_threshold}")

                if self.consecutive_negative_counter[i] >= self.consecutive_negative_threshold:
                    # Apply reset penalty
                    total_reward = self.reset_penalty
                    raw_reward = total_reward
                    self.should_reset_env[i] = True
                    # Reset the counter for this environment
                    self.consecutive_negative_counter[i] = 0
                    print(f"Env {i} - Reset triggered! Applied penalty of {self.reset_penalty}")
            else:
                # Reset counter if reward is not -1
                self.consecutive_negative_counter[i] = 0

            # Update alignment step counter: increment if conditions met
            # Condition 1: center_alignment > 0.85
            # Condition 2: center_alignment > 0.7 AND lumen_visibility > 0
            # STRICT RULE: Never increment when total_reward = -1.0 (severe penalty)
            condition_1 = center_alignment > self.alignment_threshold and lumen_reward > -0.15
            condition_2 = center_alignment > 0.7 and lumen_reward > -0.15
            not_severe_penalty = abs(total_reward - (-1.0)) > 1e-6

            #if condition_1 and not_severe_penalty:
            if condition_1:
                self.high_alignment_step_counter[i] += 1
                print("High alignment step counted", self.high_alignment_step_counter[i].item())

            # # Increment high alignment counter if center_alignment > threshold
            # if center_alignment > self.alignment_threshold:
            #     self.high_alignment_step_counter[i] += 1
            #     print("High alignment step counted", self.high_alignment_step_counter[i].item())

            # Distance-based success criterion: check if robot tip is within threshold of target position
            # The target position is extracted from init_endpose_from_csv
            distance_to_target = d2target.item()
            distance_success = distance_to_target < self.success_distance_threshold

            if distance_success:
                if not self.ablation_enabled:
                    total_reward = 1.0
                    raw_reward = total_reward
                    normalized_reward = total_reward
                self.goal_reached_per_env[i] = True
                print(f"Goal reached in environment {i}! Robot tip within {self.success_distance_threshold}m of target position (distance: {distance_to_target:.4f}m).")

            # Final clamp after reset/goal overrides for normalized rewards
            if not self.ablation_enabled:
                total_reward = np.clip(total_reward, -1.0, 1.0)
                raw_reward = total_reward
                normalized_reward = total_reward

            # Print reward components and total reward
            print(f"Env {i} - Reward - Center: {reward_components['center_alignment']:.3f}, "
                  f"Goal: {reward_components['goal_progress']:.3f}, "
                  f"Lumen_Visibility: {reward_components['lumen_visibility']:.3f}, "
                  f"Total: {total_reward:.3f}")

            metric_accumulators["s_c"].append(float(center_alignment))
            metric_accumulators["s_1"].append(float(s_1))
            metric_accumulators["s_2"].append(float(s_2))
            metric_accumulators["s_3"].append(float(s_3))
            metric_accumulators["s_o"].append(float(s_o))
            metric_accumulators["raw_step_reward"].append(float(raw_reward))
            metric_accumulators["normalized_step_reward"].append(float(normalized_reward))
            metric_accumulators["normalized_progress"].append(float(np.clip(reward_components['goal_progress'], 0.0, 1.0)))
            metric_accumulators["roi_aligned"].append(1.0 if center_alignment > self.alignment_threshold else 0.0)
            metric_accumulators["lumen_visible"].append(1.0 if s_o > -0.15 else 0.0)

            rewards.append(torch.tensor(total_reward,
                                       dtype=torch.float32,
                                       device=robot_positions.device))

        # Store quadrant information for action clipping (accessible via current_states)
        # We'll store the quadrant for each environment to be used by the environment
        if not hasattr(self, 'low_quadrants'):
            self.low_quadrants = []

        reward_tensor = torch.stack(rewards) * self.reward_scale
        self.latest_reward_metrics = {
            key: torch.tensor(values, dtype=torch.float32, device=robot_positions.device)
            for key, values in metric_accumulators.items()
        }
        return reward_tensor
