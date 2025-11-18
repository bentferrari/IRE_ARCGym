import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env import VecVideoRecorder
from gymnasium.spaces import Box

class IsaacLabVideoWrapper(VecEnvWrapper):
    """
    Wrapper to extract and convert camera observations for video recording
    """
    def __init__(self, venv):
        super().__init__(venv)
        
        # Preserve the original observation space structure
        if hasattr(venv, 'observation_space'):
            self.observation_space = venv.observation_space
        
    def reset(self):
        obs = self.venv.reset()
        return obs  # Don't modify observations, just pass them through
    
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs, rewards, dones, infos  # Don't modify observations
    
    def render(self, mode='rgb_array'):
        """
        Override render method to provide frames for video recording
        """
        if mode == 'rgb_array':
            # Get the current observation and convert it for video
            if hasattr(self, '_last_obs'):
                return self._convert_obs_to_video_frame(self._last_obs)
        return self.venv.render(mode=mode)
    
    def step_async(self, actions):
        self.venv.step_async(actions)
    
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self._last_obs = obs  # Store for rendering
        return obs, rewards, dones, infos
    
    def _convert_obs_to_video_frame(self, obs):
        """Convert observation to video frame format"""
        if isinstance(obs, torch.Tensor):
            obs_np = obs.cpu().numpy()
        elif isinstance(obs, np.ndarray):
            obs_np = obs
        else:
            return np.zeros((256, 256, 3), dtype=np.uint8)  # Fallback
        
        # Handle different shapes
        if len(obs_np.shape) == 4:  # Batch (N, C, H, W) or (N, H, W, C)
            obs_np = obs_np[0]  # Take first environment
        
        if len(obs_np.shape) == 3:
            if obs_np.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                obs_np = np.transpose(obs_np, (1, 2, 0))
        
        # Normalize to [0, 255]
        if obs_np.max() <= 1.0:
            obs_np = (obs_np * 255).astype(np.uint8)
        else:
            obs_np = obs_np.astype(np.uint8)
        
        return obs_np
    

from gymnasium.spaces import Box
import torch
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

class IsaacLabImagePreservingWrapper(VecEnvWrapper):
    """
    Wrapper that preserves image observation structure for CNN policies
    """
    def __init__(self, venv, expected_img_shape=(3, 256, 256)):
        self.expected_img_shape = expected_img_shape
        super().__init__(venv)
        
        # Override observation space to be image-like
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=expected_img_shape,  # (C, H, W) for CNN
            dtype=np.float32
        )
        
    def reset(self):
        obs = self.venv.reset()
        return self._process_obs(obs)
    
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._process_obs(obs), rewards, dones, infos
    
    def _process_obs(self, obs):
        """Convert observation to proper image format"""
        if isinstance(obs, torch.Tensor):
            obs_np = obs.cpu().numpy()
        elif isinstance(obs, np.ndarray):
            obs_np = obs
        else:
            raise ValueError(f"Unexpected observation type: {type(obs)}")
        
        # Handle different input shapes
        if len(obs_np.shape) == 1:  # Flattened (196608,)
            # Reshape to (C, H, W)
            obs_np = obs_np.reshape(self.expected_img_shape)
        elif len(obs_np.shape) == 2:  # Batch of flattened (N, 196608)
            batch_size = obs_np.shape[0]
            obs_np = obs_np.reshape(batch_size, *self.expected_img_shape)
        elif len(obs_np.shape) == 4:  # (N, C, H, W)
            if obs_np.shape[0] == 1:  # Single environment
                obs_np = obs_np[0]  # Remove batch dimension -> (C, H, W)
        elif len(obs_np.shape) == 3:  # Already (C, H, W)
            pass  # Keep as is
        
        # Ensure correct data type and range
        obs_np = obs_np.astype(np.float32)
        obs_np = np.clip(obs_np, 0.0, 1.0)
        
        return obs_np