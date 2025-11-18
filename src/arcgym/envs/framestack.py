from collections import deque

import torch
import gymnasium as gym

class FrameStack(gym.Wrapper):
    """Stack n_frames last frames by concatenating along channel dimension."""
    
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)

        if not "rgb" in env.observation_space.spaces:
            raise ValueError("FrameStack requires the observations to be a dictionary with an 'rgb' image entry")

        original_shape = env.observation_space["rgb"].shape
        channels, height, width = original_shape

        new_channels = n_stack * channels
        
        self.observation_space["rgb"] = gym.spaces.Box(
            low=env.observation_space["rgb"].low.min(),
            high=env.observation_space["rgb"].high.max(),
            shape=(new_channels, height, width),
            dtype=env.observation_space["rgb"].dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        for _ in range(self.n_stack):
            self.frames.append(obs["policy"]["rgb"].clone())

        augmented_obs = self._get_stacked_obs()
        obs["policy"]["rgb"] = augmented_obs

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs["policy"]["rgb"].clone())

        augmented_obs = self._get_stacked_obs()
        obs["policy"]["rgb"] = augmented_obs

        return obs, reward, terminated, truncated, info

    def _get_stacked_obs(self):
        frames_list = list(self.frames)
        stacked = torch.concatenate(frames_list, axis=1)
        stacked = stacked.to(torch.float32) / 255.0
        return stacked
