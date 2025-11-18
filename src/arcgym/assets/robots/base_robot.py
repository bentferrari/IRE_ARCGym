from abc import ABC, abstractmethod
import torch

class BaseRobot(ABC):
    """Abstract base class for all robots."""
    
    @abstractmethod
    def apply_action(self, actions: torch.Tensor) -> None:
        """Apply actions to the robot."""
        pass
    
    @abstractmethod
    def get_observation(self) -> torch.Tensor:
        """Get robot state observation."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset robot to initial state."""
        pass

    @property
    def action_space(self):
        return self.cfg.action_space

    @property
    def observation_space(self):
        return self.cfg.observation_space