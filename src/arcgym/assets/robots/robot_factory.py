from arcgym.assets.robots.base_robot import BaseRobot
from arcgym.assets.robots.capsule import RobotEndoscopeCapsule
from arcgym.assets.robots.robot_endoscope_soft import SoftEndoscopeChain
from arcgym.assets.robots.magnetic_endoscope import RobotEndoscopeChain
from arcgym.assets.robots.proximal_actuated_flexible import RobotEndoscopeChain as ProximalActuatedRobot

class RobotFactory:
    def __init__(self, config: dict, device="cuda"):
        self.config = config
        self.robot_config = config["robot_config"]
        self.robot_type = self.robot_config["robot_type"]

        self.device = device
        if self.robot_type == "capsule":
            self.isaac_robot_cfg = RobotEndoscopeCapsule.make_isaac_config(self.config)
        elif self.robot_type == "soft_endoscope":
            self.isaac_robot_cfg = SoftEndoscopeChain.make_isaac_config(self.config)
        elif self.robot_type == "magnetic_endoscope":
            self.isaac_robot_cfg = RobotEndoscopeChain.make_isaac_config(self.config)
        elif self.robot_type == "proximal_actuated":
            self.isaac_robot_cfg = ProximalActuatedRobot.make_isaac_config(self.config)
        else:
            raise ValueError(f"Unknown robot type in config: {self.robot_type}")

    @property
    def action_space(self):
        return self.isaac_robot_cfg.action_space

    @property
    def observation_space(self):
        return self.isaac_robot_cfg.observation_space

    def build_robot(self, scene, init_pos, init_rot):
        """Build robot instance."""
        if self.robot_type == "capsule":
            return RobotEndoscopeCapsule(scene, self.config, self.isaac_robot_cfg, init_pos, init_rot, self.device)
        elif self.robot_type == "soft_endoscope":
            return SoftEndoscopeChain(scene, self.config, self.isaac_robot_cfg, init_pos, init_rot, self.device)
        elif self.robot_type == "magnetic_endoscope":
            return RobotEndoscopeChain(scene, self.config, self.isaac_robot_cfg, init_pos, init_rot, self.device)
        elif self.robot_type == "proximal_actuated":
            return ProximalActuatedRobot(scene, self.config, self.isaac_robot_cfg, init_pos, init_rot, self.device)
        else:
            raise ValueError(f"Unknown robot type: {self.robot_type}")