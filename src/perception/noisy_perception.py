import random
from dataclasses import replace
from typing import Callable, List

from src.common.types.base import FrameId, Header
from src.common.types.obstacle import Obstacle
from src.common.types.perception import PerceptionOutput
from src.common.types.sensor import SensorFrame


class NoisyPerception:
    """Injects Gaussian noise into ground truth to simulate a real, imperfect CV model."""

    def __init__(self, ground_truth_provider: Callable[[], List[Obstacle]], 
                 pos_noise_std: float = 0.5, vel_noise_std: float = 0.3):
        self.provider = ground_truth_provider
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std

    def process(self, frame: SensorFrame) -> PerceptionOutput:
        gt_obstacles = self.provider()
        noisy_obstacles = []
        
        for obs in gt_obstacles:
            noisy_x = obs.pose.x + random.gauss(0, self.pos_noise_std)
            noisy_y = obs.pose.y + random.gauss(0, self.pos_noise_std)
            noisy_vx = obs.velocity.vx + random.gauss(0, self.vel_noise_std)
            noisy_vy = obs.velocity.vy + random.gauss(0, self.vel_noise_std)
            
            noisy_pose = replace(obs.pose, x=noisy_x, y=noisy_y)
            noisy_vel = replace(obs.velocity, vx=noisy_vx, vy=noisy_vy)
            
            noisy_obs = replace(obs, pose=noisy_pose, velocity=noisy_vel)
            noisy_obstacles.append(noisy_obs)
            
        return PerceptionOutput(
            header=Header(
                stamp=frame.header.stamp,
                frame_id=FrameId.MAP,
                source="noisy_perception",
            ),
            obstacles=noisy_obstacles,
            latency_ms=2.0, # Simulate a heavier CV model
            status="OK_NOISY",
        )