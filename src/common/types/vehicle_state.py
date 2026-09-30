from dataclasses import dataclass

from .base import Header, Pose2D, Twist2D


@dataclass
class VehicleState:
    header: Header
    pose: Pose2D
    twist: Twist2D
    steering_angle: float
    curvature: float = 0.0