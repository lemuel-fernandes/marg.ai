"""Trajectory data types."""


from dataclasses import dataclass, field
from typing import List
from .vehicle_state import VehicleState

@dataclass
class ControlCommand:
    steering_angle: float  # radians
    throttle: float        # 0.0 to 1.0
    brake: float           # 0.0 to 1.0

@dataclass
class TrajectoryPoint:
    state: VehicleState
    control: ControlCommand
    time_from_start: float # seconds

@dataclass
class LocalTrajectory:
    points: List[TrajectoryPoint] = field(default_factory=list)
    cost: float = 0.0
    is_safe: bool = True