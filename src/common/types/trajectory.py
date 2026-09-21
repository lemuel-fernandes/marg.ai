from dataclasses import dataclass, field
from typing import List

from .base import Header, Pose2D, Twist2D


@dataclass
class TrajectoryPoint:
    t: float                    # relative time from now
    pose: Pose2D
    twist: Twist2D
    curvature: float
    acceleration: float


@dataclass
class LocalTrajectory:
    header: Header
    points: List[TrajectoryPoint] = field(default_factory=list)
    cost: float = float("inf")
    is_safe: bool = False
    fallback_active: bool = False
    reason: str = ""
    debug_candidates: list = field(default_factory=list) 