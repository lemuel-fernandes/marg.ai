from dataclasses import dataclass, field
from typing import List

from .base import Header, Pose2D


@dataclass
class PathPoint:
    pose: Pose2D
    curvature: float
    target_speed: float


@dataclass
class GlobalPath:
    header: Header
    points: List[PathPoint] = field(default_factory=list)
    length_m: float = 0.0
    is_feasible: bool = False
    replan_required: bool = False
    reason: str = ""