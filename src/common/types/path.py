"""Path data types."""

from dataclasses import dataclass, field
from typing import List

@dataclass
class Waypoint:
    x: float
    y: float
    heading: float
    target_speed: float   # m/s

@dataclass
class GlobalPath:
    waypoints: List[Waypoint] = field(default_factory=list)
    is_valid: bool = True
    replan_reason: str = ""