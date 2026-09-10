import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.common.types.base import Pose2D
from src.common.types.config import CostmapConfig, DWAConfig, VehicleConfig
from src.common.types.control import ControlCommand
from src.common.types.obstacle import Obstacle
from src.common.types.path import GlobalPath
from src.common.types.vehicle_state import VehicleState


@dataclass
class RunLog:
    times: List[float] = field(default_factory=list)
    states: List[VehicleState] = field(default_factory=list)
    commands: List[ControlCommand] = field(default_factory=list)
    global_path: Optional[GlobalPath] = None
    emergency_stops: int = 0


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)


def point_polyline_distance(px: float, py: float, pts: List[Tuple[float, float]]) -> float:
    best = float("inf")
    if len(pts) == 1:
        return math.hypot(px - pts[0][0], py - pts[0][1])
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        abx, aby = bx - ax, by - ay
        l2 = abx * abx + aby * aby
        t = 0.0 if l2 < 1e-12 else max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / l2))
        cx, cy = ax + abx * t, ay + aby * t
        best = min(best, math.hypot(px - cx, py - cy))
    return best


def min_obstacle_clearance(log: RunLog, obstacles_at) -> float:
    worst = float("inf")
    for t, s in zip(log.times, log.states):
        for obs in obstacles_at(t):
            r = max(obs.length, obs.width) / 2.0
            d = math.hypot(s.pose.x - obs.pose.x, s.pose.y - obs.pose.y) - r
            worst = min(worst, d)
    return worst


def lateral_deviation_series(log: RunLog) -> List[float]:
    if log.global_path is None or not log.global_path.points:
        return []
    pts = [(p.pose.x, p.pose.y) for p in log.global_path.points]
    return [point_polyline_distance(s.pose.x, s.pose.y, pts) for s in log.states]


class Scenario:
    name = "base"
    duration_s = 12.0

    def configs(self) -> Tuple[VehicleConfig, CostmapConfig, DWAConfig]:
        return (
            VehicleConfig(),
            CostmapConfig(width_m=100, height_m=100, resolution=0.5, inflation_radius=2.0),
            DWAConfig(),
        )

    def initial_state(self) -> VehicleState:
        raise NotImplementedError

    def goal(self) -> Pose2D:
        raise NotImplementedError

    def obstacles_at(self, t: float) -> List[Obstacle]:
        return []

    def plot_tracks(self) -> Optional[List[dict]]:
        return None

    def evaluate(self, log: RunLog, v_cfg: VehicleConfig) -> ScenarioResult:
        raise NotImplementedError