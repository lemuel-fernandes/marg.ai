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


def boundary_violation(log: RunLog, road_network,
                       corridor_margin_m: float = 0.5) -> Optional[float]:
    """Strict road-corridor boundary gate.

    The ego CENTER must stay within ``half_width - corridor_margin_m`` of at
    least one road polyline at every tick — i.e. the vehicle must keep a
    hard 0.5 m buffer between its center line and the road edge. Swerving
    off the carriageway (onto the shoulder / sidewalk / oncoming verge) to
    avoid an obstacle is a boundary violation even when no collision occurs.

    Returns the worst overshoot in metres (how far the ego center exceeded
    the allowed corridor) or None when the ego stayed inside it. Scenarios
    without a road network (bounds enforced by physical curbs) don't call
    this — for them the boundary is already a solid obstacle.
    """
    if road_network is None:
        return None
    allowed = road_network.half_width - corridor_margin_m
    worst = 0.0
    for s in log.states:
        d = min(point_polyline_distance(s.pose.x, s.pose.y, poly)
                for poly in road_network.polylines)
        worst = max(worst, d - allowed)
    return worst if worst > 0.0 else None


def min_obstacle_clearance(log: RunLog, obstacles_at) -> float:
    from src.common.utils.geometry import oriented_rect_clearance
    from src.common.types.obstacle import ObstacleClass
    worst = float("inf")
    for t, s in zip(log.times, log.states):
        for obs in obstacles_at(t):
            if obs.class_label == ObstacleClass.POTHOLE:
                continue
            d = oriented_rect_clearance(
                s.pose.x, s.pose.y,
                obs.pose.x, obs.pose.y, obs.pose.heading,
                obs.length, obs.width
            )
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
    # ROADMAP vision wiring: every scenario runs perception through the
    # synthetic camera -> VisionPerceptionPipeline path by default (GT
    # passthrough covers obstacles the camera cannot see — outside FOV/range
    # or behind the vehicle — so the planner never loses them). Set False on
    # a scenario to fall back to its get_perception_module() provider.
    use_vision_perception = True

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
    
    def get_perception_module(self, provider):
        from src.perception.ground_truth_perception import GroundTruthPerception
        return GroundTruthPerception(provider)
    
    def road_network(self):
        return None