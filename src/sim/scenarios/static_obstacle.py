from typing import List, Tuple

from src.common.types.base import Covariance2D, FrameId, Header, Pose2D, Twist2D
from src.common.types.config import CostmapConfig, DWAConfig, VehicleConfig
from src.common.types.obstacle import Obstacle, ObstacleBehavior, ObstacleClass
from src.common.types.vehicle_state import VehicleState

from .base import Scenario, ScenarioResult, min_obstacle_clearance


class StaticObstacleScenario(Scenario):
    name = "static_obstacle"
    duration_s = 12.0

    def configs(self) -> Tuple[VehicleConfig, CostmapConfig, DWAConfig]:
        return (
            VehicleConfig(max_speed=8.0, max_steer_angle=0.6, wheelbase=2.5),
            CostmapConfig(width_m=100, height_m=100, resolution=0.5, inflation_radius=2.0),
            DWAConfig(),
        )

    def initial_state(self) -> VehicleState:
        return VehicleState(
            header=Header(0.0, FrameId.MAP, "scenario"),
            pose=Pose2D(0.0, 0.0, 0.0),
            twist=Twist2D(5.0, 0.0, 0.0),
            steering_angle=0.0,
        )

    def goal(self) -> Pose2D:
        return Pose2D(25.0, 5.0, 0.0)

    def obstacles_at(self, t: float) -> List[Obstacle]:
        return [Obstacle(
            header=Header(t, FrameId.MAP, "scenario"), track_id=99,
            class_label=ObstacleClass.ANIMAL, behavior=ObstacleBehavior.STATIC,
            pose=Pose2D(12.0, 1.0, 0.0), length=2.0, width=2.0,
            velocity=Twist2D(), pose_covariance=Covariance2D(),
            velocity_covariance=Covariance2D(), confidence=1.0, is_dynamic=False,
        )]

    def plot_tracks(self):
        return [{"points": [(12.0, 1.0)], "markers": [0], "radius": 1.0,
                 "color": "red", "ls": "", "label": "Obstacle"}]

    def evaluate(self, log, v_cfg) -> ScenarioResult:
        import math
        g = self.goal()
        final = log.states[-1].pose
        dist = math.hypot(g.x - final.x, g.y - final.y)
        clear = min_obstacle_clearance(log, self.obstacles_at)

        metrics = {"final_dist_m": dist, "min_clearance_m": clear,
                   "emergency_stops": float(log.emergency_stops)}
        failures = []
        if dist > 1.5:
            failures.append(f"did not reach goal (dist={dist:.2f} m)")
        if log.emergency_stops > 0:
            failures.append(f"{log.emergency_stops} emergency stops")
        if clear < 0.5:
            failures.append(f"clearance too small ({clear:.2f} m)")
        return ScenarioResult(self.name, not failures, metrics, failures)