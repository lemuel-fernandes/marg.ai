import math
from typing import List, Tuple

from src.common.types.base import Covariance2D, FrameId, Header, Pose2D, Twist2D
from src.common.types.config import CostmapConfig, DWAConfig, VehicleConfig
from src.common.types.obstacle import Obstacle, ObstacleBehavior, ObstacleClass
from src.common.types.vehicle_state import VehicleState

from .base import Scenario, ScenarioResult, min_obstacle_clearance


class SensorNoiseScenario(Scenario):
    name = "sensor_noise"
    duration_s = 18.0
    # Perception runs the standard vision path (base-class default); this
    # scenario injects its noise INTO that path at detection level so the
    # robustness test covers the full camera -> vision pipeline -> planner
    # chain. Magnitudes mirror the previous NoisyPerception injection.
    vision_pos_noise_std = 0.4   # m, base_link position noise
    vision_vel_noise_std = 0.2   # m/s, base_link velocity noise

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

    def evaluate(self, log, v_cfg) -> ScenarioResult:
        g = self.goal()
        final = log.states[-1].pose
        dist = math.hypot(g.x - final.x, g.y - final.y)
        
        # Evaluate clearance against TRUE ground truth, not the noisy perception
        clear = min_obstacle_clearance(log, self.obstacles_at)

        metrics = {
            "final_dist_m": dist,
            "min_clearance_m": clear,
            "emergency_stops": float(log.emergency_stops),
        }

        failures = []
        if dist > 2.0:
            failures.append(f"did not reach goal (dist={dist:.2f} m)")
        if log.emergency_stops > 5: # Allow a few phantom estops due to noise
            failures.append(f"too many phantom estops from noise ({log.emergency_stops})")
        if clear < 0.5:
            failures.append(f"true collision risk: min clearance {clear:.2f} m")

        return ScenarioResult(self.name, not failures, metrics, failures)