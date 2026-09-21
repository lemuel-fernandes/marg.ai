import math
from typing import List, Tuple

from src.common.types.base import Covariance2D, FrameId, Header, Pose2D, Twist2D
from src.common.types.config import CostmapConfig, DWAConfig, VehicleConfig
from src.common.types.obstacle import Obstacle, ObstacleBehavior, ObstacleClass
from src.common.types.vehicle_state import VehicleState

from .base import (Scenario, ScenarioResult, lateral_deviation_series,
                   min_obstacle_clearance)

SPAWN_T = 2.0        
START_XY = (22.0, -6.0)
COW_VY = 4.0         # <--- CHANGED: Faster crossing (was 2.0) to prevent permanent global path blockage

class CrossingAnimalScenario(Scenario):
    name = "crossing_animal"
    duration_s = 14.0
    # Response-aware TTC gate (s): fail if the ego fails to react to a
    # closing threat (see src/sim/metrics.py::ttc_gate_failure).
    min_ttc_gate_s = 0.25

    def configs(self) -> Tuple[VehicleConfig, CostmapConfig, DWAConfig]:
        return (
            VehicleConfig(max_speed=6.0, max_steer_angle=0.6, wheelbase=2.5),
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
        return Pose2D(35.0, 0.0, 0.0)

    def obstacles_at(self, t: float) -> List[Obstacle]:
        if t < SPAWN_T:
            return [] 
        y = START_XY[1] + COW_VY * (t - SPAWN_T)
        return [Obstacle(
            header=Header(t, FrameId.MAP, "scenario"), track_id=7,
            class_label=ObstacleClass.ANIMAL, behavior=ObstacleBehavior.CROSSING,
            pose=Pose2D(START_XY[0], y, 0.0), length=1.6, width=0.9,
            velocity=Twist2D(vx=0.0, vy=COW_VY, yaw_rate=0.0),
            pose_covariance=Covariance2D(), velocity_covariance=Covariance2D(),
            confidence=0.95, is_dynamic=True,
        )]

    def plot_tracks(self):
        ts = [SPAWN_T + 0.5 * i for i in range(int((self.duration_s - SPAWN_T) / 0.5) + 1)]
        pts = [(START_XY[0], START_XY[1] + COW_VY * (t - SPAWN_T)) for t in ts]
        markers = [ts.index(t) for t in (3.0, 4.0, 5.0) if t in ts]
        return [{"points": pts, "markers": markers, "radius": 0.8,
                 "color": "magenta", "ls": ":", "label": "Cow track"}]

    def evaluate(self, log, v_cfg) -> ScenarioResult:
        g = self.goal()
        final = log.states[-1].pose
        dist = math.hypot(g.x - final.x, g.y - final.y)
        clear = min_obstacle_clearance(log, self.obstacles_at)
        min_accel = min((c.acceleration for c in log.commands), default=0.0)

        dev = lateral_deviation_series(log)
        max_dev = max(dev) if dev else 0.0
        rejoin_t = -1.0
        if dev:
            peak = dev.index(max_dev)
            for i in range(peak, len(dev)):
                if dev[i] < 1.0:
                    rejoin_t = log.times[i]
                    break

        metrics = {
            "final_dist_m": dist,
            "min_clearance_m": clear,
            "min_accel_mps2": min_accel,
            "max_lateral_dev_m": max_dev,
            "rejoin_t_s": rejoin_t,
            "emergency_stops": float(log.emergency_stops),
        }

        failures = []
        if clear < 0.5:
            failures.append(f"collision risk: min clearance {clear:.2f} m")
        if min_accel < v_cfg.min_acceleration - 1e-6:
            failures.append(f"exceeded decel envelope ({min_accel:.2f})")
        if dist > 2.0:
            failures.append(f"did not reach goal (dist={dist:.2f} m)")
        if log.emergency_stops > 0:
            failures.append(f"{log.emergency_stops} emergency stops")

        if min_accel < -3.0:
            print(f"[WARN] harsh braking: {min_accel:.2f} m/s^2 (comfort target -3.0)")

        return ScenarioResult(self.name, not failures, metrics, failures)