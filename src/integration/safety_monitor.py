import math
from typing import List

from src.common.types.base import FrameId, Header
from src.common.types.config import VehicleConfig
from src.common.types.control import ControlCommand, ControlMode
from src.common.types.costmap import Costmap
from src.common.types.obstacle import Obstacle
from src.common.types.safety import SafetyDecision
from src.common.types.trajectory import LocalTrajectory
from src.common.types.vehicle_state import VehicleState


class EnvelopeSafetyMonitor:
    def __init__(self, cfg: VehicleConfig):
        self.cfg = cfg

    def _reject(self, reason: str) -> SafetyDecision:
        return SafetyDecision(False, reason, self.cfg.max_acceleration,
                              self.cfg.min_acceleration, self.cfg.max_steer_angle)

    def _ok(self) -> SafetyDecision:
        return SafetyDecision(True, "Envelope + collision OK", self.cfg.max_acceleration,
                              self.cfg.min_acceleration, self.cfg.max_steer_angle)

    def evaluate(self, state, trajectory: LocalTrajectory,
                 obstacles: List[Obstacle], costmap: Costmap) -> SafetyDecision:
        # FIX: Use exact kinematic curvature limit tan(steer)/L, not linear steer/L
        max_curvature = math.tan(self.cfg.max_steer_angle) / max(self.cfg.wheelbase, 1e-6)

        if not trajectory.points:
            return self._reject("Empty trajectory")

        for pt in trajectory.points:
            if pt.twist.vx > self.cfg.max_speed + 1e-6:
                return self._reject("Trajectory exceeds max speed")
            if pt.acceleration > self.cfg.max_acceleration + 1e-6:
                return self._reject("Trajectory exceeds max acceleration")
            if pt.acceleration < self.cfg.min_acceleration - 1e-6:
                return self._reject("Trajectory below min acceleration")
            if abs(pt.curvature) > max_curvature + 1e-6:
                return self._reject(f"Trajectory exceeds curvature envelope ({pt.curvature:.3f} > {max_curvature:.3f})")

        # If the trajectory is an emergency fallback stop, allow it to execute braking
        if trajectory.fallback_active:
            return self._ok()

        # Predictive space-time collision check.
        # oriented_rect_clearance returns the distance from the OBB *surface*
        # (0.0 = touching). Reject only on true overlap or on a small
        # speed-dependent buffer at higher speeds. Applying a large static
        # margin at t=0 freezes the vehicle whenever it is legitimately closer
        # than the margin (e.g. right after a low-speed creep-past maneuver).
        from src.common.utils.geometry import oriented_rect_clearance
        from src.common.types.obstacle import is_surface_anomaly
        for pt in trajectory.points:
            # Speed-dependent buffer for higher-speed travel
            dynamic_margin = 0.1 + 0.1 * max(pt.twist.vx, 0.0)
            for obs in obstacles:
                # Surface anomalies (potholes/small ground debris) are handled by planner costs
                if is_surface_anomaly(obs):
                    continue

                px = obs.pose.x + obs.velocity.vx * pt.t
                py = obs.pose.y + obs.velocity.vy * pt.t
                clear = oriented_rect_clearance(pt.pose.x, pt.pose.y,
                                                px, py, obs.pose.heading,
                                                obs.length, obs.width)
                if clear < dynamic_margin:
                    return self._reject(f"Predicted collision at t={pt.t:.2f}s")

        return self._ok()

    def limit_command(self, state, command: ControlCommand, decision: SafetyDecision) -> ControlCommand:
        steering = max(-decision.max_abs_steer, min(decision.max_abs_steer, command.steering_angle))
        acceleration = max(decision.min_acceleration, min(decision.max_acceleration, command.acceleration))
        steering_rate = max(-self.cfg.max_steer_rate, min(self.cfg.max_steer_rate, command.steering_rate))
        brake = max(command.brake, 1.0) if command.mode == ControlMode.EMERGENCY_STOP else command.brake

        return ControlCommand(
            header=Header(command.header.stamp, FrameId.VEHICLE, "safety_monitor", command.header.seq),
            steering_angle=steering, steering_rate=steering_rate,
            acceleration=acceleration, brake=brake, mode=command.mode,
        )