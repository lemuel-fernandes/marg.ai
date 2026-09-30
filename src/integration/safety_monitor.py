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
        from src.common.utils.geometry import oriented_rect_clearance_batch
        from src.common.types.obstacle import is_surface_anomaly
        import numpy as np

        # Evaluate every (trajectory point, obstacle) pair in ONE vectorized
        # array pass: distances are identical to the per-pair
        # oriented_rect_clearance loop it replaces (see the batch helper), and
        # the early-reject rule is preserved by scanning points in order and
        # taking the min clearance per point — the first point with a violation
        # produces the same "Predicted collision at t=..." rejection.
        solid = [o for o in obstacles if not is_surface_anomaly(o)]
        if solid:
            t_col = np.array([pt.t for pt in trajectory.points])[:, None]   # (n,1)
            fx = (np.array([o.pose.x for o in solid])[None, :]
                  + np.array([o.velocity.vx for o in solid])[None, :] * t_col)
            fy = (np.array([o.pose.y for o in solid])[None, :]
                  + np.array([o.velocity.vy for o in solid])[None, :] * t_col)
            clear = oriented_rect_clearance_batch(
                np.array([pt.pose.x for pt in trajectory.points])[:, None],
                np.array([pt.pose.y for pt in trajectory.points])[:, None],
                fx, fy,
                [o.pose.heading for o in solid],
                [o.length for o in solid], [o.width for o in solid])  # (n, n_obs)
            margins = np.array([0.1 + 0.1 * max(pt.twist.vx, 0.0)
                                for pt in trajectory.points])
            bad = (clear < margins[:, None]).any(axis=1)
            if bad.any():
                pt = trajectory.points[int(np.argmax(bad))]
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