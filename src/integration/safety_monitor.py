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
    """
    Safety is independent from planner.

    This monitor can:
    - reject trajectories
    - force fallback behavior
    - clamp control commands

    Sprint 2 should extend this with:
    - TTC checks
    - occupancy prediction
    - lateral safety corridor checks
    """

    def __init__(self, cfg: VehicleConfig):
        self.cfg = cfg

    def evaluate(
        self,
        state: VehicleState,
        trajectory: LocalTrajectory,
        obstacles: List[Obstacle],
        costmap: Costmap,
    ) -> SafetyDecision:
        max_curvature = self.cfg.max_steer_angle / max(self.cfg.wheelbase, 1e-6)

        if not trajectory.points:
            return SafetyDecision(
                allowed=False,
                reason="Empty trajectory",
                max_acceleration=self.cfg.max_acceleration,
                min_acceleration=self.cfg.min_acceleration,
                max_abs_steer=self.cfg.max_steer_angle,
            )

        for pt in trajectory.points:
            if pt.twist.vx > self.cfg.max_speed + 1e-6:
                return SafetyDecision(
                    allowed=False,
                    reason="Trajectory exceeds max speed",
                    max_acceleration=self.cfg.max_acceleration,
                    min_acceleration=self.cfg.min_acceleration,
                    max_abs_steer=self.cfg.max_steer_angle,
                )

            if pt.acceleration > self.cfg.max_acceleration + 1e-6:
                return SafetyDecision(
                    allowed=False,
                    reason="Trajectory exceeds max acceleration",
                    max_acceleration=self.cfg.max_acceleration,
                    min_acceleration=self.cfg.min_acceleration,
                    max_abs_steer=self.cfg.max_steer_angle,
                )

            if pt.acceleration < self.cfg.min_acceleration - 1e-6:
                return SafetyDecision(
                    allowed=False,
                    reason="Trajectory below min acceleration",
                    max_acceleration=self.cfg.max_acceleration,
                    min_acceleration=self.cfg.min_acceleration,
                    max_abs_steer=self.cfg.max_steer_angle,
                )

            if abs(pt.curvature) > max_curvature + 1e-6:
                return SafetyDecision(
                    allowed=False,
                    reason="Trajectory exceeds curvature envelope",
                    max_acceleration=self.cfg.max_acceleration,
                    min_acceleration=self.cfg.min_acceleration,
                    max_abs_steer=self.cfg.max_steer_angle,
                )

        return SafetyDecision(
            allowed=True,
            reason="Envelope OK",
            max_acceleration=self.cfg.max_acceleration,
            min_acceleration=self.cfg.min_acceleration,
            max_abs_steer=self.cfg.max_steer_angle,
        )

    def limit_command(
        self,
        state: VehicleState,
        command: ControlCommand,
        decision: SafetyDecision,
    ) -> ControlCommand:
        steering = max(
            -decision.max_abs_steer,
            min(decision.max_abs_steer, command.steering_angle),
        )

        acceleration = max(
            decision.min_acceleration,
            min(decision.max_acceleration, command.acceleration),
        )

        steering_rate = max(
            -self.cfg.max_steer_rate,
            min(self.cfg.max_steer_rate, command.steering_rate),
        )

        brake = command.brake
        if command.mode == ControlMode.EMERGENCY_STOP:
            brake = max(brake, 1.0)

        return ControlCommand(
            header=Header(
                stamp=command.header.stamp,
                frame_id=FrameId.VEHICLE,
                source="safety_monitor",
                seq=command.header.seq,
            ),
            steering_angle=steering,
            steering_rate=steering_rate,
            acceleration=acceleration,
            brake=brake,
            mode=command.mode,
        )