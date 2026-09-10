from dataclasses import dataclass
from typing import List

from src.common.types.control import ControlCommand, ControlMode
from src.common.types.costmap import Costmap
from src.common.types.base import Header, FrameId
from src.common.types.obstacle import Obstacle
from src.common.types.trajectory import LocalTrajectory
from src.common.types.vehicle_state import VehicleState


@dataclass
class VehicleSafetyLimits:
    max_abs_steer: float
    max_steer_rate: float
    max_acceleration: float
    min_acceleration: float
    max_speed: float


class SafetyDecision:
    def __init__(
        self,
        allowed: bool,
        reason: str,
        max_acceleration: float,
        min_acceleration: float,
        max_abs_steer: float,
    ):
        self.allowed = allowed
        self.reason = reason
        self.max_acceleration = max_acceleration
        self.min_acceleration = min_acceleration
        self.max_abs_steer = max_abs_steer


class EnvelopeSafetyMonitor:
    """
    Safety is independent from planner.
    It can reject trajectories and clamp control commands.
    """

    def __init__(self, limits: VehicleSafetyLimits):
        self.limits = limits

    def evaluate(
        self,
        state: VehicleState,
        trajectory: LocalTrajectory,
        obstacles: List[Obstacle],
        costmap: Costmap,
    ) -> SafetyDecision:
        # Basic hard envelope checks first.
        # More advanced checks (TTC, reachable sets, occupancy prediction)
        # should be added in Sprint 2/3.

        if not trajectory.points:
            return SafetyDecision(
                allowed=False,
                reason="Empty trajectory",
                max_acceleration=self.limits.max_acceleration,
                min_acceleration=self.limits.min_acceleration,
                max_abs_steer=self.limits.max_abs_steer,
            )

        for pt in trajectory.points:
            if abs(pt.acceleration) > self.limits.max_acceleration:
                return SafetyDecision(
                    allowed=False,
                    reason="Trajectory exceeds acceleration envelope",
                    max_acceleration=self.limits.max_acceleration,
                    min_acceleration=self.limits.min_acceleration,
                    max_abs_steer=self.limits.max_abs_steer,
                )

            if abs(pt.pose.heading) > 1e9:  # placeholder sanity check
                return SafetyDecision(
                    allowed=False,
                    reason="Invalid trajectory pose",
                    max_acceleration=self.limits.max_acceleration,
                    min_acceleration=self.limits.min_acceleration,
                    max_abs_steer=self.limits.max_abs_steer,
                )

        return SafetyDecision(
            allowed=True,
            reason="Envelope OK",
            max_acceleration=self.limits.max_acceleration,
            min_acceleration=self.limits.min_acceleration,
            max_abs_steer=self.limits.max_abs_steer,
        )

    def limit_command(
        self,
        state: VehicleState,
        command: ControlCommand,
        decision: SafetyDecision,
    ) -> ControlCommand:
        steering = max(-decision.max_abs_steer, min(decision.max_abs_steer, command.steering_angle))
        accel = max(decision.min_acceleration, min(decision.max_acceleration, command.acceleration))

        return ControlCommand(
            header=Header(
                stamp=command.header.stamp,
                frame_id=FrameId.VEHICLE,
                source="safety_monitor",
                seq=command.header.seq,
            ),
            steering_angle=steering,
            steering_rate=max(-self.limits.max_steer_rate, min(self.limits.max_steer_rate, command.steering_rate)),
            acceleration=accel,
            brake=command.brake,
            mode=command.mode,
        )