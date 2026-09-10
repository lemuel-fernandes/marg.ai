"""Pure pursuit controller."""


import math

from src.common.types.base import FrameId, Header, normalize_angle
from src.common.types.config import VehicleConfig
from src.common.types.control import ControlCommand, ControlMode
from src.common.types.trajectory import LocalTrajectory
from src.common.types.vehicle_state import VehicleState


class PurePursuitController:
    def __init__(self, vcfg: VehicleConfig, kp: float = 1.2,
                 min_lookahead: float = 2.0, max_lookahead: float = 8.0,
                 lookahead_gain: float = 1.0):
        self.vcfg = vcfg
        self.kp = kp
        self.min_ld = min_lookahead
        self.max_ld = max_lookahead
        self.ld_gain = lookahead_gain

    def command(self, state: VehicleState, trajectory: LocalTrajectory) -> ControlCommand:
        header = Header(state.header.stamp, FrameId.VEHICLE, "pure_pursuit")

        if trajectory.fallback_active or not trajectory.points:
            return ControlCommand(header=header, steering_angle=state.steering_angle,
                                  steering_rate=0.0, acceleration=self.vcfg.min_acceleration,
                                  brake=1.0, mode=ControlMode.EMERGENCY_STOP)

        ld = max(self.min_ld, min(self.max_ld, self.ld_gain * max(state.twist.vx, 0.5)))

        target = trajectory.points[-1]
        for pt in trajectory.points:
            if math.hypot(pt.pose.x - state.pose.x, pt.pose.y - state.pose.y) >= ld:
                target = pt
                break

        dx = target.pose.x - state.pose.x
        dy = target.pose.y - state.pose.y
        alpha = normalize_angle(math.atan2(dy, dx) - state.pose.heading)
        steer = math.atan2(2.0 * self.vcfg.wheelbase * math.sin(alpha), max(ld, 1e-6))

        desired_a = max(self.vcfg.min_acceleration,
                        min(self.vcfg.max_acceleration, self.kp * (target.twist.vx - state.twist.vx)))
        brake = min(1.0, -desired_a / abs(self.vcfg.min_acceleration)) if desired_a < -0.3 else 0.0

        return ControlCommand(header=header, steering_angle=steer, steering_rate=0.0,
                              acceleration=desired_a, brake=brake, mode=ControlMode.NORMAL)