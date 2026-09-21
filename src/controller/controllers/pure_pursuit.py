"""Pure pursuit controller."""


import math

from src.common.types.base import FrameId, Header, normalize_angle
from src.common.types.config import VehicleConfig
from src.common.types.control import ControlCommand, ControlMode
from src.common.types.trajectory import LocalTrajectory
from src.common.types.vehicle_state import VehicleState


class PurePursuitController:
    def __init__(self, vcfg: VehicleConfig, kp: float = 1.2,
                 min_lookahead: float = 2.0, max_lookahead: float = 6.0,
                 lookahead_gain: float = 0.8, control_dt: float = 0.1):
        self.vcfg = vcfg
        self.kp = kp
        self.min_ld = min_lookahead
        self.max_ld = max_lookahead
        self.ld_gain = lookahead_gain
        # One control-step tracking constant for deceleration. The DWA window
        # already limits the velocity step per tick (~max_accel * dt), so its
        # commanded speed must be applied at that same rate; a proportional
        # gain dilutes a 0.2 m/s step into ~0.24 m/s^2 and the vehicle
        # overshoots goals and brake envelopes.
        self.control_dt = control_dt

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

        delta_v = target.twist.vx - state.twist.vx
        # Speed handling is source-specific:
        # - DWA commands are one dynamic-window steps (~max_accel*dt), so they
        #   are by construction reachable within one control period. Apply the
        #   step exactly (both accel and decel): the P-gain dilutes a ~0.2 m/s
        #   step into ~0.24 m/s^2, which breaks DWA's internal motion model in
        #   BOTH directions (brake envelopes overrun, re-acceleration stalls).
        # - Polynomial planners (Frenet) bake smooth ramps into their
        #   trajectories; a proportional law tracks those without slamming.
        is_dwa = getattr(trajectory.header, "source", "") == "dwa_local_planner"
        if is_dwa:
            desired_a = max(self.vcfg.min_acceleration,
                            min(self.vcfg.max_acceleration, delta_v / self.control_dt))
        else:
            desired_a = max(self.vcfg.min_acceleration,
                            min(self.vcfg.max_acceleration, self.kp * delta_v))
        brake = min(1.0, -desired_a / abs(self.vcfg.min_acceleration)) if desired_a < -0.3 else 0.0

        return ControlCommand(header=header, steering_angle=steer, steering_rate=0.0,
                              acceleration=desired_a, brake=brake, mode=ControlMode.NORMAL)