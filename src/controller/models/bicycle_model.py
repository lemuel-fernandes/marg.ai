"""Kinematic bicycle model."""

import math
from dataclasses import replace

from src.common.types.base import FrameId, Header
from src.common.types.config import VehicleConfig
from src.common.types.control import ControlCommand
from src.common.types.vehicle_state import VehicleState


class KinematicBicycleModel:
    """
    Standard kinematic bicycle model.
    Deterministic and strictly respects vehicle configuration limits.
    """
    def __init__(self, cfg: VehicleConfig):
        self.cfg = cfg

    def step(self, state: VehicleState, cmd: ControlCommand, dt: float) -> VehicleState:
        # 1. Clamp inputs to physical limits
        steer = max(-self.cfg.max_steer_angle, min(self.cfg.max_steer_angle, cmd.steering_angle))
        accel = max(self.cfg.min_acceleration, min(self.cfg.max_acceleration, cmd.acceleration))
        
        # Apply braking if commanded
        if cmd.brake > 0.1:
            accel = min(accel, -self.cfg.min_acceleration * cmd.brake) # Hard braking

        # 2. Current state
        x = state.pose.x
        y = state.pose.y
        theta = state.pose.heading
        v = state.twist.vx

        # 3. Kinematic equations
        # x_dot = v * cos(theta)
        # y_dot = v * sin(theta)
        # theta_dot = (v / L) * tan(steer)
        
        dx = v * math.cos(theta) * dt
        dy = v * math.sin(theta) * dt
        dtheta = (v / self.cfg.wheelbase) * math.tan(steer) * dt
        dv = accel * dt

        # 4. Integration (Euler)
        next_x = x + dx
        next_y = y + dy
        next_theta = (theta + dtheta + math.pi) % (2 * math.pi) - math.pi  # Normalize
        next_v = max(0.0, v + dv) # Prevent reversing unless explicitly modeled
        
        # Cap max speed
        next_v = min(next_v, self.cfg.max_speed)

        # 5. Build new state
        new_header = Header(
            stamp=state.header.stamp + dt,
            frame_id=FrameId.MAP,
            source="bicycle_model",
            seq=state.header.seq + 1
        )

        return replace(
            state,
            header=new_header,
            pose=replace(state.pose, x=next_x, y=next_y, heading=next_theta),
            twist=replace(state.twist, vx=next_v),
            steering_angle=steer
        )