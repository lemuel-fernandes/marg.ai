"""Vehicle state data types."""


from dataclasses import dataclass

@dataclass
class VehicleState:
    x: float              # Global X (meters)
    y: float              # Global Y (meters)
    heading: float        # Yaw angle (radians, 0 = +X axis)
    velocity: float       # Longitudinal velocity (m/s)
    steering_angle: float # Current steering angle (radians)
    timestamp: float      # Simulation time (seconds)