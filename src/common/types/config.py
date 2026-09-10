from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleConfig:
    length: float = 4.2
    width: float = 1.9
    wheelbase: float = 2.7
    max_steer_angle: float = 0.6
    max_steer_rate: float = 0.35
    max_acceleration: float = 2.0
    min_acceleration: float = -3.5
    max_speed: float = 15.0


@dataclass(frozen=True)
class CostmapConfig:
    width_m: float = 60.0
    height_m: float = 60.0
    resolution: float = 0.25
    inflation_radius: float = 0.75


@dataclass(frozen=True)
class PlannerConfig:
    global_replan_distance_m: float = 2.0
    local_horizon_s: float = 3.0
    local_dt: float = 0.1
    target_speed: float = 5.0
    
    
@dataclass(frozen=True)
class DWAConfig:
    horizon_s: float = 2.5
    dt: float = 0.1
    v_samples: int = 7
    yaw_rate_samples: int = 11
    max_yaw_rate: float = 1.0
    heading_weight: float = 1.0
    clearance_weight: float = 0.8     # was 2.0 — no longer dominates
    velocity_weight: float = 1.0      # was 0.5 — progress matters
    obstacle_margin: float = 1.5
    lookahead_m: float = 6.0
    arrival_radius_m: float = 2.0
    clearance_cap_m: float = 3.0      # was 5.0 — saturate early so "far" == "far enough"
    stall_penalty: float = 1.5        # NEW: penalty for standing still