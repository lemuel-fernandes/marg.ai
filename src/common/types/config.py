from dataclasses import dataclass
from typing import Optional, Tuple

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
    fixed_origin: Optional[Tuple[float, float]] = None 

@dataclass(frozen=True)
class PlannerConfig:
    global_replan_distance_m: float = 2.0
    local_horizon_s: float = 3.0
    local_dt: float = 0.1
    target_speed: float = 5.0
    

@dataclass
class PredictionConfig:
    """Predictive-corridor stamping for the costmap.

    Closing dynamic actors are swept forward (constant velocity) over the
    horizon and stamped as elevated (NOT lethal) cost, so the global planner
    schedules maneuvers around where actors *will be* — e.g. delaying an
    overtake until an oncoming vehicle's corridor has passed — without
    hard-blocking local maneuvering (cost stays below the lethal threshold).
    """
    horizon_s: float = 5.0
    step_s: float = 0.5
    corridor_cost: float = 0.55
    min_speed_mps: float = 1.0


@dataclass
class DWAConfig:
    horizon_s: float = 2.5
    dt: float = 0.1
    v_samples: int = 7
    yaw_rate_samples: int = 11
    max_yaw_rate: float = 1.0
    heading_weight: float = 1.0
    # goal_weight is defined further down (0.5) — the last field wins in a dataclass
    clearance_weight: float = 0.8
    velocity_weight: float = 1.0
    obstacle_margin: float = 2.5      # <--- CHANGED: Must be > Safety Monitor margin (2.25m)
    lookahead_m: float = 6.0
    arrival_radius_m: float = 1.5
    clearance_cap_m: float = 3.0
    stall_penalty: float = 1.5
    goal_weight: float = 0.5          # weight for goal-proximity score term
    lethal_threshold: float = 0.9     # costmap cost treated as lethal for rollouts