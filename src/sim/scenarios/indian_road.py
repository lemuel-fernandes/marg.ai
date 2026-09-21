"""
Indian Rural Road Chaos Scenario (v3: Left-Hand Traffic + Pass Abortion)
========================================================================
Simulates a 7m wide rural Indian road with curbs, traffic, and chaos.
Forces the planner to perform complex maneuvers like "Pass Abortion" 
(crossing into the oncoming lane to pass a truck, then braking to tuck 
back in behind the truck when an oncoming bike appears).
"""

import math
import random
from typing import List, Tuple

from src.common.types.base import Covariance2D, FrameId, Header, Pose2D, Twist2D
from src.common.types.config import CostmapConfig, DWAConfig, VehicleConfig
from src.common.types.obstacle import Obstacle, ObstacleBehavior, ObstacleClass
from src.common.types.vehicle_state import VehicleState

from .base import Scenario, ScenarioResult, min_obstacle_clearance

random.seed(42)

ROAD_LENGTH = 80.0

# ---------- ROAD BOUNDARIES (Curbs) ----------
def _road_boundaries(t: float) -> List[Obstacle]:
    obs = []
    # Place 2.5m long blocks every 2m along the edges to form a wall
    for x in range(-10, int(ROAD_LENGTH) + 10, 2):
        # Left curb (Y = 4.5)
        obs.append(Obstacle(
            header=Header(t, FrameId.MAP, "scenario"), track_id=2000 + x,
            class_label=ObstacleClass.UNKNOWN, behavior=ObstacleBehavior.STATIC,
            pose=Pose2D(x, 4.5, 0.0), length=2.5, width=1.0,
            velocity=Twist2D(), pose_covariance=Covariance2D(), velocity_covariance=Covariance2D(),
            confidence=1.0, is_dynamic=False,
        ))
        # Right curb (Y = -4.5)
        obs.append(Obstacle(
            header=Header(t, FrameId.MAP, "scenario"), track_id=3000 + x,
            class_label=ObstacleClass.UNKNOWN, behavior=ObstacleBehavior.STATIC,
            pose=Pose2D(x, -4.5, 0.0), length=2.5, width=1.0,
            velocity=Twist2D(), pose_covariance=Covariance2D(), velocity_covariance=Covariance2D(),
            confidence=1.0, is_dynamic=False,
        ))
    return obs

# ---------- TRAFFIC AGENTS (Left-Hand Traffic) ----------
def _slow_truck(t: float):
    """A slow truck in the LEFT lane (forward traffic). Ego must pass it using the RIGHT (oncoming) lane."""
    x = 15.0 + 2.5 * t
    return Obstacle(
        header=Header(t, FrameId.MAP, "scenario"), track_id=50,
        class_label=ObstacleClass.VEHICLE, behavior=ObstacleBehavior.STATIC, 
        pose=Pose2D(x, 1.5, 0.0), length=6.0, width=2.0, # Left lane
        velocity=Twist2D(vx=2.5, vy=0.0, yaw_rate=0.0),
        pose_covariance=Covariance2D(), velocity_covariance=Covariance2D(),
        confidence=0.95, is_dynamic=True,
    )

def _oncoming_bike(t: float):
    """An oncoming bike in the RIGHT lane. Prevents ego from passing the truck blindly."""
    x = 100.0 - 5.0 * t
    if x < -5.0 or x > 85.0: return None
    return Obstacle(
        header=Header(t, FrameId.MAP, "scenario"), track_id=51,
        class_label=ObstacleClass.VEHICLE, behavior=ObstacleBehavior.STATIC,
        pose=Pose2D(x, -1.5, math.pi), length=2.0, width=1.0, # Right lane
        velocity=Twist2D(vx=-5.0, vy=0.0, yaw_rate=0.0),
        pose_covariance=Covariance2D(), velocity_covariance=Covariance2D(),
        confidence=0.90, is_dynamic=True,
    )

# ---------- CHAOS AGENTS ----------
POTHOLES = []
for i in range(6):
    px = random.uniform(10.0, ROAD_LENGTH - 10.0)
    py = random.uniform(-3.0, 3.0) # Keep inside the road
    size = random.uniform(0.4, 1.0)
    POTHOLES.append((px, py, size))

def _stubborn_goat(t: float):
    return Obstacle(
        header=Header(t, FrameId.MAP, "scenario"), track_id=12,
        class_label=ObstacleClass.ANIMAL, behavior=ObstacleBehavior.STATIC,
        pose=Pose2D(60.0, 1.5, 0.0), length=1.0, width=0.6, # In the left lane
        velocity=Twist2D(), pose_covariance=Covariance2D(), velocity_covariance=Covariance2D(),
        confidence=0.95, is_dynamic=False,
    )

def _darting_dog(t: float):
    spawn_t = 8.0
    despawn_t = 11.0
    if t < spawn_t or t > despawn_t: return None
    dt = t - spawn_t
    x = 45.0 + 1.0 * dt          
    y = 4.0 - 8.0 * dt            # Darts completely across the road
    return Obstacle(
        header=Header(t, FrameId.MAP, "scenario"), track_id=11,
        class_label=ObstacleClass.ANIMAL, behavior=ObstacleBehavior.STATIC,
        pose=Pose2D(x, y, 0.0), length=0.8, width=0.4,
        velocity=Twist2D(vx=1.0, vy=-8.0, yaw_rate=0.0),
        pose_covariance=Covariance2D(), velocity_covariance=Covariance2D(),
        confidence=0.75, is_dynamic=True,
    )

class IndianRoadScenario(Scenario):
    name = "indian_road"
    duration_s = 60.0 
    # Response-aware TTC gate (s): fail if the ego fails to react to a
    # closing threat (see src/sim/metrics.py::ttc_gate_failure).
    min_ttc_gate_s = 0.25

    def configs(self) -> Tuple[VehicleConfig, CostmapConfig, DWAConfig]:
        # FIX: Expanded costmap to 200x100 so the goal at X=75 is never out of bounds.
        # Reduced inflation to 1.0m so the car can physically fit past the truck.
        return (
            VehicleConfig(max_speed=6.0, max_steer_angle=0.6, wheelbase=2.5),
            CostmapConfig(width_m=200, height_m=100, resolution=0.5, inflation_radius=1.0),
            DWAConfig(obstacle_margin=1.5), 
        )

    def initial_state(self) -> VehicleState:
        return VehicleState(
            header=Header(0.0, FrameId.MAP, "scenario"),
            pose=Pose2D(0.0, 1.5, 0.0), # Start in the LEFT lane (forward traffic)
            twist=Twist2D(4.0, 0.0, 0.0),
            steering_angle=0.0,
        )

    def goal(self) -> Pose2D:
        return Pose2D(75.0, 1.5, 0.0)

    def obstacles_at(self, t: float) -> List[Obstacle]:
        obs = []
        obs.extend(_road_boundaries(t))
        obs.append(_slow_truck(t))
        
        bike = _oncoming_bike(t)
        if bike: obs.append(bike)
        
        obs.append(_stubborn_goat(t))
        
        dog = _darting_dog(t)
        if dog: obs.append(dog)

        for i, (px, py, size) in enumerate(POTHOLES):
            obs.append(Obstacle(
                header=Header(t, FrameId.MAP, "scenario"), track_id=100 + i,
                class_label=ObstacleClass.UNKNOWN, behavior=ObstacleBehavior.STATIC,
                pose=Pose2D(px, py, 0.0), length=size, width=size,
                velocity=Twist2D(), pose_covariance=Covariance2D(), velocity_covariance=Covariance2D(),
                confidence=0.70, is_dynamic=False,
            ))
        return obs

    def plot_tracks(self):
        tracks = []
        tracks.append({"points": [(x, 4.5) for x in range(-10, 82, 2)], "markers": [], "radius": 0.5, "color": "black", "ls": "-", "label": "Left Curb"})
        tracks.append({"points": [(x, -4.5) for x in range(-10, 82, 2)], "markers": [], "radius": 0.5, "color": "black", "ls": "-", "label": "Right Curb"})
        
        truck_ts = [0.1 * i for i in range(600)]
        truck_pts = [(15.0 + 2.5 * t, 1.5) for t in truck_ts]
        tracks.append({"points": truck_pts, "markers": [], "radius": 1.0, "color": "blue", "ls": "-", "label": "Slow Truck"})

        bike_ts = [0.1 * i for i in range(140)]
        bike_pts = [(65.0 - 5.0 * t, -1.5) for t in bike_ts if (65.0 - 5.0 * t) > 0]
        tracks.append({"points": bike_pts, "markers": [], "radius": 0.5, "color": "red", "ls": "-", "label": "Oncoming Bike"})

        for px, py, size in POTHOLES:
            tracks.append({"points": [(px, py)], "markers": [0], "radius": size/2, "color": "brown", "ls": "", "label": ""})
            
        tracks.append({"points": [(60.0, 1.5)], "markers": [0], "radius": 0.5, "color": "green", "ls": "", "label": "Goat"})
        
        return tracks

    def evaluate(self, log, v_cfg) -> ScenarioResult:
        g = self.goal()
        final = log.states[-1].pose
        dist = math.hypot(g.x - final.x, g.y - final.y)
        clear = min_obstacle_clearance(log, self.obstacles_at)
        min_accel = min((c.acceleration for c in log.commands), default=0.0)

        metrics = {
            "final_dist_m": dist,
            "min_clearance_m": clear,
            "min_accel_mps2": min_accel,
            "emergency_stops": float(log.emergency_stops),
        }

        failures = []
        if clear < 0.2: 
            failures.append(f"collision: min clearance {clear:.2f} m")
        if dist > 5.0:
            failures.append(f"did not reach goal (dist={dist:.2f} m)")

        if min_accel < -3.0:
            print(f"[WARN] harsh braking: {min_accel:.2f} m/s^2")

        return ScenarioResult(self.name, not failures, metrics, failures)