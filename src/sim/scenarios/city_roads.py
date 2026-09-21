"""Curved road + uncontrolled 4-way intersection + cross traffic."""
import math
from typing import List, Tuple

from src.common.types.base import Covariance2D, FrameId, Header, Pose2D, Twist2D
from src.common.types.config import CostmapConfig, DWAConfig, VehicleConfig
from src.common.types.obstacle import Obstacle, ObstacleBehavior, ObstacleClass
from src.common.types.vehicle_state import VehicleState
from src.mapping.costmap import RoadNetwork

from .base import Scenario, ScenarioResult, min_obstacle_clearance, boundary_violation


def _arc(cx, cy, r, a0, a1, n=10):
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]

# Main road: straight -> left curve -> straight -> right curve -> straight
# Arc geometry: a circle arc's chord gap grows quadratically with the
# sampling angle (12 points per quarter arc -> ~2.6 m max sagitta gap here),
# which left a dead zone between the two arcs where the ego was ~8.5 m from
# any polyline while ON the road. The boundary gate measures center distance
# to the polylines, so the road model itself must be gap-free.
def _arc_dense(cx, cy, r, a0, a1, n=48):
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]

MAIN = ([(0, 0), (30, 0)] + _arc_dense(30, 12, 12, -90, 0) +
        [(42, 16), (42, 32)] + _arc_dense(54, 32, 12, 180, 90) + [(58, 44), (85, 44)])
CROSS = [(26, 22), (58, 22)]   # 4-way intersection road


def _polyline_length(poly):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(poly[:-1], poly[1:]))


MAIN_ARC_LENGTH = _polyline_length(MAIN)


def _point_at(poly, s):
    acc = 0.0
    for (ax, ay), (bx, by) in zip(poly[:-1], poly[1:]):
        L = math.hypot(bx - ax, by - ay)
        if acc + L >= s:
            t = (s - acc) / L
            return ax + (bx - ax) * t, ay + (by - ay) * t, math.atan2(by - ay, bx - ax)
        acc += L
    return poly[-1][0], poly[-1][1], 0.0


class CityRoadsScenario(Scenario):
    name = "city_roads"
    duration_s = 60.0
    # Response-aware TTC gate (s): fail if the ego fails to react to a
    # closing threat (see src/sim/metrics.py::ttc_gate_failure).
    min_ttc_gate_s = 0.25

    def configs(self):
        return (
            VehicleConfig(max_speed=6.0, max_steer_angle=0.6, wheelbase=2.5),
            CostmapConfig(width_m=120, height_m=80, resolution=0.5,
                          inflation_radius=1.0, fixed_origin=(-15.0, -15.0)),
            DWAConfig(obstacle_margin=1.5),
        )

    def road_network(self):
        return RoadNetwork(polylines=[MAIN, CROSS], half_width=3.5)

    def initial_state(self):
        return VehicleState(header=Header(0.0, FrameId.MAP, "scenario"),
                            pose=Pose2D(0.0, 0.0, 0.0),
                            twist=Twist2D(4.0, 0.0, 0.0), steering_angle=0.0)

    def goal(self):
        return Pose2D(80.0, 44.0, 0.0)

    def obstacles_at(self, t: float) -> List[Obstacle]:
        obs = []

        def add(tid, cls, x, y, hdg, vx, vy, L, W, dyn):
            obs.append(Obstacle(header=Header(t, FrameId.MAP, "scenario"), track_id=tid,
                                class_label=cls, behavior=ObstacleBehavior.STATIC,
                                pose=Pose2D(x, y, hdg), length=L, width=W,
                                velocity=Twist2D(vx=vx, vy=vy, yaw_rate=0.0),
                                pose_covariance=Covariance2D(), velocity_covariance=Covariance2D(),
                                confidence=0.9, is_dynamic=dyn))

        # Slow truck crawling along the main road (curves included).
        # Beyond the road end it keeps driving straight in its last heading instead of
        # parking on the corridor right before the goal (which geometrically deadlocks the ego).
        truck_s = 20.0 + 2.5 * t
        tx, ty, th = _point_at(MAIN, truck_s)
        if truck_s > MAIN_ARC_LENGTH:
            overshoot = truck_s - MAIN_ARC_LENGTH
            tx += overshoot * math.cos(th)
            ty += overshoot * math.sin(th)
        add(50, ObstacleClass.VEHICLE, tx, ty, th, 2.5 * math.cos(th), 2.5 * math.sin(th), 6.0, 2.0, True)

        # Cross-traffic bike 1 (west -> east) arriving as ego nears the intersection
        if 10.0 <= t <= 22.0:
            add(51, ObstacleClass.VEHICLE, 26 + 4.0 * (t - 10.0), 22.0, 0.0, 4.0, 0.0, 2.0, 1.0, True)
        # Cross-traffic bike 2 (east -> west)
        if 18.0 <= t <= 30.0:
            add(52, ObstacleClass.VEHICLE, 58 - 3.5 * (t - 18.0), 22.0, math.pi, -3.5, 0.0, 2.0, 1.0, True)
        # Pedestrian crossing the main road just after the intersection
        if 16.0 <= t <= 26.0:
            dt = t - 16.0
            add(53, ObstacleClass.PEDESTRIAN, 38.0 + 1.2 * dt, 28.0, 0.0, 1.2, 0.0, 0.5, 0.5, True)
        # Goat parked on the final straight
        add(54, ObstacleClass.ANIMAL, 70.0, 44.5, 0.0, 0.0, 0.0, 1.0, 0.6, False)
        # Dog darting across the first straight. Spawn-answerability: the
        # surprise must be physically answerable within the braking envelope
        # (see get_local_planner comment). At spawn (t=3.0) the ego is at
        # x~13.4 at 4.5 m/s: crossing at x=15 left only 1.6 m of gap vs the
        # ~2.9 m stopping distance — unanswerable except by luck (the old
        # global-route pothole detour masked this). x=20 gives a 5.6 m gap:
        # full brake stops the ego with ~1.7 m of margin to the crossing box.
        if 3.0 <= t <= 5.0:
            add(55, ObstacleClass.ANIMAL, 20.0, 5.0 - 5.0 * (t - 3.0), 0.0, 0.0, -5.0, 0.8, 0.4, True)
        # Potholes
        for i, (px, py) in enumerate([(12, 0.8), (25, -1.2), (43, 25.0), (62, 43.0), (75, 45.0)]):
            add(100 + i, ObstacleClass.POTHOLE, px, py, 0.0, 0.0, 0.0, 0.8, 0.8, False)
        return obs

    def get_local_planner(self, v_cfg, dwa_cfg):
        from src.local_planner.frenet_local_planner import AdaptiveFrenetLocalPlanner
        # Cruise target below the 6.0 m/s scenario cap: surprise spawns (a dog
        # darting out 3 m ahead) must be physically answerable within the
        # braking envelope, and unstructured-road speeds are low anyway.
        return AdaptiveFrenetLocalPlanner(v_cfg, target_speed=4.5, creep_speed=1.5)

    def plot_tracks(self):
        tracks = [{"points": MAIN, "markers": [], "radius": 0.3, "color": "gray", "ls": "-", "label": "Main road"},
                  {"points": CROSS, "markers": [], "radius": 0.3, "color": "gray", "ls": "--", "label": "Cross road"}]
        return tracks

    def evaluate(self, log, v_cfg) -> ScenarioResult:
        g = self.goal()
        final = log.states[-1].pose
        dist = math.hypot(g.x - final.x, g.y - final.y)
        clear = min_obstacle_clearance(log, self.obstacles_at)
        bviol = boundary_violation(log, self.road_network())
        min_accel = min((c.acceleration for c in log.commands), default=0.0)
        metrics = {"final_dist_m": dist, "min_clearance_m": clear,
                   "min_accel_mps2": min_accel, "emergency_stops": float(log.emergency_stops)}
        if bviol is not None:
            metrics["boundary_overshoot_m"] = round(bviol, 3)
        failures = []
        if clear < 0.2:
            failures.append(f"collision: min clearance {clear:.2f} m")
        if dist > 5.0:
            failures.append(f"did not reach goal (dist={dist:.2f} m)")
        if bviol is not None:
            failures.append(f"boundary violation: ego left road corridor by {bviol:.2f} m")
        return ScenarioResult(self.name, not failures, metrics, failures)