import math
import os
from typing import List

from src.common.types.base import FrameId, Header, Pose2D, Twist2D
from src.common.types.config import DWAConfig, VehicleConfig
from src.common.types.costmap import Costmap
from src.common.types.obstacle import Obstacle
from src.common.types.path import GlobalPath
from src.common.types.trajectory import LocalTrajectory, TrajectoryPoint
from src.common.types.vehicle_state import VehicleState

from .algorithms.dwa import DynamicWindowApproach

LP_CODE_VERSION = "v6-final-approach"

class DWALocalPlanner:
    """Tracks the global path via a lookahead target, avoids obstacles via DWA."""

    def __init__(self, vehicle_cfg: VehicleConfig, dwa_cfg: DWAConfig):
        if os.environ.get("PATHSENSE_DEBUG"):
            print(f"[CODE] local_planner.py loaded: {LP_CODE_VERSION}")
        self.dwa = DynamicWindowApproach(vehicle_cfg, dwa_cfg)
        self.vcfg = vehicle_cfg
        self.cfg = dwa_cfg
        self._arrived = False
        self._latched_goal = None

    def _lookahead_target(self, state: VehicleState, global_path: GlobalPath):
        pts = global_path.points
        if not pts:
            return state.pose, 0.0

        goal = pts[-1].pose
        dx_goal = goal.x - state.pose.x
        dy_goal = goal.y - state.pose.y
        dist_goal = math.hypot(dx_goal, dy_goal)

        # Reset latch if goal changed
        if self._latched_goal is None or math.hypot(
            goal.x - self._latched_goal.x, goal.y - self._latched_goal.y
        ) > 1e-6:
            self._arrived = False
            self._latched_goal = goal

        # Condition 1: Standard radius arrival
        if dist_goal <= self.cfg.arrival_radius_m:
            self._arrived = True

        # Only latch an overshoot when the vehicle is already at the goal.
        # Latching several metres away leaves a forward-only vehicle unable to
        # recover because DWA cannot select reverse motion.
        dot = math.cos(state.pose.heading) * dx_goal + math.sin(state.pose.heading) * dy_goal
        if dist_goal <= self.cfg.arrival_radius_m and dot < 0.0:
            self._arrived = True

        if self._arrived:
            return goal, 0.0

        # FIX: Final-Approach Creep Mode (< 10 m)
        # Aim directly at the goal and cap speed to a creep profile.
        if dist_goal < 10.0:
            # Brake against the distance available in the vehicle's current
            # heading, not just Euclidean distance to the goal.
            braking_speed = math.sqrt(
                2.0 * abs(self.vcfg.min_acceleration)
                * max(dot - 0.5, 0.0)
            ) if dot > 0.5 else 0.0
            creep = min(2.5, math.sqrt(2.0 * 1.2 * max(dist_goal - 0.3, 0.0)))
            creep = min(creep, braking_speed)
            return goal, creep

        # Cruise mode: path lookahead
        seg_idx = 0
        min_dist = float('inf')
        for i in range(len(pts) - 1):
            p1 = pts[i].pose
            p2 = pts[i+1].pose
            dx, dy = p2.x - p1.x, p2.y - p1.y
            l2 = dx*dx + dy*dy
            t = 0.0 if l2 < 1e-6 else max(0.0, min(1.0, ((state.pose.x - p1.x)*dx + (state.pose.y - p1.y)*dy) / l2))
            d = math.hypot(state.pose.x - (p1.x + t*dx), state.pose.y - (p1.y + t*dy))
            if d < min_dist:
                min_dist = d
                seg_idx = i

        a_comfort = 2.0
        speed_cap = math.sqrt(2.0 * a_comfort * max(dist_goal - 0.5, 0.0))

        acc = 0.0
        target = pts[seg_idx].pose
        speed = pts[seg_idx].target_speed
        for i in range(seg_idx, len(pts) - 1):
            seg = math.hypot(pts[i+1].pose.x - pts[i].pose.x, pts[i+1].pose.y - pts[i].pose.y)
            acc += seg
            target = pts[i+1].pose
            speed = pts[i+1].target_speed
            if acc >= self.cfg.lookahead_m:
                break

        return target, min(speed, speed_cap)

    def plan(self, state, global_path: GlobalPath, costmap: Costmap, obstacles: List[Obstacle]) -> LocalTrajectory:
        target, speed = self._lookahead_target(state, global_path)
        best = self.dwa.plan(state, target, speed, obstacles, costmap)

        if best is None:
            return self.emergency_stop(state)

        points: List[TrajectoryPoint] = [
            TrajectoryPoint(
                t=0.0, pose=state.pose, twist=state.twist,
                curvature=0.0, acceleration=0.0,
            )
        ]
        prev_v = state.twist.vx
        for (x, y, th, t) in best.points:
            accel = (best.v - prev_v) / self.cfg.dt
            curv = best.yaw_rate / best.v if best.v > 1e-3 else 0.0
            
            points.append(TrajectoryPoint(
                t=t,
                pose=Pose2D(x=x, y=y, heading=th),
                twist=Twist2D(vx=best.v, vy=0.0, yaw_rate=best.yaw_rate),
                curvature=curv,
                acceleration=max(self.vcfg.min_acceleration, min(self.vcfg.max_acceleration, accel)),
            ))
            prev_v = best.v

        return LocalTrajectory(
            header=Header(state.header.stamp, FrameId.MAP, "dwa_local_planner"),
            points=points,
            cost=0.0,
            is_safe=True,
            fallback_active=False,
            reason="dwa optimal candidate",
            debug_candidates=getattr(self.dwa, 'last_candidates', [])
        )

    def emergency_stop(self, state: VehicleState) -> LocalTrajectory:
        dt, n = 0.1, 30
        speed = max(state.twist.vx, 0.0)
        x, y, th = state.pose.x, state.pose.y, state.pose.heading
        points = []
        for i in range(n + 1):
            v = max(speed, 0.0)
            points.append(TrajectoryPoint(
                t=i * dt, pose=Pose2D(x, y, th), twist=Twist2D(vx=v),
                curvature=0.0, acceleration=-2.0 if v > 1e-3 else 0.0,
            ))
            x += v * math.cos(th) * dt
            y += v * math.sin(th) * dt
            speed -= 2.0 * dt
        return LocalTrajectory(
            header=Header(state.header.stamp, FrameId.MAP, "dwa_local_planner"),
            points=points, is_safe=True, fallback_active=True, reason="emergency stop",
        )