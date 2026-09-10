"""Local planner facade."""

import math
from typing import List

from src.common.types.base import FrameId, Header, Pose2D, Twist2D
from src.common.types.config import DWAConfig, VehicleConfig
from src.common.types.costmap import Costmap
from src.common.types.obstacle import Obstacle
from src.common.types.path import GlobalPath
from src.common.types.trajectory import LocalTrajectory, TrajectoryPoint
from src.common.types.vehicle_state import VehicleState

from .algorithms.dwa import DynamicWindowApproach


class DWALocalPlanner:
    """Tracks the global path via a lookahead target, avoids obstacles via DWA."""

    def __init__(self, vehicle_cfg: VehicleConfig, dwa_cfg: DWAConfig):
        self.dwa = DynamicWindowApproach(vehicle_cfg, dwa_cfg)
        self.vcfg = vehicle_cfg
        self.cfg = dwa_cfg
        self._arrived = False
        self._latched_goal = None

    def _lookahead_target(self, state: VehicleState, global_path: GlobalPath):
        pts = global_path.points
        goal = pts[-1].pose
        dist_goal = math.hypot(goal.x - state.pose.x, goal.y - state.pose.y)

        # Reset latch if a new goal was issued
        if self._latched_goal is None or math.hypot(
            goal.x - self._latched_goal.x, goal.y - self._latched_goal.y
        ) > 1e-6:
            self._arrived = False
            self._latched_goal = goal

        if dist_goal < self.cfg.arrival_radius_m:
            self._arrived = True

        if self._arrived:
            return goal, 0.0   # stay stopped; no re-acceleration, no orbit

        # Braking ramp: max speed from which we can still stop comfortably
        # v_allow = sqrt(2 * a_comfort * remaining_distance)
        a_comfort = 2.0
        speed_cap = math.sqrt(2.0 * a_comfort * max(dist_goal - 0.3, 0.0))

        # nearest path index, then walk forward to lookahead distance
        nearest = min(range(len(pts)), key=lambda i: math.hypot(
            pts[i].pose.x - state.pose.x, pts[i].pose.y - state.pose.y))

        acc = 0.0
        target = pts[nearest].pose
        speed = pts[nearest].target_speed
        for i in range(nearest, len(pts) - 1):
            seg = math.hypot(pts[i+1].pose.x - pts[i].pose.x, pts[i+1].pose.y - pts[i].pose.y)
            acc += seg
            target = pts[i+1].pose
            speed = pts[i+1].target_speed
            if acc >= self.cfg.lookahead_m:
                break

        return target, min(speed, speed_cap)

    def plan(self, state, global_path: GlobalPath, costmap: Costmap, obstacles: List[Obstacle]) -> LocalTrajectory:
        target, speed = self._lookahead_target(state, global_path)
        best = self.dwa.plan(state, target, speed, obstacles)

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
            points.append(TrajectoryPoint(
                t=t,
                pose=Pose2D(x=x, y=y, heading=th),
                twist=Twist2D(vx=best.v, vy=0.0, yaw_rate=best.yaw_rate),
                curvature=best.yaw_rate / max(best.v, 0.1),
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