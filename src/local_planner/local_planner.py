import math
import os
from typing import List, Optional

from src.common.types.base import FrameId, Header, Pose2D, Twist2D
from src.common.types.config import DWAConfig, VehicleConfig
from src.common.types.costmap import Costmap
from src.common.types.obstacle import Obstacle, is_surface_anomaly
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

        # FIX: Final-Approach Creep Mode (< 14 m)
        # Aim directly at the goal and cap speed to a creep profile. The
        # radius must cover the full comfortable-braking distance from cruise
        # speed (the DWA window can only decelerate ~2 m/s^2, so a cap that
        # engages inside the stopping distance cannot prevent overshoot).
        if dist_goal < 14.0:
            # Brake against the distance available in the vehicle's current
            # heading, not just Euclidean distance to the goal. Use the
            # comfortable decel (1.2 m/s^2) for the envelope, not max braking:
            # a max-brake envelope releases the cap until ~0.5 m from the goal,
            # which the controller cannot track and the vehicle overshoots.
            braking_speed = math.sqrt(
                2.0 * 1.2
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

        a_comfort = 1.2
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

    def _corridor_speed_cap(self, state: VehicleState, obstacles: List[Obstacle]) -> Optional[float]:
        """Oncoming-corridor guard for the overtake decision.

        While the ego is laterally offset into the oncoming lane (an overtake
        in progress), an approaching oncoming actor's CV corridor would cross
        the ego within the DWA horizon. Pulling out into that corridor is a
        scheduling error no amount of local reactivity can repair (the DWA
        horizon is shorter than the closing geometry, so the "wait" answer
        must happen BEFORE committing to the wrong lane). Cap the target
        speed to stay behind the truck until the corridor is clear.
        """
        vx = state.twist.vx
        if vx < 0.5:
            return None
        c, s = math.cos(state.pose.heading), math.sin(state.pose.heading)
        horizon_s = self.cfg.horizon_s
        for obs in obstacles:
            if not obs.is_dynamic or is_surface_anomaly(obs):
                continue
            dx = obs.pose.x - state.pose.x
            dy = obs.pose.y - state.pose.y
            fwd = dx * c + dy * s
            lat = -dx * s + dy * c
            # Oncoming: significant closing velocity in the ego-forward axis.
            rel_f = vx - (obs.velocity.vx * c + obs.velocity.vy * s)
            if fwd <= 0.0 or rel_f <= 1.0:
                continue
            # Where is the actor when it reaches the ego's lateral line?
            # Use the actor's own motion only (oncoming actors don't steer
            # around us here; conservative for head-on geometry).
            t_meet = fwd / rel_f
            if t_meet > horizon_s + 1.0:
                continue
            # Actor must also actually occupy the ego's lateral line region
            # as it passes (it will cross our lane).
            lat_at_meet = lat + (-obs.velocity.vx * s + obs.velocity.vy * c) * t_meet
            if abs(lat_at_meet) > 1.5:
                continue
            # Meeting within the horizon on our lane: don't be in the
            # oncoming lane when it arrives — cap speed low so the overtake
            # is aborted/deferred (stay behind the truck).
            return 0.0
        return None

    def plan(self, state, global_path: GlobalPath, costmap: Costmap, obstacles: List[Obstacle]) -> LocalTrajectory:
        target, speed = self._lookahead_target(state, global_path)
        corridor_cap = self._corridor_speed_cap(state, obstacles)
        if corridor_cap is not None:
            speed = min(speed, corridor_cap)
        best = self.dwa.plan(state, target, speed, obstacles, costmap=costmap)

        if best is None:
            # No DWA candidate survived the margin filter (e.g. transient
            # yield under time pressure). Return a controlled strong-brake
            # ramp: the safety monitor space-time-validates it like any other
            # trajectory, so a genuinely unsafe ramp still escalates to a
            # true emergency stop, while a safe ramp executes as a normal
            # (non-emergency) harsh stop.
            return self._strong_brake(state)

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

    def _strong_brake(self, state: VehicleState) -> LocalTrajectory:
        """Controlled strong-brake ramp along the current heading.

        Unlike `emergency_stop`, this is NOT flagged fallback_active: the
        safety monitor space-time-validates it and the controller executes a
        normal (harsh) stop instead of latching emergency-stop mode.
        """
        dt = 0.1
        v0 = max(state.twist.vx, 0.0)
        a_brake = 2.0  # strong but within the comfort envelope
        n = int(math.ceil(v0 / a_brake / dt)) + 2
        x, y, th = state.pose.x, state.pose.y, state.pose.heading
        ux, uy = math.cos(th), math.sin(th)
        points: List[TrajectoryPoint] = []
        s = 0.0
        next_v = v0
        for i in range(n + 1):
            v = next_v
            next_v = max(v0 - a_brake * (i + 1) * dt, 0.0)
            points.append(TrajectoryPoint(
                t=i * dt,
                pose=Pose2D(x=x + s * ux, y=y + s * uy, heading=th),
                twist=Twist2D(vx=v, vy=0.0, yaw_rate=0.0),
                curvature=0.0,
                acceleration=(v - next_v) / dt if v > 1e-3 else 0.0,
            ))
            s += v * dt
        return LocalTrajectory(
            header=Header(state.header.stamp, FrameId.MAP, "dwa_local_planner"),
            points=points,
            cost=0.0,
            is_safe=True,
            fallback_active=False,
            reason="No viable candidate, strong brake",
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