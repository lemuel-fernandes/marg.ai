"""Dynamic Window Approach."""

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.common.types.base import Pose2D, normalize_angle
from src.common.types.config import DWAConfig, VehicleConfig
from src.common.types.obstacle import Obstacle
from src.common.types.vehicle_state import VehicleState

DWA_CODE_VERSION = "v8-fix-velocity-plateau"

@dataclass
class CandidateTrajectory:
    v: float
    yaw_rate: float
    points: List[Tuple[float, float, float, float]]  # x, y, heading, t
    min_clearance: float


class DynamicWindowApproach:
    """Samples (v, steer) in the dynamic window, rolls out, scores, picks best."""

    def __init__(self, vehicle_cfg: VehicleConfig, dwa_cfg: DWAConfig):
        if os.environ.get("PATHSENSE_DEBUG"):
            print(f"[CODE] dwa.py loaded: {DWA_CODE_VERSION}")
        self.vcfg = vehicle_cfg
        self.cfg = dwa_cfg

    def _roll_out(self, state: VehicleState, v: float, wr: float) -> List[Tuple[float, float, float, float]]:
        pts = []
        x, y, th = state.pose.x, state.pose.y, state.pose.heading
        n = int(round(self.cfg.horizon_s / self.cfg.dt))
        for i in range(1, n + 1):
            x += v * math.cos(th) * self.cfg.dt
            y += v * math.sin(th) * self.cfg.dt
            th = normalize_angle(th + wr * self.cfg.dt)
            pts.append((x, y, th, i * self.cfg.dt))
        return pts

    def _min_clearance(self, pts, obstacles: List[Obstacle]) -> float:
        """Min distance from rollout points to *predicted* obstacle positions."""
        min_clear = float("inf")
        for (x, y, _, t) in pts:
            for obs in obstacles:
                px = obs.pose.x + obs.velocity.vx * t
                py = obs.pose.y + obs.velocity.vy * t
                r = max(obs.length, obs.width) / 2.0
                min_clear = min(min_clear, math.hypot(x - px, y - py) - r)
        return min_clear

    def _candidates(self, state: VehicleState, obstacles: List[Obstacle]) -> List[CandidateTrajectory]:
        v_cur = state.twist.vx
        v_lo = max(0.0, v_cur - self.vcfg.max_acceleration * self.cfg.horizon_s)
        v_hi = min(self.vcfg.max_speed, v_cur + self.vcfg.max_acceleration * self.cfg.horizon_s)

        candidates: List[CandidateTrajectory] = []
        for i in range(self.cfg.v_samples):
            v = v_lo + (v_hi - v_lo) * i / max(1, self.cfg.v_samples - 1)
            for j in range(self.cfg.yaw_rate_samples):
                steer = -self.vcfg.max_steer_angle + 2 * self.vcfg.max_steer_angle * j / max(1, self.cfg.yaw_rate_samples - 1)
                wr = v * math.tan(steer) / self.vcfg.wheelbase

                # Kill loop candidates — yaw rate must stay inside the dynamic window
                if abs(wr) > self.cfg.max_yaw_rate:
                    continue

                pts = self._roll_out(state, v, wr)
                clear = self._min_clearance(pts, obstacles)
                if clear < self.cfg.obstacle_margin:
                    continue  # collision candidate: reject
                candidates.append(CandidateTrajectory(v=v, yaw_rate=wr, points=pts, min_clearance=clear))

        # Always offer a full stop as a safe fallback candidate
        stop_pts = self._roll_out(state, 0.0, 0.0)
        stop_clear = self._min_clearance(stop_pts, obstacles)
        if stop_clear >= self.cfg.obstacle_margin:
            candidates.append(CandidateTrajectory(v=0.0, yaw_rate=0.0, points=stop_pts, min_clearance=stop_clear))

        return candidates

    def plan(
        self,
        state: VehicleState,
        target_pose: Pose2D,
        target_speed: float,
        obstacles: List[Obstacle],
    ) -> Optional[CandidateTrajectory]:
        candidates = self._candidates(state, obstacles)
        self.last_candidates = candidates 
        if not candidates:
            return None

        arrival_mode = target_speed < 0.3

        # Heading score: alignment of the rollout's FINAL heading with the
        # direction from the CURRENT state to the target.
        desired_start = math.atan2(target_pose.y - state.pose.y, target_pose.x - state.pose.x)

        heading_raw, goal_raw, clear_raw, vel_raw = [], [], [], []
        for c in candidates:
            fx, fy, fth, _ = c.points[-1]
            heading_raw.append(math.cos(normalize_angle(desired_start - fth)))
            goal_raw.append(-math.hypot(target_pose.x - fx, target_pose.y - fy))
            clear_raw.append(min(c.min_clearance, self.cfg.clearance_cap_m) / self.cfg.clearance_cap_m)

            if arrival_mode:
                # Reward stopping
                vel_raw.append(max(0.0, 1.0 - c.v / 1.0))
            else:
                # FIX: Strictly penalize exceeding target_speed to enforce braking ramps
                if c.v > target_speed + 1e-3:
                    overshoot = c.v - target_speed
                    max_overshoot = self.vcfg.max_speed - target_speed
                    vel_raw.append(max(0.0, 1.0 - overshoot / (max_overshoot + 1e-6)))
                else:
                    # Reward matching target_speed
                    vel_raw.append(c.v / (target_speed + 1e-6))

        def norm(vals):
            lo, hi = min(vals), max(vals)
            return [1.0 if hi - lo < 1e-9 else (v - lo) / (hi - lo) for v in vals]

        hn, gn, cn, vn = (norm(heading_raw), norm(goal_raw),
                  norm(clear_raw), norm(vel_raw))

        scores = []
        for i, c in enumerate(candidates):
            score = (
                self.cfg.heading_weight * hn[i]
                + self.cfg.goal_weight * gn[i]
                + self.cfg.clearance_weight * cn[i]
                + self.cfg.velocity_weight * vn[i]
            )
            if c.v < 0.3 and not arrival_mode:
                score -= self.cfg.stall_penalty
            scores.append(score)

        best_idx = max(range(len(candidates)), key=lambda i: scores[i])
        
        if os.environ.get("PATHSENSE_DEBUG"):
            order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:3]
            for k in order:
                c = candidates[k]
                print(f"    [DWA] v={c.v:.2f} wr={c.yaw_rate:+.2f} head={hn[k]:+.2f} "
                      f"clear={cn[k]:.2f} vel={vn[k]:.2f} score={scores[k]:+.2f}")
            print(f"    [DWA] state=({state.pose.x:.1f},{state.pose.y:.1f}) "
                  f"hdg={math.degrees(state.pose.heading):+.0f}deg "
                  f"target=({target_pose.x:.1f},{target_pose.y:.1f}) desired={math.degrees(desired_start):+.0f}deg "
                  f"target_speed={target_speed:.2f}")
                  
        return candidates[best_idx]