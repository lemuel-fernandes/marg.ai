"""Dynamic Window Approach."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.common.types.base import Pose2D, normalize_angle
from src.common.types.config import DWAConfig, VehicleConfig
from src.common.types.obstacle import Obstacle
from src.common.types.vehicle_state import VehicleState


@dataclass
class CandidateTrajectory:
    v: float
    yaw_rate: float
    points: List[Tuple[float, float, float, float]]  # x, y, heading, t
    min_clearance: float


class DynamicWindowApproach:
    """Samples (v, yaw_rate) in the dynamic window, rolls out, scores, picks best."""

    def __init__(self, vehicle_cfg: VehicleConfig, dwa_cfg: DWAConfig):
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
        min_clear = float("inf")
        for (x, y, _, _) in pts:
            for obs in obstacles:
                r = max(obs.length, obs.width) / 2.0
                min_clear = min(min_clear, math.hypot(x - obs.pose.x, y - obs.pose.y) - r)
        return min_clear

    def _candidates(self, state: VehicleState, obstacles: List[Obstacle]) -> List[CandidateTrajectory]:
        v_cur = state.twist.vx
        v_lo = max(0.0, v_cur - self.vcfg.max_acceleration * self.cfg.horizon_s)
        v_hi = min(self.vcfg.max_speed, v_cur + self.vcfg.max_acceleration * self.cfg.horizon_s)

        candidates: List[CandidateTrajectory] = []
        for i in range(self.cfg.v_samples):
            v = v_lo + (v_hi - v_lo) * i / max(1, self.cfg.v_samples - 1)
            for j in range(self.cfg.yaw_rate_samples):
                wr = -self.cfg.max_yaw_rate + 2 * self.cfg.max_yaw_rate * j / max(1, self.cfg.yaw_rate_samples - 1)
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
        if not candidates:
            return None

        # Arrival mode: the planner is asking us to stop (target_speed ~ 0).
        arrival_mode = target_speed < 0.3

        heading_raw, clear_raw, vel_raw = [], [], []
        for c in candidates:
            fx, fy, fth, _ = c.points[-1]
            desired = math.atan2(target_pose.y - fy, target_pose.x - fx)
            heading_raw.append(math.cos(normalize_angle(desired - fth)))
            clear_raw.append(min(c.min_clearance, self.cfg.clearance_cap_m) / self.cfg.clearance_cap_m)

            if arrival_mode:
                # Arrival mode: reward SLOW candidates instead of fast ones.
                # v=0 scores 1.0, v>=1.0 scores 0.0.
                vel_raw.append(max(0.0, 1.0 - c.v / 1.0))
            else:
                # Cruising mode: reward matching the requested target speed.
                vel_raw.append(min(c.v, target_speed) / target_speed)

        def norm(vals):
            lo, hi = min(vals), max(vals)
            return [1.0 if hi - lo < 1e-9 else (v - lo) / (hi - lo) for v in vals]

        hn, cn, vn = norm(heading_raw), norm(clear_raw), norm(vel_raw)

        scores = []
        for i, c in enumerate(candidates):
            score = (
                self.cfg.heading_weight * hn[i]
                + self.cfg.clearance_weight * cn[i]
                + self.cfg.velocity_weight * vn[i]
            )
            # Progress bias: standing still is only acceptable if everything
            # moving is significantly worse (i.e., genuinely blocked).
            # Disabled in arrival mode, where stopping IS the goal.
            if c.v < 0.3 and not arrival_mode:
                score -= self.cfg.stall_penalty
            scores.append(score)

        best_idx = max(range(len(candidates)), key=lambda i: scores[i])
        return candidates[best_idx]