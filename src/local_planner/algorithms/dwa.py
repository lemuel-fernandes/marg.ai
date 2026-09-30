"""Dynamic Window Approach."""

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.common.types.base import Pose2D, normalize_angle
from src.common.types.config import DWAConfig, VehicleConfig
from src.common.types.obstacle import Obstacle, is_surface_anomaly
from src.common.types.vehicle_state import VehicleState

DWA_CODE_VERSION = "v8-fix-velocity-plateau"

@dataclass
class CandidateTrajectory:
    v: float
    yaw_rate: float
    points: List[Tuple[float, float, float, float]]  # x, y, heading, t
    min_clearance: float
    anomaly_cost: float = 0.0


class DynamicWindowApproach:
    """Samples (v, steer) in the dynamic window, rolls out, scores, picks best."""

    def __init__(self, vehicle_cfg: VehicleConfig, dwa_cfg: DWAConfig):
        if os.environ.get("PATHSENSE_DEBUG"):
            print(f"[CODE] dwa.py loaded: {DWA_CODE_VERSION}")
        self.vcfg = vehicle_cfg
        self.cfg = dwa_cfg

    def _roll_out(self, state: VehicleState, v: float, wr: float) -> np.ndarray:
        """Constant-curvature arc rollout (vectorized, shape (n, 4)):
        columns are x, y, heading, t.

        Closed-form arc sampling (exact for the bicycle model at constant
        (v, yaw_rate)) replaces the per-step Euler integration — identical
        geometry, no Python loop. Returns an ndarray; downstream consumers
        index rows the same way as the old list of tuples.
        """
        n = int(round(self.cfg.horizon_s / self.cfg.dt))
        i = np.arange(1, n + 1, dtype=np.float64)
        t = i * self.cfg.dt
        if abs(wr) < 1e-9:
            x = state.pose.x + v * math.cos(state.pose.heading) * t
            y = state.pose.y + v * math.sin(state.pose.heading) * t
            th = np.full(n, state.pose.heading)
        else:
            th = state.pose.heading + wr * t
            x = state.pose.x + v / wr * (np.sin(th) - math.sin(state.pose.heading))
            y = state.pose.y - v / wr * (np.cos(th) - math.cos(state.pose.heading))
        return np.column_stack((x, y, th, t))

    @staticmethod
    def _prepare_obstacles(obstacles: List[Obstacle]):
        """Precompute per-tick obstacle arrays once (not per rollout).

        Returns (solid_mask, arrays-dict) consumed by _min_clearance.
        solid_mask[i] is True when obstacle i is solid (non-anomaly).
        Building these once per planning tick instead of once per candidate
        rollout removes an O(n_obs) Python loop from every rollout call.
        """
        return (
            np.array([not is_surface_anomaly(o) for o in obstacles]),
            dict(
                ox=np.array([o.pose.x for o in obstacles]),
                oy=np.array([o.pose.y for o in obstacles]),
                ovx=np.array([o.velocity.vx for o in obstacles]),
                ovy=np.array([o.velocity.vy for o in obstacles]),
                oh=np.array([o.pose.heading for o in obstacles]),
                ohl=np.array([o.length / 2.0 for o in obstacles]),
                ohw=np.array([o.width / 2.0 for o in obstacles]),
            ),
        )

    def _min_clearance(self, pts, obstacles: List[Obstacle],
                       bundle=None) -> Tuple[float, float]:
        """Return (min solid clearance, anomaly proximity cost) for the rollout.

        ``bundle`` is the precomputed obstacle layout from
        _prepare_obstacles; callers evaluating many rollouts against the
        same obstacle set (i.e. _candidates) must pass it to avoid
        rebuilding the arrays per rollout.
        """
        if not obstacles or len(pts) == 0:
            return float("inf"), 0.0
        if bundle is None:
            bundle = self._prepare_obstacles(obstacles)
        solid_mask, arrs = bundle

        # Vectorized layout: (n_pts, n_obs) matrices. Forecasted positions use
        # the same per-point time t the rollout carries (dynamic obstacle
        # constant-velocity propagation), identical to the previous loop.
        pts_arr = np.asarray(pts, dtype=np.float64)
        pts_xy = pts_arr[:, :2]                                      # (n, 2)
        t = pts_arr[:, 3]                                            # (n,)
        ox, oy = arrs["ox"], arrs["oy"]
        ovx, ovy = arrs["ovx"], arrs["ovy"]
        oh, ohl, ohw = arrs["oh"], arrs["ohl"], arrs["ohw"]

        # Forecast obstacle positions: (n_pts, n_obs, 2).
        pos = np.stack([ox + ovx * t[:, None], oy + ovy * t[:, None]], axis=-1)
        dx = pts_xy[:, None, 0] - pos[..., 0]
        dy = pts_xy[:, None, 1] - pos[..., 1]
        cos_h = np.cos(oh)[None, :]
        sin_h = np.sin(oh)[None, :]

        # Rotate the query point into the obstacle's local frame: local_x is
        # along the obstacle's heading (its "length" axis), local_y is
        # perpendicular to it (its "width" axis).
        local_x = dx * cos_h + dy * sin_h
        local_y = -dx * sin_h + dy * cos_h
        clear_x = np.abs(local_x) - ohl[None, :]
        clear_y = np.abs(local_y) - ohw[None, :]
        # Query point inside the footprint -> distance 0, else rounded-box dist.
        dist = np.hypot(np.maximum(clear_x, 0.0), np.maximum(clear_y, 0.0))
        dist[(clear_x <= 0.0) & (clear_y <= 0.0)] = 0.0

        # Surface anomalies (potholes / small static debris) are NOT solid:
        # they never enter the veto/margin channel (a hard margin against a
        # pothole ON the lane center deadlocks the planner — every forward
        # rollout rejected, the stop candidate survives). They are returned
        # separately as a smooth proximity cost the scorer adds so the planner
        # prefers straddling an anomaly over stalling in front of it.
        # (solid_mask comes precomputed in the bundle.)

        # Anomaly influence radius: just inside obstacle_margin so the kernel
        # shapes costs without recreating a de-facto veto ring.
        anomaly_radius = max(0.5, self.cfg.obstacle_margin - 0.5)
        if solid_mask.all():
            return float(dist.min()) if dist.size else float("inf"), 0.0

        # d for anomalies: distance to their forecasted center.
        d_anom = np.hypot(dx, dy)[:, ~solid_mask]
        anom_cost = float(np.sum(
            np.clip(1.0 - d_anom / anomaly_radius, 0.0, 1.0)))
        if solid_mask.any():
            return float(dist[:, solid_mask].min()), anom_cost
        return float("inf"), anom_cost

    def _candidates(self, state: VehicleState, obstacles: List[Obstacle], costmap=None,
                    v_cap: Optional[float] = None) -> List[CandidateTrajectory]:
        v_cur = state.twist.vx
        v_lo = max(0.0, v_cur - self.vcfg.max_acceleration * self.cfg.dt)
        v_hi = min(self.vcfg.max_speed, v_cur + self.vcfg.max_acceleration * self.cfg.dt)
        if v_cap is not None and v_cap < v_hi:
            # Hard clamp: the planner-provided cap (brake envelope / creep
            # profile) constrains the window. If the vehicle is above the cap
            # and cannot reach it within one step, keep the floor at v_lo so
            # the window still allows full-rate deceleration.
            v_hi = max(v_cap, v_lo)

        # ---- Sample (v, steer) exactly as before (same order/tie-breaks) ----
        n_v = self.cfg.v_samples
        n_w = self.cfg.yaw_rate_samples
        vs, wrs = [], []
        for i in range(n_v):
            v = v_lo + (v_hi - v_lo) * i / max(1, n_v - 1)
            for j in range(n_w):
                steer = -self.vcfg.max_steer_angle \
                    + 2 * self.vcfg.max_steer_angle * j / max(1, n_w - 1)
                wr = v * math.tan(steer) / self.vcfg.wheelbase
                # Kill loop candidates — yaw rate must stay inside the
                # dynamic window (same rejection as the per-rollout loop).
                if abs(wr) > self.cfg.max_yaw_rate:
                    continue
                vs.append(v)
                wrs.append(wr)
        if not vs:
            vs, wrs = [0.0], [0.0]  # degenerate window: evaluate the stop arc

        # ---- Batch rollouts: every candidate in ONE set of array ops ------
        # Closed-form constant-curvature arcs (identical geometry to the
        # per-candidate Euler loop it replaced — see _roll_out docstring),
        # evaluated for all candidates simultaneously: (m, n, 4) where m =
        # #candidates, n = horizon steps.
        vs_a = np.asarray(vs, dtype=np.float64)                     # (m,)
        wrs_a = np.asarray(wrs, dtype=np.float64)                   # (m,)
        n = int(round(self.cfg.horizon_s / self.cfg.dt))
        i = np.arange(1, n + 1, dtype=np.float64)
        t = i * self.cfg.dt                                         # (n,)
        th0 = state.pose.heading
        th = th0 + wrs_a[:, None] * t[None, :]                      # (m, n)
        straight = np.abs(wrs_a) < 1e-9
        # Straight arcs: th stays th0; v/wr is singular -> closed form via cos/sin.
        # Curved arcs: exact circle sampling around the instant center.
        with np.errstate(divide="ignore", invalid="ignore"):
            k = vs_a[:, None] / wrs_a[:, None]                      # (m, 1)
        x_arc = state.pose.x + k * (np.sin(th) - math.sin(th0))
        y_arc = state.pose.y - k * (np.cos(th) - math.cos(th0))
        x_str = state.pose.x + vs_a[:, None] * math.cos(th0) * t[None, :]
        y_str = state.pose.y + vs_a[:, None] * math.sin(th0) * t[None, :]
        xs = np.where(straight[:, None], x_str, x_arc)
        ys = np.where(straight[:, None], y_str, y_arc)
        ths = np.where(straight[:, None], th0, th)
        ts = np.broadcast_to(t, (len(vs), n))
        all_pts = np.stack([xs, ys, ths, ts], axis=-1)              # (m, n, 4)

        # ---- Costmap veto: one vectorized pass for all candidates ----------
        flat = all_pts.reshape(-1, 4)
        lethal_flat = np.zeros(len(flat), dtype=bool)
        if costmap is not None:
            lethal_threshold = getattr(self.cfg, "lethal_threshold", 0.9)
            col = np.rint((flat[:, 0] - costmap.origin_x) / costmap.resolution).astype(int)
            row = np.rint((flat[:, 1] - costmap.origin_y) / costmap.resolution).astype(int)
            oob = ((row < 0) | (row >= costmap.height)
                   | (col < 0) | (col >= costmap.width))
            lethal_flat[oob] = True  # outside known grid — unsafe, not free
            inb = ~oob
            if inb.any():
                lethal_flat[inb] = costmap.data[row[inb], col[inb]] >= lethal_threshold
        lethal = lethal_flat.reshape(len(vs), n).any(axis=1)

        # ---- Obstacle clearance: one vectorized pass for all candidates ---
        # Same oriented-rectangle math and constant-velocity forecast as
        # _min_clearance, lifted to (m, n) x n_obs.
        if obstacles:
            solid_mask, arrs = self._prepare_obstacles(obstacles)
            ox, oy = arrs["ox"], arrs["oy"]
            ovx, ovy = arrs["ovx"], arrs["ovy"]
            oh, ohl, ohw = arrs["oh"], arrs["ohl"], arrs["ohw"]
            pos = ox + ovx * ts[..., None], oy + ovy * ts[..., None]  # (m, n, n_obs)
            dx = xs[..., None] - pos[0]
            dy = ys[..., None] - pos[1]
            cos_h = np.cos(oh)[None, None, :]
            sin_h = np.sin(oh)[None, None, :]
            local_x = dx * cos_h + dy * sin_h
            local_y = -dx * sin_h + dy * cos_h
            clear_x = np.abs(local_x) - ohl
            clear_y = np.abs(local_y) - ohw
            dist = np.hypot(np.maximum(clear_x, 0.0), np.maximum(clear_y, 0.0))
            dist[(clear_x <= 0.0) & (clear_y <= 0.0)] = 0.0
            # Solid: min over time steps then over obstacles -> (m, n_obs)
            solid_dist = dist[..., solid_mask]
            min_solid = solid_dist.min(axis=1).min(axis=1) if solid_dist.size \
                else np.full(len(vs), np.inf)
            # Anomalies: smooth proximity kernel over the whole rollout,
            # same 1 - d/radius form as _min_clearance (clipped at 0).
            anomaly_radius = max(0.5, self.cfg.obstacle_margin - 0.5)
            if solid_mask.all():
                anom = np.zeros(len(vs))
            else:
                d_anom = np.hypot(dx[..., ~solid_mask], dy[..., ~solid_mask])
                anom = np.clip(1.0 - d_anom / anomaly_radius, 0.0, 1.0).sum(axis=(1, 2))
        else:
            min_solid = np.full(len(vs), np.inf)
            anom = np.zeros(len(vs))

        # ---- Assemble candidates in the original sample order -------------
        candidates: List[CandidateTrajectory] = []
        for m_idx in range(len(vs)):
            if lethal[m_idx] or min_solid[m_idx] < self.cfg.obstacle_margin:
                continue  # collision candidate: reject
            pts = all_pts[m_idx]
            candidates.append(CandidateTrajectory(
                v=float(vs[m_idx]), yaw_rate=float(wrs[m_idx]), points=pts,
                min_clearance=float(min_solid[m_idx]), anomaly_cost=float(anom[m_idx])))

        # Always offer a full stop as a safe fallback candidate
        stop_pts = self._roll_out(state, 0.0, 0.0)
        is_lethal, _ = self._costmap_check(stop_pts, costmap)
        stop_clear, stop_anom = self._min_clearance(stop_pts, obstacles,
                                                    self._prepare_obstacles(obstacles)
                                                    if obstacles else None)
        if not is_lethal and stop_clear >= self.cfg.obstacle_margin:
            candidates.append(CandidateTrajectory(v=0.0, yaw_rate=0.0, points=stop_pts,
                                                  min_clearance=stop_clear, anomaly_cost=stop_anom))
        return candidates
    def _costmap_check(self, pts, costmap) -> Tuple[bool, float]:
        """Returns (is_lethal, max_cost) for a rollout against the road-boundary costmap."""
        if costmap is None:
            return False, 0.0
        lethal_threshold = getattr(self.cfg, "lethal_threshold", 0.9)
        pts_arr = np.asarray(pts, dtype=np.float64)
        col = np.rint((pts_arr[:, 0] - costmap.origin_x) / costmap.resolution).astype(int)
        row = np.rint((pts_arr[:, 1] - costmap.origin_y) / costmap.resolution).astype(int)
        if ((row < 0) | (row >= costmap.height) | (col < 0) | (col >= costmap.width)).any():
            return True, 1.0  # outside known local grid — treat as unsafe, not free
        cell_costs = costmap.data[row, col]
        return bool((cell_costs >= lethal_threshold).any()), float(cell_costs.max())

    def plan(
        self,
        state: VehicleState,
        target_pose: Pose2D,
        target_speed: float,

        obstacles: List[Obstacle],
        costmap: Optional["Costmap"] = None,
    ) -> Optional[CandidateTrajectory]:
        candidates = self._candidates(state, obstacles, costmap, v_cap=target_speed)
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

        # Surface-anomaly proximity cost (potholes etc.): normalized against
        # the rollout length so the term is comparable to the other [0,1]
        # score components regardless of horizon length.
        anom_raw = [c.anomaly_cost / max(len(c.points), 1) for c in candidates]

        hn, gn, cn, vn, an = (norm(heading_raw), norm(goal_raw),
                  norm(clear_raw), norm(vel_raw), norm(anom_raw))

        scores = []
        for i, c in enumerate(candidates):
            score = (
                self.cfg.heading_weight * hn[i]
                + self.cfg.goal_weight * gn[i]
                + self.cfg.clearance_weight * cn[i]
                + self.cfg.velocity_weight * vn[i]
                - self.cfg.clearance_weight * an[i]
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