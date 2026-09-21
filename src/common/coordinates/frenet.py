import math
from typing import List, Tuple, Optional
import numpy as np


class CartesianFrenetConverter:
    """
    Transforms points and trajectories between Cartesian (x, y, vx, vy)
    and Frenet (s, d, s_dot, d_dot) coordinates relative to a continuous 2D reference line.
    """

    def __init__(self, waypoints: np.ndarray):
        """
        waypoints: Nx2 or Nx3 array [[x, y] or [x, y, heading]]
        """
        if len(waypoints) < 2:
            raise ValueError("Reference line requires at least 2 waypoints")

        self.waypoints = np.array(waypoints, dtype=np.float64)
        self.pts = self.waypoints[:, :2]

        # Precompute segment lengths and cumulative arc-lengths s
        diffs = np.diff(self.pts, axis=0)
        self.seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        # Avoid zero-length segments
        self.seg_lengths = np.maximum(self.seg_lengths, 1e-6)

        self.s_vals = np.zeros(len(self.pts), dtype=np.float64)
        self.s_vals[1:] = np.cumsum(self.seg_lengths)
        self.total_length = float(self.s_vals[-1])

        # Precompute segment headings
        self.seg_headings = np.arctan2(diffs[:, 1], diffs[:, 0])

    def find_nearest_index(self, x: float, y: float) -> int:
        dists_sq = (self.pts[:, 0] - x) ** 2 + (self.pts[:, 1] - y) ** 2
        return int(np.argmin(dists_sq))

    def to_frenet(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> Tuple[float, float, float, float]:
        """
        Maps a Cartesian position (x, y) and velocity (vx, vy) to Frenet (s, d, s_dot, d_dot).
        d is positive to the left of reference line, negative to the right.
        """
        idx = self.find_nearest_index(x, y)
        
        # Consider segments touching idx
        cand_segs = []
        if idx > 0:
            cand_segs.append(idx - 1)
        if idx < len(self.seg_lengths):
            cand_segs.append(idx)

        best_dist = float("inf")
        best_s = self.s_vals[idx]
        best_d = 0.0
        best_r_theta = self.seg_headings[cand_segs[0]] if cand_segs else 0.0

        for s_idx in cand_segs:
            p1 = self.pts[s_idx]
            p2 = self.pts[s_idx + 1]
            seg_len = self.seg_lengths[s_idx]
            r_theta = self.seg_headings[s_idx]

            dx_seg = p2[0] - p1[0]
            dy_seg = p2[1] - p1[1]

            # Vector from p1 to (x, y)
            dx = x - p1[0]
            dy = y - p1[1]

            # Along segment projection
            proj = (dx * dx_seg + dy * dy_seg) / (seg_len * seg_len)
            proj = max(0.0, min(1.0, proj))

            proj_x = p1[0] + proj * dx_seg
            proj_y = p1[1] + proj * dy_seg

            dist = math.hypot(x - proj_x, y - proj_y)
            if dist < best_dist:
                best_dist = dist
                best_s = self.s_vals[s_idx] + proj * seg_len
                best_r_theta = r_theta
                # Lateral offset d = - (x - proj_x) * sin(th) + (y - proj_y) * cos(th)
                best_d = -(x - proj_x) * math.sin(r_theta) + (y - proj_y) * math.cos(r_theta)

        # Velocity projection
        cos_t = math.cos(best_r_theta)
        sin_t = math.sin(best_r_theta)
        s_dot = vx * cos_t + vy * sin_t
        d_dot = -vx * sin_t + vy * cos_t

        return best_s, best_d, s_dot, d_dot

    def to_cartesian(self, s: float, d: float) -> Tuple[float, float, float]:
        """
        Maps Frenet (s, d) to Cartesian (x, y, heading).
        """
        s = max(0.0, min(s, self.total_length))
        idx = int(np.searchsorted(self.s_vals, s))
        if idx == 0:
            s_idx = 0
            t = 0.0
        elif idx >= len(self.pts):
            s_idx = len(self.seg_lengths) - 1
            t = 1.0
        else:
            s_idx = idx - 1
            seg_s0 = self.s_vals[s_idx]
            seg_len = self.seg_lengths[s_idx]
            t = (s - seg_s0) / seg_len

        p1 = self.pts[s_idx]
        p2 = self.pts[s_idx + 1]
        r_theta = self.seg_headings[s_idx]

        rx = p1[0] + t * (p2[0] - p1[0])
        ry = p1[1] + t * (p2[1] - p1[1])

        # Shift laterally by d perpendicular to heading
        # left is (-sin, cos)
        x = rx - d * math.sin(r_theta)
        y = ry + d * math.cos(r_theta)

        return x, y, r_theta
