"""Costmap construction."""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.common.types.base import FrameId, Header, Pose2D
from src.common.types.config import CostmapConfig
from src.common.types.costmap import Costmap
from src.common.types.obstacle import Obstacle


@dataclass
class RoadNetwork:
    """Centerline polylines and half-width for off-road masking."""

    polylines: List[List[Tuple[float, float]]]
    half_width: float = 3.5


class LocalGridCostmapBuilder:
    """
    Minimal local costmap builder.

    This is not the final production mapper.
    It exists to give the integration layer a valid typed Costmap.
    """

    def __init__(self, cfg: CostmapConfig, road_network: Optional[RoadNetwork] = None):
        self.cfg = cfg
        self.road_network = road_network
        self._static_grid: Optional[np.ndarray] = None
        self._static_origin: Optional[Tuple[float, float]] = None

    def update(
        self,
        obstacles: List[Obstacle],
        previous_costmap: Optional[Costmap],
        stamp: float,
        ego_pose: Pose2D,
    ) -> Costmap:
        width = int(round(self.cfg.width_m / self.cfg.resolution))
        height = int(round(self.cfg.height_m / self.cfg.resolution))

        if self.cfg.fixed_origin is not None:
            origin_x, origin_y = self.cfg.fixed_origin
        else:
            origin_x = ego_pose.x - self.cfg.width_m / 2.0
            origin_y = ego_pose.y - self.cfg.height_m / 2.0

        origin = (origin_x, origin_y)
        if self.road_network is not None:
            if self._static_grid is None or self._static_origin != origin:
                self._static_grid = self._build_static(origin, width, height)
                self._static_origin = origin
            data = self._static_grid.copy()
        else:
            data = np.zeros((height, width), dtype=np.float32)

        for obs in obstacles:
            self._add_obstacle(
                data=data,
                obs=obs,
                origin_x=origin_x,
                origin_y=origin_y,
            )

        header = Header(
            stamp=stamp,
            frame_id=FrameId.MAP,
            source="local_grid_costmap",
        )

        return Costmap(
            header=header,
            data=data,
            resolution=self.cfg.resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            width=width,
            height=height,
            inflation_radius=self.cfg.inflation_radius,
        )

    def _build_static(self, origin, width, height) -> np.ndarray:
        """Mark cells outside the configured road network as lethal."""
        xs = origin[0] + (np.arange(width) + 0.5) * self.cfg.resolution
        ys = origin[1] + (np.arange(height) + 0.5) * self.cfg.resolution
        grid_x, grid_y = np.meshgrid(xs, ys)
        distance = np.full((height, width), np.inf, dtype=np.float32)

        for polyline in self.road_network.polylines:
            for (ax, ay), (bx, by) in zip(polyline[:-1], polyline[1:]):
                abx, aby = bx - ax, by - ay
                length_sq = abx * abx + aby * aby
                if length_sq < 1e-9:
                    continue
                t = np.clip(((grid_x - ax) * abx + (grid_y - ay) * aby) / length_sq, 0.0, 1.0)
                segment_distance = np.hypot(
                    grid_x - (ax + abx * t), grid_y - (ay + aby * t)
                )
                distance = np.minimum(distance, segment_distance)

        return np.where(distance > self.road_network.half_width, 1.0, 0.0).astype(np.float32)

    def _add_obstacle(self, data, obs, origin_x, origin_y):
        height, width = data.shape
        extent = max(obs.length, obs.width) / 2.0 + self.cfg.inflation_radius
        cx = int(round((obs.pose.x - origin_x) / self.cfg.resolution))
        cy = int(round((obs.pose.y - origin_y) / self.cfg.resolution))
        r = int(math.ceil(extent / self.cfg.resolution))
        x0, x1 = max(0, cx - r), min(width - 1, cx + r)
        y0, y1 = max(0, cy - r), min(height - 1, cy + r)
        if x0 > x1 or y0 > y1:
            return
        xs = origin_x + np.arange(x0, x1 + 1) * self.cfg.resolution
        ys = origin_y + np.arange(y0, y1 + 1) * self.cfg.resolution
        X, Y = np.meshgrid(xs, ys)
        dx, dy = X - obs.pose.x, Y - obs.pose.y
        c, s = math.cos(obs.pose.heading), math.sin(obs.pose.heading)
        lx = dx * c + dy * s
        ly = -dx * s + dy * c
        clear_x = np.abs(lx) - obs.length / 2.0
        clear_y = np.abs(ly) - obs.width / 2.0
        inside = (clear_x <= 0) & (clear_y <= 0)
        d = np.where(inside, 0.0, np.hypot(np.maximum(clear_x, 0), np.maximum(clear_y, 0)))
        cost = np.clip(1.0 - d / (self.cfg.inflation_radius + 1e-6), 0.0, 1.0)
        cost[inside] = 1.0
        window = data[y0:y1 + 1, x0:x1 + 1]
        np.maximum(window, cost, out=window)