"""Costmap construction."""
import math
from typing import List, Optional

import numpy as np

from src.common.types.base import FrameId, Header, Pose2D
from src.common.types.config import CostmapConfig
from src.common.types.costmap import Costmap
from src.common.types.obstacle import Obstacle


class LocalGridCostmapBuilder:
    """
    Minimal local costmap builder.

    This is not the final production mapper.
    It exists to give the integration layer a valid typed Costmap.
    """

    def __init__(self, cfg: CostmapConfig):
        self.cfg = cfg

    def update(
        self,
        obstacles: List[Obstacle],
        previous_costmap: Optional[Costmap],
        stamp: float,
        ego_pose: Pose2D,
    ) -> Costmap:
        width = int(round(self.cfg.width_m / self.cfg.resolution))
        height = int(round(self.cfg.height_m / self.cfg.resolution))

        origin_x = ego_pose.x - self.cfg.width_m / 2.0
        origin_y = ego_pose.y - self.cfg.height_m / 2.0

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

    def _add_obstacle(
        self,
        data: np.ndarray,
        obs: Obstacle,
        origin_x: float,
        origin_y: float,
    ) -> None:
        height, width = data.shape

        radius = max(obs.length, obs.width) / 2.0 + self.cfg.inflation_radius
        if radius <= 0.0:
            return

        cx = int(round((obs.pose.x - origin_x) / self.cfg.resolution))
        cy = int(round((obs.pose.y - origin_y) / self.cfg.resolution))

        r_cells = int(math.ceil(radius / self.cfg.resolution))

        if (
            cx < -r_cells
            or cy < -r_cells
            or cx >= width + r_cells
            or cy >= height + r_cells
        ):
            return

        min_x = max(0, cx - r_cells)
        max_x = min(width - 1, cx + r_cells)
        min_y = max(0, cy - r_cells)
        max_y = min(height - 1, cy + r_cells)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                dist = math.hypot(x - cx, y - cy) * self.cfg.resolution
                if dist <= radius:
                    cost = 1.0 - (dist / (radius + 1e-6))
                    if cost > data[y, x]:
                        data[y, x] = cost