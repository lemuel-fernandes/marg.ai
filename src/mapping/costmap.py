"""Costmap construction."""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.common.types.base import FrameId, Header, Pose2D
from src.common.types.config import CostmapConfig, PredictionConfig
from src.common.types.costmap import Costmap
from src.common.types.obstacle import Obstacle, is_surface_anomaly


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

    def __init__(self, cfg: CostmapConfig, road_network: Optional[RoadNetwork] = None,
                 prediction_cfg: Optional[PredictionConfig] = None):
        self.cfg = cfg
        self.road_network = road_network
        self.prediction_cfg = prediction_cfg or PredictionConfig()
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

        # Predictive corridors: sweep closing dynamic actors forward and stamp
        # elevated (non-lethal) cost where they will be, so the global planner
        # schedules maneuvers around predicted occupancy — an overtake waits
        # for an oncoming corridor to pass instead of meeting it mid-road.
        self._stamp_prediction_corridors(data, obstacles, origin_x, origin_y)

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

        # Lethal outside the road, plus an on-road cost gradient that grows
        # toward the road edge. A* minimizes path cost, so with a flat free
        # interior it cuts chords across curve insides and the "shortest"
        # route hugs the road boundary — the global reference line then runs
        # on the road rim and drags the local planner out of the corridor
        # with it (observed as the city_roads boundary violation at the
        # double arc). The gradient is shallow through the interior and
        # steepens sharply in the outer 25% of the half-width, so routes
        # stay near the centerline while edge cells remain far below A*'s
        # off-road cost (1.0 -> 3.5x edge multiplier) and below the lethal
        # threshold, keeping legitimate edge maneuvers (squeezing past a
        # parked truck) available when obstacles block the interior.
        frac = np.clip(distance / self.road_network.half_width, 0.0, 1.0)
        road_cost = np.where(
            frac <= 0.75,
            frac * 0.16,
            0.12 + (frac - 0.75) / 0.25 * 0.43,
        )
        return np.where(distance > self.road_network.half_width, 1.0, road_cost).astype(np.float32)

    def _add_obstacle(self, data, obs, origin_x, origin_y):
        # Surface anomalies (potholes, small debris) are non-solid: they are
        # handled exclusively by the local planner's cost terms. Stamping them
        # as lethal here makes A* re-route the global path around them — and
        # since the stamped footprint depends on the perceived anomaly size,
        # the route (and hence the whole downstream plan) becomes sensitive to
        # detection-size noise. Every other module already excludes them.
        if is_surface_anomaly(obs):
            return
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

    def _stamp_prediction_corridors(self, data, obstacles, origin_x, origin_y):
        """Stamp CV-predicted corridors of closing dynamic actors.

        For each dynamic obstacle moving toward the ego region, sweep its
        footprint forward over the horizon and write ``corridor_cost`` into
        the swept cells (never lethal — the local planners must retain the
        freedom to execute an emergency maneuver through a corridor if the
        prediction turns out wrong; the cost only biases route *scheduling*).
        """
        if not obstacles:
            return
        pc = self.prediction_cfg
        height, width = data.shape
        res = self.cfg.resolution
        n_steps = max(1, int(round(pc.horizon_s / pc.step_s)))

        for obs in obstacles:
            if not obs.is_dynamic or is_surface_anomaly(obs):
                continue
            speed = math.hypot(obs.velocity.vx, obs.velocity.vy)
            if speed < pc.min_speed_mps:
                continue
            for k in range(1, n_steps + 1):
                t = k * pc.step_s
                px = obs.pose.x + obs.velocity.vx * t
                py = obs.pose.y + obs.velocity.vy * t
                # Stamp the actor's swept footprint with a small pad.
                extent = max(obs.length, obs.width) / 2.0 + 0.3
                cx = int(round((px - origin_x) / res))
                cy = int(round((py - origin_y) / res))
                r = int(math.ceil(extent / res))
                x0, x1 = max(0, cx - r), min(width - 1, cx + r)
                y0, y1 = max(0, cy - r), min(height - 1, cy + r)
                if x0 > x1 or y0 > y1:
                    continue
                window = data[y0:y1 + 1, x0:x1 + 1]
                np.maximum(window, pc.corridor_cost, out=window)