from dataclasses import replace
from typing import List

import numpy as np

from src.common.types.base import (
    FrameId,
    Header,
    Pose2D,
    Twist2D,
    normalize_angle,
)
from src.common.types.obstacle import Obstacle
from src.common.types.vehicle_state import VehicleState


class TransformError(RuntimeError):
    pass


class TransformTree:
    """
    Minimal deterministic transform manager.

    Production note:
    In a ROS2 system this would be backed by tf2.
    Here we keep it explicit and testable.
    """

    def __init__(self):
        self._ego_state: VehicleState | None = None

    @property
    def ego_state(self) -> VehicleState | None:
        """Latest ego state pushed via update_ego_state (read-only access)."""
        return self._ego_state

    def update_ego_state(self, ego_state: VehicleState) -> None:
        self._ego_state = ego_state

    def obstacles_to_map(self, obstacles: List[Obstacle]) -> List[Obstacle]:
        if self._ego_state is None:
            raise TransformError("Ego state not available for transforms.")

        transformed: List[Obstacle] = []
        for obs in obstacles:
            if obs.header.frame_id == FrameId.MAP:
                transformed.append(obs)
            elif obs.header.frame_id == FrameId.VEHICLE:
                transformed.append(self._obstacle_vehicle_to_map(obs))
            else:
                raise TransformError(
                    f"No transform implemented for frame {obs.header.frame_id}"
                )
        return transformed

    def _obstacle_vehicle_to_map(self, obs: Obstacle) -> Obstacle:
        assert self._ego_state is not None

        ego_pose = self._ego_state.pose
        cos_h = np.cos(ego_pose.heading)
        sin_h = np.sin(ego_pose.heading)

        # Pose transform
        x_map = ego_pose.x + obs.pose.x * cos_h - obs.pose.y * sin_h
        y_map = ego_pose.y + obs.pose.x * sin_h + obs.pose.y * cos_h
        heading_map = normalize_angle(obs.pose.heading + ego_pose.heading)

        # Velocity transform (rotate into map frame)
        vx_map = obs.velocity.vx * cos_h - obs.velocity.vy * sin_h
        vy_map = obs.velocity.vx * sin_h + obs.velocity.vy * cos_h

        new_header = Header(
            stamp=obs.header.stamp,
            frame_id=FrameId.MAP,
            source="transform_tree",
            seq=obs.header.seq,
        )

        return replace(
            obs,
            header=new_header,
            pose=Pose2D(x=x_map, y=y_map, heading=heading_map),
            velocity=Twist2D(vx=vx_map, vy=vy_map, yaw_rate=obs.velocity.yaw_rate),
        )