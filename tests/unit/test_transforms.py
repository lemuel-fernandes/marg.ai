import math
import pytest
import numpy as np

from src.common.coordinates.transforms import TransformTree
from src.common.types.base import FrameId, Header, Pose2D, Twist2D, Covariance2D
from src.common.types.obstacle import Obstacle, ObstacleClass, ObstacleBehavior


def test_vehicle_to_map_transform(dummy_vehicle_state):
    tf = TransformTree()
    tf.update_ego_state(dummy_vehicle_state)

    # Obstacle is 5 meters directly in front of the vehicle (in vehicle frame)
    # Vehicle is at (10, 10) facing +Y (pi/2).
    # So in map frame, the obstacle should be at (10, 15).
    obs_vehicle = Obstacle(
        header=Header(stamp=0.0, frame_id=FrameId.VEHICLE, source="test"),
        track_id=1,
        class_label=ObstacleClass.POTHOLE,
        behavior=ObstacleBehavior.STATIC,
        pose=Pose2D(x=5.0, y=0.0, heading=0.0),
        length=1.0, width=1.0,
        velocity=Twist2D(),
        pose_covariance=Covariance2D(),
        velocity_covariance=Covariance2D(),
        confidence=1.0,
        is_dynamic=False
    )

    map_obstacles = tf.obstacles_to_map([obs_vehicle])
    
    assert len(map_obstacles) == 1
    map_obs = map_obstacles[0]
    
    assert map_obs.header.frame_id == FrameId.MAP
    assert math.isclose(map_obs.pose.x, 10.0, abs_tol=1e-6)
    assert math.isclose(map_obs.pose.y, 15.0, abs_tol=1e-6)
    assert math.isclose(map_obs.pose.heading, np.pi / 2, abs_tol=1e-6)