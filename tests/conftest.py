import pytest
import numpy as np

from src.common.types.base import FrameId, Header, Pose2D, Twist2D, Covariance2D
from src.common.types.config import VehicleConfig, CostmapConfig, PlannerConfig
from src.common.types.vehicle_state import VehicleState
from src.common.types.obstacle import Obstacle, ObstacleClass, ObstacleBehavior
from src.common.types.sensor import SensorFrame
from src.common.types.trajectory import LocalTrajectory, TrajectoryPoint
from src.common.types.control import ControlCommand, ControlMode


@pytest.fixture
def vehicle_cfg():
    return VehicleConfig(
        max_speed=10.0,
        max_acceleration=2.0,
        min_acceleration=-3.0,
        max_steer_angle=0.5,
        max_steer_rate=0.5,
        wheelbase=2.5
    )

@pytest.fixture
def dummy_vehicle_state():
    return VehicleState(
        header=Header(stamp=0.0, frame_id=FrameId.MAP, source="test"),
        pose=Pose2D(x=10.0, y=10.0, heading=np.pi / 2),  # Facing +Y
        twist=Twist2D(vx=5.0, vy=0.0, yaw_rate=0.0),
        steering_angle=0.0,
        curvature=0.0
    )

@pytest.fixture
def dummy_sensor_frame():
    return SensorFrame(
        header=Header(stamp=0.0, frame_id=FrameId.SENSOR_FRONT, source="test")
    )