from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum


class FrameId(str, Enum):
    MAP = "map"
    VEHICLE = "vehicle"
    SENSOR_FRONT = "sensor_front"
    SENSOR_LIDAR = "sensor_lidar"


@dataclass
class Header:
    stamp: float                  # simulation time in seconds
    frame_id: FrameId
    source: str
    seq: int = 0
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Pose2D:
    x: float
    y: float
    heading: float  # radians


@dataclass
class Twist2D:
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0


@dataclass
class Covariance2D:
    xx: float = 0.0
    yy: float = 0.0
    xy: float = 0.0


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi