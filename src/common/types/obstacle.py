from dataclasses import dataclass
from enum import Enum

from .base import Header, Pose2D, Twist2D, Covariance2D


class ObstacleClass(str, Enum):
    UNKNOWN = "unknown"
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    ANIMAL = "animal"
    POTHOLE = "pothole"
    ENCROACHMENT = "encroachment"
    PARKED_VEHICLE = "parked_vehicle"
    TWO_WHEELER = "two_wheeler"


class ObstacleBehavior(str, Enum):
    UNKNOWN = "unknown"
    STATIC = "static"
    CROSSING = "crossing"
    WEAVING = "weaving"
    STOPPING = "stopping"
    PARKED = "parked"
    DRIFTING = "drifting"
    ONCOMING_WRONG_SIDE = "oncoming_wrong_side"


@dataclass
class Obstacle:
    header: Header
    track_id: int
    class_label: ObstacleClass
    behavior: ObstacleBehavior
    pose: Pose2D
    length: float
    width: float
    velocity: Twist2D
    pose_covariance: Covariance2D
    velocity_covariance: Covariance2D
    confidence: float
    is_dynamic: bool