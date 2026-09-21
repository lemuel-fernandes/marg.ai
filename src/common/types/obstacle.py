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


def is_surface_anomaly(obs: "Obstacle") -> bool:
    """Road-surface anomalies (potholes / small static debris) are non-solid.

    Single source of truth for the predicate used across the stack: they are
    handled by planner cost terms ONLY — never by solid-obstacle geometry
    (global-route invalidation, safety-monitor collision checks, DWA veto
    margins). A pothole must never block a route or trip an emergency stop;
    it only penalizes trajectories that would drive over it.
    """
    return obs.class_label == ObstacleClass.POTHOLE or (
        obs.class_label == ObstacleClass.UNKNOWN
        and not obs.is_dynamic
        and obs.length <= 1.0
    )