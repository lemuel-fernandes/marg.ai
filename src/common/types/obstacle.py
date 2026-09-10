"""Obstacle data types."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ObstacleType(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"

class ObstacleClass(Enum):
    UNKNOWN = "unknown"
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    ANIMAL = "animal"       # Cows, dogs, goats (Crucial for Indian roads)
    POTHOle = "pothole"
    ENCROACHMENT = "encroachment" # Carts, stalls, parked autos

@dataclass
class BoundingBox:
    x: float          # Center X in global frame (meters)
    y: float          # Center Y in global frame (meters)
    width: float      # meters
    height: float     # meters
    heading: float = 0.0  # radians

@dataclass
class Obstacle:
    id: int
    obstacle_type: ObstacleType
    obstacle_class: ObstacleClass
    bbox: BoundingBox
    velocity_x: float = 0.0   # m/s
    velocity_y: float = 0.0   # m/s
    confidence: float = 1.0   # 0.0 to 1.0