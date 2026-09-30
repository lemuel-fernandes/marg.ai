from dataclasses import dataclass
from enum import Enum

from .base import Header


class ControlMode(str, Enum):
    NORMAL = "normal"
    EMERGENCY_STOP = "emergency_stop"
    PARK = "park"


@dataclass
class ControlCommand:
    header: Header
    steering_angle: float
    steering_rate: float
    acceleration: float
    brake: float
    mode: ControlMode