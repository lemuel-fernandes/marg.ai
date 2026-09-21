from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .base import Header
from .obstacle import Obstacle


@dataclass
class SensorFrame:
    header: Header
    rgb: Optional[np.ndarray] = None
    lidar: Optional[np.ndarray] = None
    ground_truth_obstacles: List[Obstacle] = field(default_factory=list)