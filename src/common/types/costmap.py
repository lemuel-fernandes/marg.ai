from dataclasses import dataclass

import numpy as np

from .base import Header


@dataclass
class Costmap:
    header: Header
    data: np.ndarray       # 2D cost grid: 0.0 free, 1.0 lethal
    resolution: float      # meters / cell
    origin_x: float
    origin_y: float
    width: int
    height: int
    inflation_radius: float