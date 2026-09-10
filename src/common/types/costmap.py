"""Costmap data types."""


from dataclasses import dataclass
import numpy as np

@dataclass
class Costmap:
    data: np.ndarray      # 2D numpy array of costs (0.0 = free, 1.0 = lethal)
    resolution: float     # meters per pixel
    origin_x: float       # Global X of the bottom-left corner
    origin_y: float       # Global Y of the bottom-left corner
    width: int            # Number of cells in X
    height: int           # Number of cells in Y