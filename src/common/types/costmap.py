"""Typed costmap message shared by mapping and planning modules."""

import numpy as np
from dataclasses import dataclass

from src.common.types.base import Header


@dataclass
class Costmap:
    """A local occupancy/cost grid expressed in the map frame."""

    header: Header
    data: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int
    inflation_radius: float = 0.0