"""Shared obstacle geometry. ALL clearance math in the repo goes through here."""
import math
from typing import List, Sequence, Tuple

import numpy as np


def oriented_rect_clearance(px: float, py: float,
                            cx: float, cy: float, heading: float,
                            length: float, width: float) -> float:
    """Distance from point (px,py) to the surface of an oriented rectangle.
    Returns 0.0 if the point is inside the footprint."""
    dx = px - cx
    dy = py - cy
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    local_x = dx * cos_h + dy * sin_h     # along obstacle's length axis
    local_y = -dx * sin_h + dy * cos_h    # along obstacle's width axis
    clear_x = abs(local_x) - length / 2.0
    clear_y = abs(local_y) - width / 2.0
    if clear_x <= 0.0 and clear_y <= 0.0:
        return 0.0
    return math.hypot(max(clear_x, 0.0), max(clear_y, 0.0))


def oriented_rect_clearance_batch(px: Sequence[float], py: Sequence[float],
                                  cx: Sequence[float], cy: Sequence[float],
                                  heading: Sequence[float],
                                  length: Sequence[float],
                                  width: Sequence[float]) -> np.ndarray:
    """Vectorized oriented_rect_clearance over broadcastable sequences.

    Exact per-element replication of oriented_rect_clearance (same rotation
    into the obstacle frame, same abs/half-extent subtraction, same
    inside-clip-to-zero rule) so callers can swap the per-pair Python loop
    for one array pass without changing any decision boundary. Pass 2D
    arrays (e.g. (n_pts, n_obs)) for any argument to broadcast.
    """
    px_a = np.asarray(px, dtype=np.float64)
    py_a = np.asarray(py, dtype=np.float64)
    dx = px_a - np.asarray(cx, dtype=np.float64)
    dy = py_a - np.asarray(cy, dtype=np.float64)
    cos_h = np.cos(np.asarray(heading, dtype=np.float64))
    sin_h = np.sin(np.asarray(heading, dtype=np.float64))
    local_x = dx * cos_h + dy * sin_h
    local_y = -dx * sin_h + dy * cos_h
    clear_x = np.abs(local_x) - np.asarray(length, dtype=np.float64) / 2.0
    clear_y = np.abs(local_y) - np.asarray(width, dtype=np.float64) / 2.0
    return np.hypot(np.maximum(clear_x, 0.0), np.maximum(clear_y, 0.0))