"""Shared obstacle geometry. ALL clearance math in the repo goes through here."""
import math


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