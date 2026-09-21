"""A* global planning algorithm."""

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


class AStarPlanner:
    """
    Standard A* pathfinding on a 2D occupancy/cost grid.
    Uses 8-way connectivity and penalizes moving through high-cost (inflated) cells.
    """

    def __init__(self, grid: np.ndarray, resolution: float, lethal_threshold: float = 0.9):
        self.grid = grid
        self.height, self.width = grid.shape
        self.resolution = resolution
        self.lethal_threshold = lethal_threshold

    def _is_valid(self, row: int, col: int) -> bool:
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        return self.grid[row, col] < self.lethal_threshold

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        # Octile distance (admissible for 8-way grids)
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

    def plan(
        self, 
        start_rc: Tuple[int, int], 
        goal_rc: Tuple[int, int],
        weight: float = 1.0
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Returns a list of (row, col) tuples from start to goal, or None if no path exists.
        """
        if not self._is_valid(*start_rc) or not self._is_valid(*goal_rc):
            return None

        # Priority queue: (f_score, counter, row, col)
        # Counter is used to break ties deterministically
        open_set = []
        counter = 0
        heapq.heappush(open_set, (0.0, counter, start_rc[0], start_rc[1]))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # g_score[r, c] = cost of cheapest path from start to (r, c)
        g_score = np.full((self.height, self.width), np.inf)
        g_score[start_rc[0], start_rc[1]] = 0.0

        # 8-way movements: (d_row, d_col, cost_multiplier)
        movements = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), 
            (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))
        ]

        while open_set:
            _, _, r, c = heapq.heappop(open_set)
            current = (r, c)

            if current == goal_rc:
                return self._reconstruct_path(came_from, current)

            for dr, dc, move_cost in movements:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)

                if not self._is_valid(nr, nc):
                    continue

                # Base movement cost + costmap penalty
                cell_cost = self.grid[nr, nc]
                # Penalize moving near obstacles (inflation layer)
                edge_cost = move_cost * (1.0 + 2.5 * cell_cost) 
                
                tentative_g = g_score[r, c] + edge_cost

                if tentative_g < g_score[nr, nc]:
                    came_from[neighbor] = current
                    g_score[nr, nc] = tentative_g
                    f_score = tentative_g + weight * self._heuristic(neighbor, goal_rc)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, nr, nc))

        return None  # No path found

    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path