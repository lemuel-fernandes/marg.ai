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

        # ---- Hot-loop fast paths (identical semantics, plain-Python lookups).
        # The old inner loop made a _is_valid() call plus numpy scalar reads
        # per neighbor; at millions of neighbor expansions the Python call
        # overhead and np.float64 boxing dominate the whole planner.
        # Precompute once per plan():
        #   validity: bool grid with a False border ring, so one lookup
        #     replaces both _is_valid checks (ring False == out of bounds;
        #     interior False == cell >= lethal_threshold).
        #   base: (1.0 + 2.5 * cell_cost) per cell — loop does one multiply.
        #   g_score: nested Python lists of native floats (no np.float64
        #     boxing on every read/compare).
        base = (1.0 + 2.5 * self.grid).tolist()
        W = self.width
        validity = [[c < self.lethal_threshold for c in row] for row in self.grid.tolist()]
        ring = [False] * (W + 2)
        validity = [ring[:]] + [[False] + row + [False] for row in validity] + [ring[:]]

        # Priority queue: (f_score, counter, row, col)
        # Counter is used to break ties deterministically
        open_set = []
        counter = 0
        heapq.heappush(open_set, (0.0, counter, start_rc[0], start_rc[1]))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # g_score[r][c] = cost of cheapest path from start to (r, c)
        inf = float("inf")
        g_score = [[inf] * W for _ in range(self.height)]
        g_score[start_rc[0]][start_rc[1]] = 0.0

        # 8-way movements: (d_row, d_col, cost_multiplier)
        movements = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), 
            (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))
        ]

        goal_r, goal_c = goal_rc
        # Octile heuristic inlined in the loop (identical math to _heuristic):
        # it runs once per improved neighbor — millions of times on large
        # maps — so the method-call overhead is worth removing.
        k_oct = math.sqrt(2) - 2.0
        while open_set:
            _, _, r, c = heapq.heappop(open_set)

            if r == goal_r and c == goal_c:
                return self._reconstruct_path(came_from, (r, c))

            # g_score[r][c] is constant while (r, c) is expanded: only
            # neighbors' scores are written, never the expanded cell's own.
            g_cur = g_score[r][c]
            pr, pc = r + 1, c + 1  # (r, c) coordinates in the padded grid

            for dr, dc, move_cost in movements:
                nr = r + dr
                nc = c + dc

                # Single lookup replaces _is_valid(nr, nc): the padded ring
                # is False (out of bounds) and interior cells hold
                # (grid < lethal_threshold) — exactly _is_valid's two checks.
                if not validity[pr + dr][pc + dc]:
                    continue

                # Base movement cost + costmap penalty (inflation layer)
                edge_cost = move_cost * base[nr][nc]

                tentative_g = g_cur + edge_cost

                if tentative_g < g_score[nr][nc]:
                    came_from[(nr, nc)] = (r, c)
                    g_score[nr][nc] = tentative_g
                    h_dx = nr - goal_r
                    h_dy = nc - goal_c
                    if h_dx < 0:
                        h_dx = -h_dx
                    if h_dy < 0:
                        h_dy = -h_dy
                    h = (h_dx + h_dy) + k_oct * (h_dx if h_dx < h_dy else h_dy)
                    f_score = tentative_g + weight * h
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