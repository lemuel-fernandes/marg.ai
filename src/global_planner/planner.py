"""Global planner facade."""


import math
from typing import List, Optional, Tuple

from src.common.types.base import FrameId, Header, Pose2D
from src.common.types.costmap import Costmap
from src.common.types.path import GlobalPath, PathPoint
from src.integration.contracts import GlobalPlanner
from .algorithms.astar import AStarPlanner


class AStarGlobalPlanner(GlobalPlanner):
    """
    Adapter that bridges the typed Costmap/Pose2D contracts with the core A* grid solver.
    """

    def __init__(self, target_speed: float = 5.0, path_downsample_dist: float = 1.0):
        self.target_speed = target_speed
        self.path_downsample_dist = path_downsample_dist

    def _meters_to_grid(self, pose: Pose2D, costmap: Costmap) -> Optional[Tuple[int, int]]:
        col = int(round((pose.x - costmap.origin_x) / costmap.resolution))
        row = int(round((pose.y - costmap.origin_y) / costmap.resolution))
        
        if 0 <= row < costmap.height and 0 <= col < costmap.width:
            return (row, col)
        return None

    def _grid_to_meters(self, rc: Tuple[int, int], costmap: Costmap) -> Pose2D:
        row, col = rc
        x = costmap.origin_x + (col * costmap.resolution)
        y = costmap.origin_y + (row * costmap.resolution)
        return Pose2D(x=x, y=y, heading=0.0)

    def _downsample_path(self, grid_path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Simple distance-based downsampling to prevent thousands of micro-waypoints."""
        if len(grid_path) <= 2:
            return grid_path
            
        downsampled = [grid_path[0]]
        min_dist_sq = (self.path_downsample_dist / 1.0) ** 2  # rough grid units
        
        for i in range(1, len(grid_path) - 1):
            last = downsampled[-1]
            curr = grid_path[i]
            dist_sq = (curr[0] - last[0])**2 + (curr[1] - last[1])**2
            if dist_sq >= min_dist_sq:
                downsampled.append(curr)
                
        downsampled.append(grid_path[-1]) # Always keep goal
        return downsampled

    def plan(self, costmap: Costmap, start: Pose2D, goal: Pose2D) -> GlobalPath:
        header = Header(
            stamp=costmap.header.stamp,
            frame_id=FrameId.MAP,
            source="astar_global_planner"
        )

        start_rc = self._meters_to_grid(start, costmap)
        goal_rc = self._meters_to_grid(goal, costmap)

        if start_rc is None or goal_rc is None:
            return GlobalPath(
                header=header, points=[], length_m=0.0,
                is_feasible=False, replan_required=True,
                reason="Start or Goal is outside costmap bounds"
            )

        # Run A*
        astar = AStarPlanner(grid=costmap.data, resolution=costmap.resolution)
        grid_path = astar.plan(start_rc, goal_rc)

        if grid_path is None:
            return GlobalPath(
                header=header, points=[], length_m=0.0,
                is_feasible=False, replan_required=True,
                reason="No valid path found (Goal blocked by obstacles)"
            )

        # Downsample and convert back to meters
        grid_path = self._downsample_path(grid_path)
        path_points: List[PathPoint] = []
        total_length = 0.0

        for i, rc in enumerate(grid_path):
            pose = self._grid_to_meters(rc, costmap)
            
            # Calculate heading based on next point
            if i < len(grid_path) - 1:
                next_rc = grid_path[i+1]
                next_pose = self._grid_to_meters(next_rc, costmap)
                dx = next_pose.x - pose.x
                dy = next_pose.y - pose.y
                pose.heading = math.atan2(dy, dx)
                total_length += math.hypot(dx, dy)
            elif i > 0:
                pose.heading = path_points[-1].pose.heading

            path_points.append(PathPoint(
                pose=pose,
                curvature=0.0,  # Global planner doesn't strictly compute curvature
                target_speed=self.target_speed
            ))

        return GlobalPath(
            header=header,
            points=path_points,
            length_m=total_length,
            is_feasible=True,
            replan_required=False,
            reason="Path found successfully"
        )