"""Check whether the A* global path stays on the road network."""
import numpy as np

from src.common.types.config import CostmapConfig
from src.mapping.costmap import LocalGridCostmapBuilder
from src.common.types.base import Pose2D
from src.global_planner.planner import AStarGlobalPlanner
from src.sim.scenario_registry import REGISTRY
from src.sim.scenarios.base import point_polyline_distance


def main():
    sc = REGISTRY["city_roads"]()
    v_cfg, c_cfg, dwa_cfg = sc.configs()
    rn = sc.road_network()
    builder = LocalGridCostmapBuilder(c_cfg, road_network=rn)

    cm = builder.update(obstacles=[], previous_costmap=None, stamp=0.0,
                        ego_pose=Pose2D(0, 0, 0))

    gp = AStarGlobalPlanner(target_speed=v_cfg.max_speed)
    path = gp.plan(costmap=cm, start=Pose2D(0, 0, 0), goal=Pose2D(80, 44, 0))
    print(f"path points: {len(path.points)}  feasible={path.is_feasible}")

    allowed = rn.half_width - 0.5
    worst = 0.0
    off = []
    for p in path.points:
        d = min(point_polyline_distance(p.pose.x, p.pose.y, poly) for poly in rn.polylines)
        worst = max(worst, d)
        if d > allowed:
            off.append((p.pose.x, p.pose.y, d))
    print(f"max center dist to road: {worst:.2f} (allowed {allowed})")
    print(f"off-corridor points: {len(off)}")
    for x, y, d in off[:15]:
        print(f"  ({x:.1f},{y:.1f}) d={d:.2f}")

    # Also check the static grid at the observed off-road ego position
    for (ex, ey) in [(52.2, 35.3), (46.5, 27.4), (55.5, 37.0)]:
        col = int(round((ex - cm.origin_x) / cm.resolution))
        row = int(round((ey - cm.origin_y) / cm.resolution))
        print(f"static grid at ({ex},{ey}): {cm.data[row, col]:.2f}")


if __name__ == "__main__":
    main()
