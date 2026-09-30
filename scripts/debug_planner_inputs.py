"""Diff planner inputs: GT vs vision city_roads around the ped53 encounter.

Logs every AdaptiveFrenetLocalPlanner.plan call in t=[21, 26]: ego state, each
obstacle the planner received (id, class, pose, velocity), and the returned
trajectory reason. Run for both perception modes and diff.
"""
import sys
sys.path.insert(0, ".")

from src.sim.scenario_registry import REGISTRY
from src.sim.executor import ScenarioExecutor
from src.local_planner import frenet_local_planner as flp

orig_plan = flp.AdaptiveFrenetLocalPlanner.plan
rows = []


def patched(self, state, global_path, costmap, obstacles):
    out = orig_plan(self, state, global_path, costmap, obstacles)
    t = state.header.stamp
    if 21.0 <= t <= 26.0:
        obs_desc = "; ".join(
            f"{o.track_id}:{o.class_label.value}@({o.pose.x:.2f},{o.pose.y:.2f})"
            f"v({o.velocity.vx:.2f},{o.velocity.vy:.2f})d{o.is_dynamic:.0f}"
            f"f{o.header.frame_id.name[:3]}"
            for o in obstacles)
        rows.append((t, state.pose.x, state.pose.y, state.twist.vx,
                     out.points[0].twist.vx if out.points else -1,
                     out.reason, obs_desc))
    return out


flp.AdaptiveFrenetLocalPlanner.plan = patched

mode = sys.argv[1] if len(sys.argv) > 1 else "vision"
sc = REGISTRY["city_roads"]()
sc.use_vision_perception = (mode == "vision")
ex = ScenarioExecutor(sc)
result = ex.run()
print(f"\n===== {mode.upper()} ===== pass={result.passed} "
      f"min_ttc={result.metrics.get('min_ttc_s')}")
print(f"{'t':>5} {'ego_x':>6} {'ego_y':>6} {'v':>5} {'v0':>5}  reason | obstacles")
for (t, x, y, v, v0, reason, obs) in rows:
    print(f"{t:5.1f} {x:6.2f} {y:6.2f} {v:5.2f} {v0:5.2f}  {reason[:34]:34s} | {obs}")
