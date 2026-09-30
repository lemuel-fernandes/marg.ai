"""Diagnose city_roads: when/where do emergency stops and boundary violations happen?"""
import math

from src.sim.executor import ScenarioExecutor
from src.sim.scenario_registry import REGISTRY
from src.sim.scenarios.city_roads import MAIN, CROSS
from src.sim.scenarios.base import point_polyline_distance


def main():
    sc = REGISTRY["city_roads"]()
    ex = ScenarioExecutor(sc)
    result = ex.run(save_plot=None)

    log = ex.last_run_log
    rn = sc.road_network()
    allowed = rn.half_width - 0.5

    print("\n--- E-stop events ---")
    estop_runs = []
    cur = None
    for t, s, c in zip(log.times, log.states, log.commands):
        if c.mode.value == "emergency_stop":
            if cur is None:
                cur = [t, t, s.pose.x, s.pose.y, s.twist.vx]
            else:
                cur[1] = t
        else:
            if cur is not None:
                estop_runs.append(cur)
                cur = None
    if cur is not None:
        estop_runs.append(cur)
    for t0, t1, x, y, v in estop_runs:
        print(f"  t={t0:.1f}s..{t1:.1f}s  pose=({x:.1f},{y:.1f})  v={v:.1f} m/s")

    print("\n--- Boundary distance over time (worst segments) ---")
    dists = []
    for t, s in zip(log.times, log.states):
        d = min(point_polyline_distance(s.pose.x, s.pose.y, poly) for poly in rn.polylines)
        dists.append((t, s.pose.x, s.pose.y, d))
    # print every tick where d > allowed
    worst = max(dists, key=lambda e: e[3])
    print(f"  worst: t={worst[0]:.1f}s pose=({worst[1]:.1f},{worst[2]:.1f}) dist={worst[3]:.2f} (allowed {allowed})")
    for t, x, y, d in dists:
        if d > allowed + 0.2:
            print(f"  VIOLATION t={t:5.1f}s pose=({x:6.2f},{y:6.2f}) d={d:.2f}")

    print("\n--- Global path corridor check ---")
    gp = log.global_path
    if gp and gp.points:
        gd = [min(point_polyline_distance(p.pose.x, p.pose.y, poly) for poly in rn.polylines)
              for p in gp.points]
        print(f"  points: {len(gp.points)}  max d={max(gd):.2f}  feasible={gp.is_feasible}")
        bad = [(p.pose.x, p.pose.y, d) for p, d in zip(gp.points, gd) if d > allowed]
        print(f"  off-corridor pts: {len(bad)}" + (f"  first={bad[0]}" if bad else ""))
        for x, y, d in bad[:10]:
            print(f"    ({x:.1f},{y:.1f}) d={d:.2f}")
    else:
        print("  no global path logged")

    print("\n--- Speed profile sample (every 2s) ---")
    for t, s in zip(log.times, log.states):
        if abs(t % 2.0) < 1e-6:
            print(f"  t={t:5.1f}s pose=({s.pose.x:6.2f},{s.pose.y:6.2f}) v={s.twist.vx:5.2f}")


if __name__ == "__main__":
    main()
