"""Definitive trace: is the live global reference off-road, or is the local trajectory diverging from it?"""
import math

from src.sim.executor import ScenarioExecutor
from src.sim.scenario_registry import REGISTRY
from src.sim.scenarios.base import point_polyline_distance

TRACE = []


def road_d(x, y, polylines):
    return min(point_polyline_distance(x, y, poly) for poly in polylines)


def main():
    sc = REGISTRY["city_roads"]()
    rn = sc.road_network()
    polylines = rn.polylines
    orig_get_lp = sc.get_local_planner

    def traced_get_lp(v_cfg, dwa_cfg):
        lp = orig_get_lp(v_cfg, dwa_cfg)
        orig_plan = lp.plan

        def traced_plan(state, global_path, costmap, obstacles):
            traj = orig_plan(state, global_path, costmap, obstacles)
            try:
                t = state.header.stamp
                ex_d = road_d(state.pose.x, state.pose.y, polylines)
                rec = {"t": t, "ego": (state.pose.x, state.pose.y), "ego_rd": ex_d,
                       "reason": traj.reason, "n_gp": len(global_path.points) if global_path else 0}
                if global_path and global_path.points:
                    # reference points within +-15 m of ego along the polyline
                    near = []
                    for p in global_path.points:
                        d_ego = math.hypot(p.pose.x - state.pose.x, p.pose.y - state.pose.y)
                        if d_ego < 15.0:
                            near.append((p.pose.x, p.pose.y, road_d(p.pose.x, p.pose.y, polylines)))
                    rec["ref_worst_rd"] = max((r[2] for r in near), default=None)
                    rec["ref_n"] = len(near)
                if traj.points:
                    last = traj.points[-1].pose
                    rec["traj_end"] = (last.x, last.y)
                    rec["traj_end_rd"] = road_d(last.x, last.y, polylines)
                    rec["traj_v_end"] = traj.points[-1].twist.vx
                TRACE.append(rec)
            except Exception as e:
                TRACE.append({"err": str(e)})
            return traj

        lp.plan = traced_plan
        return lp

    sc.get_local_planner = traced_get_lp
    ScenarioExecutor(sc).run(save_plot=None)

    print("\n--- ticks t=20..32s ---")
    for r in TRACE:
        if isinstance(r, dict) and "err" not in r and 20.0 <= r["t"] <= 32.0:
            te = r.get("traj_end", (float("nan"), float("nan")))
            print(f"t={r['t']:5.1f} ego=({r['ego'][0]:6.2f},{r['ego'][1]:6.2f}) ego_rd={r['ego_rd']:5.2f} "
                  f"ref_n={r.get('ref_n')} ref_worst_rd={r.get('ref_worst_rd')} "
                  f"traj_end=({te[0]:6.1f},{te[1]:6.1f}) "
                  f"traj_end_rd={r.get('traj_end_rd')} v_end={r.get('traj_v_end')}")
    # Also show a couple of ticks before the exit begins
    print("\n--- ticks t=12..17s ---")
    for r in TRACE:
        if isinstance(r, dict) and "err" not in r and 12.0 <= r["t"] <= 17.0 and abs(r["t"] * 10 - round(r["t"] * 10)) < 1e-6:
            print(f"t={r['t']:5.1f} ego=({r['ego'][0]:6.2f},{r['ego'][1]:6.2f}) ego_rd={r['ego_rd']:5.2f} "
                  f"ref_n={r.get('ref_n')} ref_worst_rd={r.get('ref_worst_rd')} "
                  f"traj_end_rd={r.get('traj_end_rd')} v_end={r.get('traj_v_end')} reason={r['reason'][:40]}")


if __name__ == "__main__":
    main()
