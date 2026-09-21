"""Trace unresponded TTC events in a vision-mode city_roads run."""
import sys
sys.path.insert(0, ".")

from src.sim.scenario_registry import REGISTRY
from src.sim.executor import ScenarioExecutor
from src.sim.metrics import _ttc_events, _responding_mask, _ego_accel_series

sc = REGISTRY["city_roads"]()
ex = ScenarioExecutor(sc)  # vision flag is read from the scenario attribute
result = ex.run()

events = _ttc_events(ex.last_run_log, sc.obstacles_at, ego_half_width=0.95)
accels = _ego_accel_series(ex.last_run_log)
speeds = [st.twist.vx for st in ex.last_run_log.states]
responding = _responding_mask(list(ex.last_run_log.times), accels, speeds)

bad = [(e["ttc"], e["t"], e["i"], e["track_id"], e["clear"])
       for e in events if e["ttc"] > 0 and not responding[e["i"]]]
bad.sort()
print("== unresponded closing events (worst 15) ==")
for ttc, t, i, oid, clr in bad[:15]:
    st = ex.last_run_log.states[i]
    print(f"  ttc={ttc:.3f} t={t:.2f} obs={oid} clr={clr:.2f} ego_v={st.twist.vx:.2f} "
          f"pos=({st.pose.x:.2f},{st.pose.y:.2f})")
if not bad:
    print("  none")
print("metrics:", {k: round(v, 3) for k, v in result.metrics.items() if "ttc" in k})
print("pass:", result.passed, "| final:", result.metrics.get("final_dist_m"))
