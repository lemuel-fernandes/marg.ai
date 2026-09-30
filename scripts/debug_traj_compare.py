"""Compare GT vs vision city_roads trajectories around the ped encounter."""
import sys
sys.path.insert(0, ".")

import math

from src.sim.scenario_registry import REGISTRY
from src.sim.executor import ScenarioExecutor


def run(vision):
    sc = REGISTRY["city_roads"]()
    sc.use_vision_perception = vision
    ex = ScenarioExecutor(sc)
    result = ex.run()
    return ex, sc, result


for vision in (False, True):
    ex, sc, result = run(vision)
    log = ex.last_run_log
    print(f"\n===== {'VISION' if vision else 'GT'} mode ===== "
          f"pass={result.passed} final={result.metrics.get('final_dist_m', -1):.2f} "
          f"min_ttc={result.metrics.get('min_ttc_s', -1)}")
    print(f"{'t':>5} {'x':>6} {'y':>6} {'hdg':>5} {'v':>5}")
    for t, st in zip(log.times, log.states):
        if abs(t - round(t)) < 1e-6 and 17.0 <= t <= 28.0:
            print(f"{t:5.0f} {st.pose.x:6.2f} {st.pose.y:6.2f} "
                  f"{st.pose.heading:5.2f} {st.twist.vx:5.2f}")
    # ped #53 ground truth at those times
    print("  ped53 x:", [round(38.0 + 1.2 * (t - 16.0), 1) for t in range(17, 27)])
