"""Measure ego-vs-ped53 clearance and lateral offsets in GT vs vision runs."""
import sys
sys.path.insert(0, ".")

import math

from src.sim.scenario_registry import REGISTRY
from src.sim.executor import ScenarioExecutor
from src.sim.scenarios.base import min_obstacle_clearance
from src.common.utils.geometry import oriented_rect_clearance


def run(vision):
    sc = REGISTRY["city_roads"]()
    sc.use_vision_perception = vision
    ex = ScenarioExecutor(sc)
    ex.run()
    return ex, sc


for vision in (False, True):
    ex, sc = run(vision)
    log = ex.last_run_log
    worst = (1e9, None)
    rows = []
    for t, st in zip(log.times, log.states):
        if 22.5 <= t <= 26.0:
            peds = [o for o in sc.obstacles_at(t) if o.track_id == 53]
            if not peds:
                continue
            p = peds[0]
            surf = oriented_rect_clearance(st.pose.x, st.pose.y,
                                           p.pose.x, p.pose.y, p.pose.heading,
                                           p.length, p.width) - 0.95
            if surf < worst[0]:
                worst = (surf, t)
            if abs(t - round(t, 1)) < 1e-6:
                rows.append((t, st.pose.x, st.pose.y, st.twist.vx, p.pose.x, surf))
    print(f"\n===== {'VISION' if vision else 'GT'} ===== min ped53 surface clearance "
          f"{worst[0]:.2f} m at t={worst[1]:.1f}")
    print(f"{'t':>5} {'ego_x':>6} {'ego_y':>6} {'v':>5} {'ped_x':>6} {'surf':>6}")
    for t, x, y, v, px, surf in rows:
        print(f"{t:5.1f} {x:6.2f} {y:6.2f} {v:5.2f} {px:6.2f} {surf:6.2f}")
