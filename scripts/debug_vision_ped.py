"""Compare vision-perceived vs GT state for city_roads ped #53 during the t=24 pass."""
import sys
sys.path.insert(0, ".")

import math

from src.sim.scenario_registry import REGISTRY
from src.sim.executor import ScenarioExecutor
from src.perception import vision_node as vn

sc = REGISTRY["city_roads"]()

captured = []
orig_process = vn.VisionPerceptionNode.process

def patched(self, sensor_frame):
    out = orig_process(self, sensor_frame)
    t = sensor_frame.header.stamp
    if 23.5 <= t <= 24.6:
        # Compare in MAP frame: transform VEHICLE-frame vision output with the
        # current ego state (same transform the pipeline applies downstream).
        for o in self.tf.obstacles_to_map(
                [o for o in out.obstacles if o.track_id == 53]):
            captured.append((t, o.pose.x, o.pose.y, o.velocity.vx, o.velocity.vy,
                             o.header.frame_id.name))
    return out

vn.VisionPerceptionNode.process = patched

ex = ScenarioExecutor(sc)
result = ex.run()

print("\n== perceived ped #53 (vision output, transformed to map frame) ==")
gt = {round(t, 1): None for t in [23.5 + 0.1 * i for i in range(12)]}
for t in [23.5 + 0.1 * i for i in range(12)]:
    g = [o for o in sc.obstacles_at(round(t, 1)) if o.track_id == 53]
    if g:
        gt[round(t, 1)] = (g[0].pose.x, g[0].pose.y, g[0].velocity.vx)
print(f"{'t':>5} {'perc_x':>7} {'perc_y':>7} {'vx':>5} {'vy':>5} {'frame':>8} | "
      f"{'gt_x':>7} {'gt_y':>7} {'dpos':>6}")
for (t, px, py, vx, vy, fr) in captured:
    g = gt.get(round(t, 1))
    if g:
        d = math.hypot(px - g[0], py - g[1])
        print(f"{t:5.1f} {px:7.2f} {py:7.2f} {vx:5.2f} {vy:5.2f} {fr:>8} | "
              f"{g[0]:7.2f} {g[1]:7.2f} {d:6.2f}")

print("\n== ego state + full GT ped distance, t=23.4..24.6 ==")
log = ex.last_run_log
for i, (t, st) in enumerate(zip(log.times, log.states)):
    if 23.4 <= t <= 24.6:
        ob = [o for o in sc.obstacles_at(t) if o.track_id == 53]
        d = math.hypot(ob[0].pose.x - st.pose.x, ob[0].pose.y - st.pose.y) if ob else -1
        print(f"t={t:5.1f} ego=({st.pose.x:6.2f},{st.pose.y:6.2f}) "
              f"hdg={st.pose.heading:5.2f} v={st.twist.vx:4.2f} ped_d={d:5.2f}")
