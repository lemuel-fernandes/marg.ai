"""Probe occluded_siren: when does the acoustic hold release, and what does
the planner do afterward? Wraps the planner's AudioPlannerPolicy calls and
logs ego state + cue/track data per tick.
"""
import math
import sys

from src.sim.executor import ScenarioExecutor
from src.sim.scenario_registry import REGISTRY

LOG = []


def main():
    sc = REGISTRY["occluded_siren"]()
    orig_get_lp = sc.get_local_planner

    def traced_get_lp(v_cfg, dwa_cfg):
        lp = orig_get_lp(v_cfg, dwa_cfg)
        policy = lp.audio_policy
        orig_release = policy.creep_release
        orig_cap = policy.siren_cap

        def release(cues, track_azimuths):
            out = orig_release(cues, track_azimuths)
            LOG.append({"kind": "release", "out": out, "n_cues": len(cues),
                        "n_tracks": len(track_azimuths)})
            return out

        def cap(cues):
            out = orig_cap(cues)
            LOG.append({"kind": "cap", "out": out, "n_cues": len(cues)})
            return out

        policy.creep_release = release
        policy.siren_cap = cap

        orig_plan = lp.plan

        def plan(state, global_path, costmap, obstacles):
            traj = orig_plan(state, global_path, costmap, obstacles)
            LOG.append({"kind": "plan", "t": state.header.stamp,
                        "x": state.pose.x, "y": state.pose.y,
                        "v": state.twist.vx, "reason": traj.reason,
                        "hold": getattr(lp, "_hold_active", None),
                        "anchor": getattr(lp, "_hold_anchor_d", None),
                        "cues": len(getattr(lp, "_acoustic_cues", []))})
            return traj

        lp.plan = plan
        return lp

    sc.get_local_planner = traced_get_lp
    result = ScenarioExecutor(sc).run(save_plot=None)

    print("\n--- ticks 1.5..9s (plan rows) ---")
    for e in LOG:
        if e["kind"] != "plan" or not (1.5 <= e["t"] <= 9.0):
            continue
        a = f"{e['anchor']:+.1f}" if isinstance(e.get("anchor"), float) else "  -  "
        print(f"t={e['t']:5.1f} pos=({e['x']:6.2f},{e['y']:5.2f}) "
              f"v={e['v']:5.2f} hold={int(bool(e['hold']))} anchor={a} "
              f"cues={e['cues']} reason={e['reason'][:36]}")

    # How often is the hold active?
    rel = [e for e in LOG if e["kind"] == "release"]
    holds = [i * 0.1 for i, e in enumerate(rel) if e["out"]]
    print(f"\nrelease calls: {len(rel)}, raw hold ticks: {len(holds)}")
    if holds:
        # Compress to contiguous ranges
        ranges = []
        s = p = holds[0]
        for t in holds[1:]:
            if abs(t - p - 0.1) < 1e-6:
                p = t
                continue
            ranges.append((s, p))
            s = p = t
        ranges.append((s, p))
        for a, b in ranges:
            print(f"  raw hold t={a:.1f}..{b:.1f}")


if __name__ == "__main__":
    main()
