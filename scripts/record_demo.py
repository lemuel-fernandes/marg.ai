"""Record the PathSense simulation suite to an mp4 demo video.

Runs scenarios headlessly (matplotlib Agg), snapshots live telemetry every
simulation tick via the message bus, renders each tick to a 960x540 frame and
streams frames into an mp4 encoded by the ffmpeg binary bundled with
imageio-ffmpeg. Multiple segments are concatenated losslessly with ffmpeg
(-f concat -c copy).

Usage:
    python scripts/record_demo.py static_obstacle crossing_animal -o seg_a.mp4
    python scripts/record_demo.py --intro -o seg_intro.mp4
    python scripts/record_demo.py --outro -o seg_outro.mp4
"""
import argparse
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # headless: no window, no Qt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, Polygon

from src.sim.executor import ScenarioExecutor
from src.sim.scenario_registry import REGISTRY

OUT_DIR = "data/recording"
FPS = 10            # sim runs at dt=0.1 s -> real-time playback
FIGSIZE = (9.6, 5.4)   # 100 dpi -> 960x540 px
DPI = 100

LEAD_FRAMES = 10     # hold the opening shot before motion
TAIL_FRAMES = 30     # hold the closing shot after the PASS badge
SLOW_SCENARIOS = {"crossing_animal": 1, "occluded_siren": 1}  # dwell/tick
SKIP_OVER_S = 30.0   # scenarios longer than this render every 2nd tick

TITLES = {
    "static_obstacle": ("Static Obstacle Avoidance",
                        "A* plans around a blocked lane - DWA keeps a safe margin"),
    "crossing_animal": ("Sudden Animal Crossing",
                        "Time-aware DWA predicts the animal and brakes + swerves within dynamic limits"),
    "multi_animal": ("Multiple Animals",
                     "Simultaneous dynamic obstacles handled with zero emergency stops"),
    "indian_road": ("Unstructured Indian Road",
                    "Encroaching carts, wrong-side traffic and potholes on a single route"),
    "city_roads": ("City Roads",
                   "Dense urban traffic - emergency stops used, zero collisions"),
    "sensor_noise": ("Sensor Noise Robustness",
                     "Noisy detections still yield a stable, collision-free path"),
    "occluded_siren": ("Occluded Siren (Acoustic Attention)",
                       "The vehicle hears the siren it cannot see and creeps forward cautiously"),
}

C_TRAIL = "#1565c0"
C_GLOBAL = "#00b8d4"
C_DYN = "#ab47bc"
C_STAT = "#e53935"
C_EGO = "#1e88e5"
C_CAND = "#9e9e9e"
C_CUE = "#ff6f00"

EGO_LEN, EGO_WID = 4.2, 1.9


class FrameRenderer:
    """Renders one telemetry snapshot onto a fixed matplotlib figure."""

    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    def render(self, snap, title, caption, trail, ego, t_now, passed):
        ax = self.ax
        ax.clear()
        ax.set_aspect("equal")
        ax.set_facecolor("#fafafa")
        ax.grid(True, linestyle="--", alpha=0.35)

        snap = snap or {}
        obstacles = snap.get("obstacles", [])
        cmap = snap.get("cmap")
        gp = snap.get("gp")
        local = snap.get("local")
        cand = snap.get("cand")
        cue = snap.get("cue")

        # --- world bounds (16:9 window around the action) -----------------
        xs, ys = [], []
        for x, y, _ in trail:
            xs.append(x); ys.append(y)
        for o in obstacles:
            xs.append(o.pose.x); ys.append(o.pose.y)
        if gp is not None and getattr(gp, "points", None):
            xs += [p.pose.x for p in gp.points]
            ys += [p.pose.y for p in gp.points]
        if local is not None and getattr(local, "points", None):
            xs += [p.pose.x for p in local.points]
            ys += [p.pose.y for p in local.points]
        if not xs:
            xs, ys = [0, 10], [0, 10]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half = min(max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 12, 90)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half * 0.5625, cy + half * 0.5625)

        # --- costmap -------------------------------------------------------
        if cmap is not None and getattr(cmap, "data", None) is not None \
                and cmap.data.size > 0:
            ext = [cmap.origin_x, cmap.origin_x + cmap.width * cmap.resolution,
                   cmap.origin_y, cmap.origin_y + cmap.height * cmap.resolution]
            ax.imshow(cmap.data, cmap="Reds", origin="lower", extent=ext,
                      alpha=0.35, vmin=0, vmax=1, zorder=1)

        # --- global path ----------------------------------------------------
        if gp is not None and getattr(gp, "points", None):
            ax.plot([p.pose.x for p in gp.points],
                    [p.pose.y for p in gp.points],
                    "--", color=C_GLOBAL, lw=2.0, alpha=0.9, zorder=2,
                    label="Global Path (A*)")

        # --- DWA candidate cloud (thinned for render speed) -----------------
        if cand:
            shown = 0
            for c in cand[::3]:
                if shown >= 24:
                    break
                pts = getattr(c, "points", None)
                if not pts:
                    continue
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=C_CAND, lw=0.5, alpha=0.15, zorder=3)
                shown += 1

        # --- obstacles --------------------------------------------------------
        for o in obstacles:
            col = C_DYN if o.is_dynamic else C_STAT
            r = max(o.length, o.width) / 2.0
            ax.add_patch(plt.Circle((o.pose.x, o.pose.y), r, color=col,
                                    alpha=0.75, zorder=5))
            if o.is_dynamic:
                v = o.velocity
                ax.add_patch(FancyArrow(o.pose.x, o.pose.y,
                                        float(v.vx) * 1.5, float(v.vy) * 1.5,
                                        head_width=0.35, head_length=0.25,
                                        fc=col, ec=col, zorder=6))

        # --- acoustic cue cone -------------------------------------------------
        if cue is not None and trail:
            ex, ey, eh = trail[-1]
            az = eh + float(getattr(cue, "azimuth_rad", 0.0))
            for da in (-math.radians(15), math.radians(15)):
                ax.plot([ex, ex + 18 * math.cos(az + da)],
                        [ey, ey + 18 * math.sin(az + da)],
                        color=C_CUE, lw=1.5, ls=":", alpha=0.9, zorder=6)
            ax.text(ex + 6 * math.cos(az), ey + 6 * math.sin(az) + 1.0,
                    f"audio cue: {getattr(cue, 'event_class', 'sound')}",
                    color=C_CUE, fontsize=9, weight="bold", zorder=10)

        # --- trail ---------------------------------------------------------------
        if len(trail) > 1:
            ax.plot([p[0] for p in trail], [p[1] for p in trail], "-",
                    color=C_TRAIL, lw=2.5, zorder=4, label="Vehicle Trail")

        # --- ego -------------------------------------------------------------------
        if ego is not None:
            x, y, h = ego
            ch, sh = math.cos(h), math.sin(h)
            corners = [(-EGO_LEN / 2, -EGO_WID / 2), (EGO_LEN / 2, -EGO_WID / 2),
                       (EGO_LEN / 2, EGO_WID / 2), (-EGO_LEN / 2, EGO_WID / 2)]
            rot = [(x + a * ch - b * sh, y + a * sh + b * ch)
                   for a, b in corners]
            ax.add_patch(Polygon(rot, closed=True, color=C_EGO, alpha=0.9,
                                 zorder=8, label="Ego Vehicle"))
            ax.add_patch(FancyArrow(x, y, 2.5 * ch, 2.5 * sh, head_width=0.5,
                                    head_length=0.35, fc=C_EGO, ec=C_EGO,
                                    zorder=9))

        # --- goal --------------------------------------------------------------------
        if gp is not None and getattr(gp, "points", None):
            ax.plot(gp.points[-1].pose.x, gp.points[-1].pose.y, "*",
                    color="#d84315", markersize=16, zorder=7, label="Goal")

        # --- HUD ------------------------------------------------------------------------
        ax.set_title(f"PathSense  -  {title}", fontsize=15, weight="bold",
                     loc="left", pad=16)
        ax.text(0.0, 1.02, caption, transform=ax.transAxes, fontsize=10,
                color="#455a64")
        ax.text(0.995, 1.02, f"t = {t_now:5.1f} s", transform=ax.transAxes,
                fontsize=10, ha="right", color="#37474f", family="monospace")
        if passed is not None:
            badge = "REACHED GOAL - PASS" if passed else "FAIL"
            col = "#2e7d32" if passed else "#c62828"
            ax.text(0.5, 0.05, badge, transform=ax.transAxes, fontsize=15,
                    weight="bold", ha="center", color="white", zorder=20,
                    bbox=dict(boxstyle="round,pad=0.45", fc=col, ec="none"))
        leg = ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
        if leg:
            leg.set_zorder(20)

    def grab(self):
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        return buf[..., :3].copy()

    def close(self):
        plt.close(self.fig)


class SegmentWriter:
    """Streams frames into an mp4, with cached-frame holds (no redraws)."""

    def __init__(self, path):
        import imageio.v2 as imageio
        self._writer = imageio.get_writer(
            path, fps=FPS, codec="libx264", quality=8,
            pixelformat="yuv420p", macro_block_size=2)
        self._last = None

    def append(self, frame):
        self._last = frame
        self._writer.append_data(frame)

    def hold(self, n):
        if self._last is None:
            return
        for _ in range(n):
            self._writer.append_data(self._last)

    def close(self):
        self._writer.close()


def record_scenario(name, cls, seg, renderer):
    title, caption = TITLES.get(name, (name.replace("_", " ").title(), ""))
    snaps = []
    snap_tpl = {"obstacles": [], "cmap": None, "gp": None, "local": None,
                "cand": None, "cue": None}

    def on_bus(bus):
        from src.integration.message_bus import Topic

        cur = dict(snap_tpl)

        def on_perc(msg):
            cur["obstacles"] = list(msg.obstacles)

        def on_cmap(msg):
            cur["cmap"] = msg

        def on_gp(msg):
            cur["gp"] = msg

        def on_local(msg):
            cur["local"] = msg
            cur["cand"] = getattr(msg, "debug_candidates", None)
            snaps.append(dict(cur))   # one snapshot per planning tick

        def on_cue(msg):
            cur["cue"] = msg

        bus.subscribe(Topic.PERCEPTION, on_perc)
        bus.subscribe(Topic.COSTMAP, on_cmap)
        bus.subscribe(Topic.GLOBAL_PATH, on_gp)
        bus.subscribe(Topic.LOCAL_TRAJECTORY, on_local)
        bus.subscribe(Topic.ACOUSTIC_CUE, on_cue)

    executor = ScenarioExecutor(cls(), live=False)
    result = executor.run(on_bus=on_bus)

    trail = []
    log = getattr(executor, "last_run_log", None)
    if log is not None:
        trail = [(s.pose.x, s.pose.y, s.pose.heading) for s in log.states]

    dwell = SLOW_SCENARIOS.get(name, 0)
    sc = cls()
    skip = 2 if getattr(sc, "duration_s", 12.0) > SKIP_OVER_S else 1

    # Lead-in: scene before motion (start point only).
    first = snaps[0] if snaps else None
    t0 = 0.0
    if first is not None and first.get("local") is not None:
        t0 = float(getattr(first["local"].header, "stamp", 0.0))
    renderer.render(first, title, caption, trail[:1], trail[0] if trail
                    else None, t0, None)
    seg.append(renderer.grab())
    seg.hold(LEAD_FRAMES)

    # Playback: trail grows tick by tick; long scenarios render every 2nd
    # tick and duplicate the frame so playback stays real-time.
    for i, s in enumerate(snaps):
        if skip > 1 and i % 2 == 1:
            continue
        ego = None
        if s.get("local") is not None and getattr(s["local"], "points", None):
            p = s["local"].points[0].pose
            ego = (p.x, p.y, p.heading)
        elif i < len(trail):
            ego = trail[i]
        t_now = float(getattr(s["local"].header, "stamp", i * 0.1)) \
            if s.get("local") is not None else i * 0.1
        renderer.render(s, title, caption, trail[: i + 2], ego, t_now, None)
        seg.append(renderer.grab())
        seg.hold(dwell)

    # Outro: full trail + PASS badge.
    last = snaps[-1] if snaps else None
    t_end = (len(snaps) - 1) * 0.1
    renderer.render(last, title, caption, trail,
                    trail[-1] if trail else None, t_end, bool(result.passed))
    seg.append(renderer.grab())
    seg.hold(TAIL_FRAMES)

    print(f"[Recorder] {name}: {'PASS' if result.passed else 'FAIL'} "
          f"-> {len(snaps)} ticks", flush=True)
    return result


def render_card(lines, seg, renderer, n_frames):
    ax = renderer.ax
    ax.clear()
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    ax.axis("off")
    y = 6.8
    for text, size, weight, color in lines:
        ax.text(8, y, text, ha="center", va="center", fontsize=size,
                weight=weight, color=color)
        y -= size * 0.10 + 0.55
    seg.append(renderer.grab())
    seg.hold(n_frames - 1)


INTRO = [
    ("PathSense", 34, "bold", "#0d47a1"),
    ("Adaptive Path Planning & Collision Avoidance", 15, "normal", "#263238"),
    ("for Unstructured Indian Roads  -  SIH26037", 15, "normal", "#263238"),
    ("", 8, "normal", "#263238"),
    ("A* global planning  -  Time-aware DWA  -  Envelope safety monitor", 11, "normal", "#455a64"),
    ("Vision + acoustic attention fusion", 11, "normal", "#455a64"),
]

OUTRO = [
    ("Results: 7 / 7 scenarios PASS - zero collisions", 17, "bold", "#1b5e20"),
    ("Zero emergency stops on dynamic-animal scenarios", 12, "normal", "#263238"),
    ("Path efficiency up to 100%  -  bounded jerk & braking", 12, "normal", "#263238"),
    ("Acoustic attention handles occluded sirens", 12, "normal", "#263238"),
    ("", 8, "normal", "#263238"),
    ("github.com/  -  PathSense team, SIH26037", 10, "normal", "#607d8b"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*")
    ap.add_argument("-o", "--out", required=True, help="segment mp4 path")
    ap.add_argument("--intro", action="store_true")
    ap.add_argument("--outro", action="store_true")
    args = ap.parse_args()

    for n in args.scenarios:
        if n not in REGISTRY:
            print(f"Unknown scenario '{n}'. Available: {list(REGISTRY)}")
            sys.exit(2)

    os.makedirs(OUT_DIR, exist_ok=True)
    seg = SegmentWriter(args.out)
    renderer = FrameRenderer()
    ok = True
    try:
        if args.intro:
            render_card(INTRO, seg, renderer, 35)
        for name in args.scenarios:
            result = record_scenario(name, REGISTRY[name], seg, renderer)
            ok &= result.passed
        if args.outro:
            render_card(OUTRO, seg, renderer, 40)
    finally:
        seg.close()
        renderer.close()
    print(f"[Recorder] Segment saved: {args.out}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
