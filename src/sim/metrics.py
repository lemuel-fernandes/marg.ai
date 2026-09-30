import math
from typing import List, Callable
from src.sim.scenarios.base import RunLog
from src.common.types.obstacle import Obstacle

def calculate_path_length(log: RunLog) -> float:
    length = 0.0
    for i in range(1, len(log.states)):
        p1 = log.states[i-1].pose
        p2 = log.states[i].pose
        length += math.hypot(p2.x - p1.x, p2.y - p1.y)
    return length

def calculate_path_efficiency(log: RunLog) -> float:
    if len(log.states) < 2: return 1.0
    start = log.states[0].pose
    end = log.states[-1].pose
    optimal = math.hypot(end.x - start.x, end.y - start.y)
    actual = calculate_path_length(log)
    return optimal / actual if actual > 0 else 0.0

def calculate_avg_jerk(log: RunLog, dt: float = 0.1) -> float:
    if len(log.commands) < 3: return 0.0
    jerks = []
    for i in range(2, len(log.commands)):
        a1 = log.commands[i-1].acceleration
        a2 = log.commands[i].acceleration
        jerks.append(abs(a2 - a1) / dt)
    return sum(jerks) / len(jerks)

from src.common.utils.geometry import oriented_rect_clearance

# Response-aware TTC tuning:
# A closing event is considered "actively responded to" when the ego shows
# real braking (beyond coasting noise) within this many seconds up to and
# including the event tick, OR when the ego is already (near-)stopped: a
# stopped vehicle has no remaining response authority — coming to a halt IS
# the response, and further closure is purely the obstacle's own motion.
RESPOND_ACCEL_THRESH = -0.4   # m/s^2
RESPOND_WINDOW_S = 0.3
STOPPED_SPEED_THRESH = 0.3    # m/s — at/below this the ego cannot open the gap further


def _ttc_events(log, obstacles_at, ego_half_width: float):
    """Extract per-(tick, obstacle) closing events.

    Each event: dict(t, ttc, closing, clear, track_id). TTC is the standard
    constant-velocity extrapolation clear/closing along the line of sight,
    with the ego approximated as a disc of ego_half_width radius.
    """
    events = []
    for idx, (t, state) in enumerate(zip(log.times, log.states)):
        evx = state.twist.vx * math.cos(state.pose.heading)
        evy = state.twist.vx * math.sin(state.pose.heading)
        for obs in obstacles_at(t):
            if not obs.is_dynamic:
                continue
            dx, dy = obs.pose.x - state.pose.x, obs.pose.y - state.pose.y
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                events.append({"t": t, "i": idx, "ttc": 0.0, "closing": float("inf"),
                               "clear": 0.0, "track_id": obs.track_id})
                continue
            ux, uy = dx / dist, dy / dist
            closing = (evx - obs.velocity.vx) * ux + (evy - obs.velocity.vy) * uy
            if closing <= 0.5:
                continue
            clear = oriented_rect_clearance(state.pose.x, state.pose.y,
                                            obs.pose.x, obs.pose.y, obs.pose.heading,
                                            obs.length, obs.width) - ego_half_width
            if clear <= 0:
                continue
            events.append({"t": t, "i": idx, "ttc": clear / closing, "closing": closing,
                           "clear": clear, "track_id": obs.track_id})
    return events


def _ego_accel_series(log) -> List[float]:
    """Actual longitudinal acceleration per tick, from consecutive state speeds."""
    accels: List[float] = []
    prev_v = prev_t = None
    for t, st in zip(log.times, log.states):
        if prev_t is None or t - prev_t <= 1e-9:
            accels.append(0.0)
        else:
            accels.append((st.twist.vx - prev_v) / (t - prev_t))
        prev_v, prev_t = st.twist.vx, t
    return accels


def _responding_mask(times: List[float], accels: List[float],
                     speeds: List[float]) -> List[bool]:
    """Per-tick flag: the ego cannot be further demanded a response here.

    A tick is "responding" when either
    - the ego has been braking within the response window ending here, or
    - the ego is already (near-)stopped (no response authority left).

    Uses actual tick timestamps (works for non-uniform dt).
    """
    mask = [False] * len(accels)
    for i, t in enumerate(times):
        j = i
        while j > 0 and t - times[j - 1] <= RESPOND_WINDOW_S:
            j -= 1
        braking = any(a <= RESPOND_ACCEL_THRESH for a in accels[j:i + 1])
        stopped = speeds[i] <= STOPPED_SPEED_THRESH
        mask[i] = braking or stopped
    return mask


def calculate_min_ttc(log, obstacles_at, ego_half_width: float = 0.95):
    """Response-aware minimum time-to-collision.

    The classic CV-extrapolation TTC penalizes any instant with small
    clearance / closing-speed — including the instants right after a surprise
    event (e.g. a VRU spawning a few meters ahead), where braking has not yet
    had time to open the gap. Those samples dominate the metric even when the
    ego reacts correctly and no contact occurs.

    This metric therefore reports the minimum TTC over closing events during
    which the ego was NOT actively responding (no braking of
    RESPOND_ACCEL_THRESH or stronger within the previous RESPOND_WINDOW_S).
    Low values now mean "a threat the system failed to react to", which is
    the safety-relevant signal. The unfiltered minimum remains available via
    :func:`calculate_min_ttc_overall`.

    Returns -1.0 when no qualifying event exists.
    """
    events = _ttc_events(log, obstacles_at, ego_half_width)
    if not events:
        return -1.0
    accels = _ego_accel_series(log)
    speeds = [st.twist.vx for st in log.states]
    responding = _responding_mask(list(log.times), accels, speeds)
    min_ttc = float("inf")
    for ev in events:
        if responding[ev["i"]]:
            continue
        min_ttc = min(min_ttc, ev["ttc"])
    return min_ttc if min_ttc != float("inf") else -1.0


def calculate_min_ttc_overall(log, obstacles_at, ego_half_width: float = 0.95):
    """Unfiltered minimum TTC over all closing events (classic semantics)."""
    events = _ttc_events(log, obstacles_at, ego_half_width)
    if not events:
        return -1.0
    return min(ev["ttc"] for ev in events)


def ttc_gate_failure(min_ttc: float, threshold: float = 0.25):
    """Standard pass/fail gate on the response-aware min TTC.

    Returns a failure message when the unresponded min TTC falls below the
    threshold, None otherwise. ``threshold=None`` disables the gate. A
    ``min_ttc`` of -1.0 (no closing events) always passes.
    """
    if threshold is None:
        return None
    if 0.0 <= min_ttc < threshold:
        return (f"unresponded min TTC {min_ttc:.2f}s below "
                f"{threshold:.2f}s gate")
    return None


def calculate_min_ttc_legacy(log, obstacles_at, ego_half_width: float = 0.95):
    """Original tick-loop implementation kept for reference/tests."""
    min_ttc = float("inf")
    for t, state in zip(log.times, log.states):
        evx = state.twist.vx * math.cos(state.pose.heading)
        evy = state.twist.vx * math.sin(state.pose.heading)
        for obs in obstacles_at(t):
            if not obs.is_dynamic:
                continue
            dx, dy = obs.pose.x - state.pose.x, obs.pose.y - state.pose.y
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                return 0.0
            ux, uy = dx / dist, dy / dist
            closing = (evx - obs.velocity.vx) * ux + (evy - obs.velocity.vy) * uy
            if closing > 0.5:
                clear = oriented_rect_clearance(state.pose.x, state.pose.y,
                                                obs.pose.x, obs.pose.y, obs.pose.heading,
                                                obs.length, obs.width) - ego_half_width
                if clear > 0:
                    min_ttc = min(min_ttc, clear / closing)
    return min_ttc if min_ttc != float("inf") else -1.0