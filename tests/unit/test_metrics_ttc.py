"""Unit tests for the response-aware min-TTC metric (src/sim/metrics.py)."""
import math
from types import SimpleNamespace

from src.common.types.base import Covariance2D, FrameId, Header, Pose2D, Twist2D
from src.common.types.obstacle import Obstacle, ObstacleBehavior, ObstacleClass
from src.sim.metrics import (calculate_min_ttc, calculate_min_ttc_overall,
                             calculate_min_ttc_legacy)


def _make_obstacle(x, y=0.0, vx=0.0, vy=0.0, length=1.6, width=0.9):
    return Obstacle(
        header=Header(0.0, FrameId.MAP, "test"), track_id=1,
        class_label=ObstacleClass.ANIMAL, behavior=ObstacleBehavior.CROSSING,
        pose=Pose2D(x, y, 0.0), length=length, width=width,
        velocity=Twist2D(vx=vx, vy=vy, yaw_rate=0.0),
        pose_covariance=Covariance2D(), velocity_covariance=Covariance2D(),
        confidence=1.0, is_dynamic=True,
    )


def _make_log(series):
    """series: list of (t, x, v) tuples for the ego (heading 0, y 0)."""
    return SimpleNamespace(
        times=[s[0] for s in series],
        states=[SimpleNamespace(pose=Pose2D(s[1], 0.0, 0.0),
                                twist=Twist2D(vx=s[2], vy=0.0))
                for s in series],
    )


def test_no_response_equals_classic_ttc():
    """Constant-speed approach, no braking: response-aware == classic."""
    # Ego at x=0, v=5; dynamic obstacle 10 m ahead, stationary.
    log = _make_log([(0.0, 0.0, 5.0), (0.1, 0.5, 5.0), (0.2, 1.0, 5.0)])
    obs_at = lambda t: [_make_obstacle(10.0)]
    # Minimum lands at the last tick: ego advanced to x=1.0.
    expected = (10.0 - 1.0 - 0.8 - 0.95) / 5.0  # 1.45 s
    assert calculate_min_ttc(log, obs_at) == calculate_min_ttc_legacy(log, obs_at)
    assert math.isclose(calculate_min_ttc(log, obs_at), expected, rel_tol=1e-6)
    assert math.isclose(calculate_min_ttc_overall(log, obs_at), expected, rel_tol=1e-6)


def test_braking_samples_excluded_from_primary_metric():
    """Ego braking while threat closes: responding samples excluded from
    min_ttc but still visible in the overall metric."""
    # Ego decelerates 5 -> 0 at 10 m/s^2 (a = -10 per-tick series).
    # Obstacle approaches at 8 m/s (consistent position + velocity).
    series = []
    v = 5.0
    for i in range(6):
        t = round(0.1 * i, 1)
        series.append((t, 0.0, v))
        v = max(v - 1.0, 0.0)  # -10 m/s^2
    obs_x = lambda t: 9.0 - 8.0 * t
    log = _make_log(series)
    obs_at = lambda t: [_make_obstacle(obs_x(t), vx=-8.0)]

    overall = calculate_min_ttc_overall(log, obs_at)
    primary = calculate_min_ttc(log, obs_at)

    # Overall (classic) minimum lands on the final, hardest-braking tick.
    clear_final = (obs_x(0.5) - 0.8) - 0.95   # 3.25
    assert math.isclose(overall, clear_final / 8.0, rel_tol=1e-6)
    # Primary metric excludes all braking ticks -> only t=0 qualifies.
    clear_first = (obs_x(0.0) - 0.8) - 0.95   # 7.25
    assert math.isclose(primary, clear_first / 13.0, rel_tol=1e-6)
    assert primary > overall


def test_no_closing_events_returns_sentinel():
    """Obstacle receding: no closing events -> -1.0."""
    log = _make_log([(0.0, 0.0, 5.0)])
    obs_at = lambda t: [_make_obstacle(10.0, vx=5.0)]  # same speed, static gap
    assert calculate_min_ttc(log, obs_at) == -1.0
    assert calculate_min_ttc_overall(log, obs_at) == -1.0


def test_direct_overlap_scores_zero():
    """Obstacle on top of the ego -> TTC 0.0 in both metrics."""
    log = _make_log([(0.0, 0.0, 5.0), (0.1, 0.5, 5.0)])
    obs_at = lambda t: [_make_obstacle(0.0 if t < 0.05 else 0.5, vx=-5.0)]
    assert calculate_min_ttc_overall(log, obs_at) == 0.0
    assert calculate_min_ttc(log, obs_at) == 0.0


def test_short_brake_pulse_within_window_counts():
    """A brief braking pulse inside the response window marks later ticks."""
    # Ego brakes hard for one tick at t=0.1, then coasts.
    series = [(0.0, 0.0, 5.0), (0.1, 0.5, 4.0), (0.2, 0.9, 4.0), (0.3, 1.3, 4.0)]
    # Obstacle closing slowly from ahead; clearance shrinks over time so the
    # minimum lands at t=0.3, which is within 0.3s of the brake pulse.
    obs_x = lambda t: 6.0 - 2.0 * t
    log = _make_log(series)
    obs_at = lambda t: [_make_obstacle(obs_x(t), vx=-2.0)]
    primary = calculate_min_ttc(log, obs_at)
    overall = calculate_min_ttc_overall(log, obs_at)
    # All ticks after the pulse are "responding", so the primary metric only
    # sees t=0.0 -> strictly larger than the overall minimum at t=0.3.
    assert primary > overall
