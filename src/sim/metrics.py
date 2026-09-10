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

def calculate_min_ttc(log, obstacles_at, ego_half_width: float = 0.95):
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