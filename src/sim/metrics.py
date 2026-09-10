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

def calculate_min_ttc(log: RunLog, obstacles_at: Callable[[float], List[Obstacle]]) -> float:
    """Calculates minimum Time-To-Collision with dynamic obstacles."""
    min_ttc = float('inf')
    ego_r = 1.15  # Rough ego radius (width/2 + padding)
    
    for t, state in zip(log.times, log.states):
        for obs in obstacles_at(t):
            if not obs.is_dynamic: continue
            dx = obs.pose.x - state.pose.x
            dy = obs.pose.y - state.pose.y
            dist = math.hypot(dx, dy)
            v_rel = state.twist.vx  # Simplified relative velocity
            
            if v_rel > 0.5 and dist > ego_r:
                # TTC = distance_to_surface / closing_speed
                clearance = dist - ego_r - max(obs.length, obs.width)/2
                if clearance > 0:
                    ttc = clearance / v_rel
                    min_ttc = min(min_ttc, ttc)
                    
    return min_ttc if min_ttc != float('inf') else -1.0