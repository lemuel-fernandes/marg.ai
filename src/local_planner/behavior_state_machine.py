from enum import Enum
from typing import Dict, List, Optional, Tuple
import math


class PlannerState(str, Enum):
    LANE_KEEP = "LANE_KEEP"
    NUDGE_CHECK = "NUDGE_CHECK"
    NUDGE_PASSTHRU = "NUDGE_PASSTHRU"
    CREEP_AND_YIELD = "CREEP_AND_YIELD"
    RE_ALIGN = "RE_ALIGN"
    EMERGENCY_REACTIVE_STEER = "EMERGENCY_REACTIVE_STEER"


class BehaviorStateMachine:
    """
    Hierarchical Behavioral State Machine for unstructured Indian traffic.
    Bridges local Frenet trajectory planning and vehicle actuation modes.
    """

    def __init__(self, normal_target_speed: float = 6.0, creep_speed: float = 1.0):
        self.state = PlannerState.LANE_KEEP
        self.normal_target_speed = normal_target_speed
        self.creep_speed = creep_speed
        self.time_in_state = 0.0

    def step(
        self,
        current_s: float,
        current_d: float,
        current_v: float,
        obstacles: List[Dict],
        gap_available: bool,
        min_ttc: float,
        is_intersection: bool = False,
    ) -> Tuple[PlannerState, Dict]:
        """
        Transitions state machine based on perception & TTC triggers.
        Returns (current_state, behavior_directives).
        """
        # 1. Immediate Emergency Override
        if min_ttc < 1.0 and current_v > 0.5:
            self.state = PlannerState.EMERGENCY_REACTIVE_STEER
            return self.state, {
                "target_speed": 0.0,
                "max_accel": -4.5,
                "allow_lateral_nudge": True,
                "preferred_d": current_d,
                "reason": f"TTC critically low ({min_ttc:.2f}s)"
            }

        # 2. Check for ahead blocking obstacles in reference corridor
        ahead_obstacle = False
        obstacle_dist = float('inf')
        for obs in obstacles:
            ds = obs['s'] - current_s
            dd = abs(obs['d'] - current_d)
            if 0.0 < ds < 25.0 and dd < 1.8:
                ahead_obstacle = True
                if ds < obstacle_dist:
                    obstacle_dist = ds

        # State transition logic
        if self.state == PlannerState.LANE_KEEP:
            if is_intersection and ahead_obstacle:
                self.state = PlannerState.CREEP_AND_YIELD
            elif ahead_obstacle:
                self.state = PlannerState.NUDGE_CHECK
            else:
                self.state = PlannerState.LANE_KEEP

        elif self.state == PlannerState.NUDGE_CHECK:
            if not ahead_obstacle:
                self.state = PlannerState.LANE_KEEP
            elif gap_available:
                self.state = PlannerState.NUDGE_PASSTHRU
            else:
                self.state = PlannerState.CREEP_AND_YIELD

        elif self.state == PlannerState.NUDGE_PASSTHRU:
            if not ahead_obstacle:
                self.state = PlannerState.RE_ALIGN
            elif not gap_available and obstacle_dist < 6.0:
                self.state = PlannerState.CREEP_AND_YIELD

        elif self.state == PlannerState.CREEP_AND_YIELD:
            if gap_available and ahead_obstacle:
                self.state = PlannerState.NUDGE_PASSTHRU
            elif not ahead_obstacle:
                self.state = PlannerState.RE_ALIGN

        elif self.state == PlannerState.RE_ALIGN:
            if ahead_obstacle and gap_available:
                self.state = PlannerState.NUDGE_PASSTHRU
            elif abs(current_d) < 0.2:
                self.state = PlannerState.LANE_KEEP

        elif self.state == PlannerState.EMERGENCY_REACTIVE_STEER:
            if min_ttc >= 1.5 or current_v <= 0.2:
                self.state = PlannerState.CREEP_AND_YIELD

        # Behavior directives based on active state
        if self.state == PlannerState.LANE_KEEP:
            directives = {
                "target_speed": self.normal_target_speed,
                "allow_lateral_nudge": False,
                "preferred_d": 0.0,
                "d_range": (-0.6, 0.7, 0.3),
            }
        elif self.state == PlannerState.NUDGE_CHECK:
            directives = {
                "target_speed": min(self.normal_target_speed * 0.7, 3.5),
                "allow_lateral_nudge": True,
                "preferred_d": current_d,
                "d_range": (-1.8, 1.9, 0.3),
            }
        elif self.state == PlannerState.NUDGE_PASSTHRU:
            directives = {
                "target_speed": min(self.normal_target_speed * 0.8, 4.0),
                "allow_lateral_nudge": True,
                "preferred_d": current_d,
                "d_range": (-1.8, 1.9, 0.3),
            }
        elif self.state == PlannerState.CREEP_AND_YIELD:
            # Full nudge range while creeping: with a narrow +-1.0 m band the
            # planner cannot line up a creep-past trajectory around obstacles
            # parked further off-center and the vehicle deadlocks.
            directives = {
                "target_speed": self.creep_speed,
                "allow_lateral_nudge": True,
                "preferred_d": current_d,
                "d_range": (-1.8, 1.9, 0.3),
            }
        elif self.state == PlannerState.RE_ALIGN:
            directives = {
                "target_speed": self.normal_target_speed,
                "allow_lateral_nudge": True,
                "preferred_d": 0.0,
                "d_range": (-0.8, 0.9, 0.2),
            }
        else:  # EMERGENCY_REACTIVE_STEER
            directives = {
                "target_speed": 0.0,
                "allow_lateral_nudge": True,
                "preferred_d": current_d,
                "d_range": (-1.8, 1.9, 0.3),
            }

        return self.state, directives
