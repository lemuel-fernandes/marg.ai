from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reason: str
    max_acceleration: float
    min_acceleration: float
    max_abs_steer: float