from dataclasses import dataclass, field
from typing import List

from .base import Header
from .obstacle import Obstacle


@dataclass
class PerceptionOutput:
    header: Header
    obstacles: List[Obstacle] = field(default_factory=list)
    latency_ms: float = 0.0
    status: str = "OK"