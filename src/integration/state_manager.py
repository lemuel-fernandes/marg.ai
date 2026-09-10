"""System state management."""

from enum import Enum
from typing import List, Tuple


class SystemState(str, Enum):
    BOOT = "boot"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAULT = "fault"


class SystemStateManager:
    def __init__(self):
        self.state = SystemState.BOOT
        self.faults: List[Tuple[str, str]] = []

    def set_running(self) -> None:
        if not self.faults:
            self.state = SystemState.RUNNING

    def raise_fault(self, module: str, reason: str) -> None:
        self.faults.append((module, reason))
        self.state = SystemState.FAULT

    def clear_faults(self) -> None:
        self.faults.clear()
        self.state = SystemState.RUNNING