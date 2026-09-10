from typing import List, Optional, Protocol, runtime_checkable

from src.common.types.costmap import Costmap
from src.common.types.control import ControlCommand
from src.common.types.base import Pose2D
from src.common.types.obstacle import Obstacle
from src.common.types.path import GlobalPath
from src.common.types.perception import PerceptionOutput
from src.common.types.sensor import SensorFrame
from src.common.types.trajectory import LocalTrajectory
from src.common.types.vehicle_state import VehicleState


class SafetyDecision:
    def __init__(
        self,
        allowed: bool,
        reason: str,
        max_acceleration: float,
        min_acceleration: float,
        max_abs_steer: float,
    ):
        self.allowed = allowed
        self.reason = reason
        self.max_acceleration = max_acceleration
        self.min_acceleration = min_acceleration
        self.max_abs_steer = max_abs_steer


@runtime_checkable
class PerceptionNode(Protocol):
    def process(self, frame: SensorFrame) -> PerceptionOutput:
        ...


@runtime_checkable
class CostmapBuilder(Protocol):
    def update(
        self,
        obstacles: List[Obstacle],
        previous_costmap: Optional[Costmap],
        stamp: float,
    ) -> Costmap:
        ...


@runtime_checkable
class GlobalPlanner(Protocol):
    def plan(
        self,
        costmap: Costmap,
        start: Pose2D,
        goal: Pose2D,
    ) -> GlobalPath:
        ...


@runtime_checkable
class LocalPlanner(Protocol):
    def plan(
        self,
        state: VehicleState,
        global_path: GlobalPath,
        costmap: Costmap,
        obstacles: List[Obstacle],
    ) -> LocalTrajectory:
        ...

    def emergency_stop(self, state: VehicleState) -> LocalTrajectory:
        ...


@runtime_checkable
class Controller(Protocol):
    def command(
        self,
        state: VehicleState,
        trajectory: LocalTrajectory,
    ) -> ControlCommand:
        ...


@runtime_checkable
class SafetyMonitor(Protocol):
    def evaluate(
        self,
        state: VehicleState,
        trajectory: LocalTrajectory,
        obstacles: List[Obstacle],
        costmap: Costmap,
    ) -> SafetyDecision:
        ...

    def limit_command(
        self,
        state: VehicleState,
        command: ControlCommand,
        decision: SafetyDecision,
    ) -> ControlCommand:
        ...