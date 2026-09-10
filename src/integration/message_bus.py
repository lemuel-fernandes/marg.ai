from collections import defaultdict
from typing import Any, Callable, Dict, List

from src.common.types.control import ControlCommand
from src.common.types.costmap import Costmap
from src.common.types.path import GlobalPath
from src.common.types.perception import PerceptionOutput
from src.common.types.trajectory import LocalTrajectory


class Topic:
    PERCEPTION = "perception/output"
    COSTMAP = "mapping/costmap"
    GLOBAL_PATH = "planning/global_path"
    LOCAL_TRAJECTORY = "planning/local_trajectory"
    CONTROL_COMMAND = "control/command"


TOPIC_TYPES: Dict[str, type] = {
    Topic.PERCEPTION: PerceptionOutput,
    Topic.COSTMAP: Costmap,
    Topic.GLOBAL_PATH: GlobalPath,
    Topic.LOCAL_TRAJECTORY: LocalTrajectory,
    Topic.CONTROL_COMMAND: ControlCommand,
}


class MessageValidationError(RuntimeError):
    pass


class TypedMessageBus:
    """
    Typed telemetry bus.

    This is intentionally NOT the primary execution engine.
    It is used for observability and downstream non-critical consumers.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        self._subscribers[topic].append(handler)

    def publish(self, topic: str, message: Any) -> None:
        expected_type = TOPIC_TYPES.get(topic)
        if expected_type is not None and not isinstance(message, expected_type):
            raise MessageValidationError(
                f"Topic '{topic}' expects {expected_type.__name__}, "
                f"got {type(message).__name__}"
            )

        for handler in self._subscribers.get(topic, []):
            handler(message)