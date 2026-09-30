from typing import Callable, List

from src.common.types.base import FrameId, Header
from src.common.types.obstacle import Obstacle
from src.common.types.perception import PerceptionOutput
from src.common.types.sensor import SensorFrame


class GroundTruthPerception:
    """Perfect sensor reading world-fixed obstacles from the simulator."""

    def __init__(self, obstacles_provider: Callable[[], List[Obstacle]]):
        self.provider = obstacles_provider

    def process(self, frame: SensorFrame) -> PerceptionOutput:
        return PerceptionOutput(
            header=Header(
                stamp=frame.header.stamp,
                frame_id=FrameId.MAP,
                source="ground_truth_perception",
            ),
            obstacles=self.provider(),
            latency_ms=0.5,
            status="OK",
        )