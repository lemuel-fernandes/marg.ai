"""System runner."""

from src.common.coordinates.transforms import TransformTree
from src.common.types.base import FrameId, Header, Pose2D, Twist2D
from src.common.types.config import CostmapConfig, PlannerConfig, VehicleConfig
from src.common.types.sensor import SensorFrame
from src.common.types.vehicle_state import VehicleState
from src.integration.message_bus import TypedMessageBus
from src.integration.mocks import (
    MockController,
    MockGlobalPlanner,
    MockLocalPlanner,
    MockPerception,
)
from src.integration.pipeline import IntegrationPipeline
from src.integration.safety_monitor import EnvelopeSafetyMonitor
from src.integration.state_manager import SystemStateManager
from src.mapping.costmap import LocalGridCostmapBuilder


def build_pipeline() -> IntegrationPipeline:
    vehicle_cfg = VehicleConfig()
    costmap_cfg = CostmapConfig()
    planner_cfg = PlannerConfig()

    transform_tree = TransformTree()
    message_bus = TypedMessageBus()
    state_manager = SystemStateManager()

    perception = MockPerception()
    costmap_builder = LocalGridCostmapBuilder(costmap_cfg)
    global_planner = MockGlobalPlanner()
    local_planner = MockLocalPlanner(planner_cfg, vehicle_cfg)
    controller = MockController()
    safety_monitor = EnvelopeSafetyMonitor(vehicle_cfg)

    return IntegrationPipeline(
        perception=perception,
        costmap_builder=costmap_builder,
        global_planner=global_planner,
        local_planner=local_planner,
        controller=controller,
        safety_monitor=safety_monitor,
        transform_tree=transform_tree,
        message_bus=message_bus,
        state_manager=state_manager,
    )


def run_demo() -> None:
    pipeline = build_pipeline()

    now = 0.0
    header = Header(
        stamp=now,
        frame_id=FrameId.MAP,
        source="system_runner",
    )

    vehicle_state = VehicleState(
        header=header,
        pose=Pose2D(x=0.0, y=0.0, heading=0.0),
        twist=Twist2D(vx=5.0, vy=0.0, yaw_rate=0.0),
        steering_angle=0.0,
        curvature=0.0,
    )

    sensor_frame = SensorFrame(
        header=Header(
            stamp=now,
            frame_id=FrameId.SENSOR_FRONT,
            source="sim",
        )
    )

    goal = Pose2D(x=25.0, y=0.0, heading=0.0)

    command = pipeline.tick(
        now=now,
        vehicle_state=vehicle_state,
        sensor_frame=sensor_frame,
        goal=goal,
    )

    print("Control command:")
    print(command)