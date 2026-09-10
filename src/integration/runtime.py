import time
from typing import Optional

from src.common.types.base import Header, FrameId, Pose2D
from src.common.types.control import ControlCommand, ControlMode
from src.common.types.costmap import Costmap
from src.common.types.obstacle import Obstacle
from src.common.types.path import GlobalPath
from src.common.types.perception import PerceptionOutput
from src.common.types.sensor import SensorFrame
from src.common.types.trajectory import LocalTrajectory
from src.common.types.vehicle_state import VehicleState

from .contracts import (
    Controller,
    CostmapBuilder,
    GlobalPlanner,
    LocalPlanner,
    PerceptionNode,
    SafetyMonitor,
)
from .message_bus import TypedMessageBus, Topic
from .safety import SafetyDecision


class IntegrationFault(RuntimeError):
    pass


class IntegrationRuntime:
    """
    Deterministic runtime for the safety-critical path.

    This is the core M1 integration layer.
    Telemetry is published to the bus, but execution is sequential and explicit.
    """

    def __init__(
        self,
        perception: PerceptionNode,
        costmap_builder: CostmapBuilder,
        global_planner: GlobalPlanner,
        local_planner: LocalPlanner,
        controller: Controller,
        safety_monitor: SafetyMonitor,
        transform_tree,
        message_bus: Optional[TypedMessageBus] = None,
    ):
        self.perception = perception
        self.costmap_builder = costmap_builder
        self.global_planner = global_planner
        self.local_planner = local_planner
        self.controller = controller
        self.safety_monitor = safety_monitor
        self.tf = transform_tree
        self.bus = message_bus

        self._current_costmap: Optional[Costmap] = None
        self._current_global_path: Optional[GlobalPath] = None
        self._current_goal: Optional[Pose2D] = None

    def tick(
        self,
        now: float,
        vehicle_state: VehicleState,
        sensor_frame: SensorFrame,
        goal: Pose2D,
    ) -> ControlCommand:
        self.tf.update_ego_state(vehicle_state)

        try:
            # 1. Perception
            perception_out = self.perception.process(sensor_frame)
            map_obstacles = self.tf.obstacles_to_map(perception_out.obstacles)
            perception_map = PerceptionOutput(
                header=Header(stamp=now, frame_id=FrameId.MAP, source="integration"),
                obstacles=map_obstacles,
            )

            # 2. Costmap
            self._current_costmap = self.costmap_builder.update(
                obstacles=map_obstacles,
                previous_costmap=self._current_costmap,
                stamp=now,
            )

            # 3. Global planning
            if self._should_replan(goal):
                self._current_global_path = self.global_planner.plan(
                    costmap=self._current_costmap,
                    start=vehicle_state.pose,
                    goal=goal,
                )
                self._current_goal = goal

            if self._current_global_path is None:
                raise IntegrationFault("Global path unavailable")

            # 4. Local planning
            local_traj = self.local_planner.plan(
                state=vehicle_state,
                global_path=self._current_global_path,
                costmap=self._current_costmap,
                obstacles=map_obstacles,
            )

            # 5. Safety evaluation
            decision = self.safety_monitor.evaluate(
                state=vehicle_state,
                trajectory=local_traj,
                obstacles=map_obstacles,
                costmap=self._current_costmap,
            )

            if not decision.allowed:
                local_traj = self.local_planner.emergency_stop(vehicle_state)
                decision = SafetyDecision(
                    allowed=True,
                    reason="Fallback emergency stop active",
                    max_acceleration=0.0,
                    min_acceleration=-3.0,
                    max_abs_steer=0.0,
                )

            # 6. Control
            command = self.controller.command(vehicle_state, local_traj)
            command = self.safety_monitor.limit_command(vehicle_state, command, decision)

            # 7. Telemetry
            self._publish_telemetry(
                perception_map,
                self._current_costmap,
                self._current_global_path,
                local_traj,
                command,
            )

            return command

        except Exception as exc:
            # Fail safe. Do not propagate unknown faults into control.
            return self._emergency_stop_command(now, vehicle_state, str(exc))

    def _should_replan(self, goal: Pose2D) -> bool:
        if self._current_global_path is None:
            return True

        if self._current_goal is None:
            return True

        goal_changed = (
            abs(goal.x - self._current_goal.x) > 1e-6
            or abs(goal.y - self._current_goal.y) > 1e-6
        )

        return (
            goal_changed
            or self._current_global_path.replan_required
            or not self._current_global_path.is_feasible
        )

    def _publish_telemetry(
        self,
        perception_out: PerceptionOutput,
        costmap: Costmap,
        global_path: GlobalPath,
        local_traj: LocalTrajectory,
        command: ControlCommand,
    ) -> None:
        if self.bus is None:
            return

        self.bus.publish(Topic.PERCEPTION, perception_out)
        self.bus.publish(Topic.COSTMAP, costmap)
        self.bus.publish(Topic.GLOBAL_PATH, global_path)
        self.bus.publish(Topic.LOCAL_TRAJECTORY, local_traj)
        self.bus.publish(Topic.CONTROL_COMMAND, command)

    def _emergency_stop_command(
        self,
        now: float,
        state: VehicleState,
        reason: str,
    ) -> ControlCommand:
        print(f"[RUNTIME FAULT] Emergency stop triggered: {reason}")

        return ControlCommand(
            header=Header(
                stamp=now,
                frame_id=FrameId.VEHICLE,
                source="integration_runtime",
            ),
            steering_angle=state.steering_angle,
            steering_rate=0.0,
            acceleration=-3.0,
            brake=1.0,
            mode=ControlMode.EMERGENCY_STOP,
        )