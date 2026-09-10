"""End-to-end planning pipeline."""

from typing import List, Optional

from src.common.types.base import FrameId, Header, Pose2D
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
from .message_bus import Topic, TypedMessageBus
from .state_manager import SystemStateManager


class IntegrationFault(RuntimeError):
    pass


class IntegrationPipeline:
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
        state_manager: Optional[SystemStateManager] = None,
    ):
        self.perception = perception
        self.costmap_builder = costmap_builder
        self.global_planner = global_planner
        self.local_planner = local_planner
        self.controller = controller
        self.safety_monitor = safety_monitor
        self.tf = transform_tree
        self.bus = message_bus
        self.state_manager = state_manager

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
        if self.state_manager is not None:
            self.state_manager.set_running()

        self.tf.update_ego_state(vehicle_state)

        try:
            # 1. Perception
            perception_out = self.perception.process(sensor_frame)
            map_obstacles = self.tf.obstacles_to_map(perception_out.obstacles)
            perception_map = PerceptionOutput(
                header=Header(
                    stamp=now,
                    frame_id=FrameId.MAP,
                    source="integration",
                ),
                obstacles=map_obstacles,
                latency_ms=perception_out.latency_ms,
                status=perception_out.status,
            )

            # 2. Costmap
            self._current_costmap = self.costmap_builder.update(
                obstacles=map_obstacles,
                previous_costmap=self._current_costmap,
                stamp=now,
                ego_pose=vehicle_state.pose,
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

            if not self._current_global_path.is_feasible:
                # Do not fault the pipeline; just issue a stop command and wait for replan
                return self._emergency_stop_command(now, vehicle_state, "Global path temporarily blocked")
      
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
                local_traj = self._safe_emergency_stop(vehicle_state)
                decision = self.safety_monitor.evaluate(
                    state=vehicle_state,
                    trajectory=local_traj,
                    obstacles=map_obstacles,
                    costmap=self._current_costmap,
                )

                if not decision.allowed:
                    return self._emergency_stop_command(
                        now=now,
                        state=vehicle_state,
                        reason=f"Safety fallback rejected: {decision.reason}",
                    )

            # 6. Control
            command = self.controller.command(vehicle_state, local_traj)
            command = self.safety_monitor.limit_command(
                state=vehicle_state,
                command=command,
                decision=decision,
            )

            # 7. Telemetry
            self._publish_telemetry(
                perception_out=perception_map,
                costmap=self._current_costmap,
                global_path=self._current_global_path,
                local_trajectory=local_traj,
                command=command,
            )

            return command

        except Exception as exc:
            if self.state_manager is not None:
                self.state_manager.raise_fault("pipeline", str(exc))

            return self._emergency_stop_command(
                now=now,
                state=vehicle_state,
                reason=str(exc),
            )

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

    def _safe_emergency_stop(self, state: VehicleState) -> LocalTrajectory:
        try:
            return self.local_planner.emergency_stop(state)
        except Exception:
            return LocalTrajectory(
                header=Header(
                    stamp=state.header.stamp,
                    frame_id=FrameId.MAP,
                    source="pipeline_fallback",
                ),
                points=[],
                cost=float("inf"),
                is_safe=False,
                fallback_active=True,
                reason="Emergency fallback due to local planner failure",
            )

    def _publish_telemetry(
        self,
        perception_out: PerceptionOutput,
        costmap: Optional[Costmap],
        global_path: Optional[GlobalPath],
        local_trajectory: LocalTrajectory,
        command: ControlCommand,
    ) -> None:
        if self.bus is None:
            return

        self.bus.publish(Topic.PERCEPTION, perception_out)

        if costmap is not None:
            self.bus.publish(Topic.COSTMAP, costmap)

        if global_path is not None:
            self.bus.publish(Topic.GLOBAL_PATH, global_path)

        self.bus.publish(Topic.LOCAL_TRAJECTORY, local_trajectory)
        self.bus.publish(Topic.CONTROL_COMMAND, command)

    def _emergency_stop_command(
        self,
        now: float,
        state: VehicleState,
        reason: str,
    ) -> ControlCommand:
        print(f"[PIPELINE FAULT] Emergency stop triggered: {reason}")

        return ControlCommand(
            header=Header(
                stamp=now,
                frame_id=FrameId.VEHICLE,
                source="integration_pipeline",
            ),
            steering_angle=state.steering_angle,
            steering_rate=0.0,
            acceleration=-3.0,
            brake=1.0,
            mode=ControlMode.EMERGENCY_STOP,
        )