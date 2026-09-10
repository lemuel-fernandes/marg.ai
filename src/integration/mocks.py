import math
from typing import List

from src.common.types.base import (
    Covariance2D,
    FrameId,
    Header,
    Pose2D,
    Twist2D,
)
from src.common.types.config import PlannerConfig, VehicleConfig
from src.common.types.control import ControlCommand, ControlMode
from src.common.types.costmap import Costmap
from src.common.types.obstacle import Obstacle, ObstacleBehavior, ObstacleClass
from src.common.types.path import GlobalPath, PathPoint
from src.common.types.perception import PerceptionOutput
from src.common.types.sensor import SensorFrame
from src.common.types.trajectory import LocalTrajectory, TrajectoryPoint
from src.common.types.vehicle_state import VehicleState


class MockPerception:
    def process(self, frame: SensorFrame) -> PerceptionOutput:
        header = Header(
            stamp=frame.header.stamp,
            frame_id=FrameId.VEHICLE,
            source="mock_perception",
        )

        obstacle = Obstacle(
            header=Header(
                stamp=frame.header.stamp,
                frame_id=FrameId.VEHICLE,
                source="mock_perception",
            ),
            track_id=1,
            class_label=ObstacleClass.ANIMAL,
            behavior=ObstacleBehavior.CROSSING,
            pose=Pose2D(x=10.0, y=0.0, heading=0.0),
            length=1.2,
            width=0.8,
            velocity=Twist2D(vx=0.0, vy=0.4, yaw_rate=0.0),
            pose_covariance=Covariance2D(),
            velocity_covariance=Covariance2D(),
            confidence=0.9,
            is_dynamic=True,
        )

        return PerceptionOutput(
            header=header,
            obstacles=[obstacle],
            latency_ms=1.0,
            status="OK",
        )


class MockGlobalPlanner:
    def plan(
        self,
        costmap: Costmap,
        start: Pose2D,
        goal: Pose2D,
    ) -> GlobalPath:
        dx = goal.x - start.x
        dy = goal.y - start.y
        length = math.hypot(dx, dy)
        heading = 0.0 if length < 1e-6 else math.atan2(dy, dx)

        points: List[PathPoint] = [
            PathPoint(
                pose=start,
                curvature=0.0,
                target_speed=5.0,
            ),
            PathPoint(
                pose=goal,
                curvature=0.0,
                target_speed=5.0,
            ),
        ]

        return GlobalPath(
            header=Header(
                stamp=costmap.header.stamp,
                frame_id=FrameId.MAP,
                source="mock_global_planner",
            ),
            points=points,
            length_m=length,
            is_feasible=True,
            replan_required=False,
            reason="mock plan",
        )


class MockLocalPlanner:
    def __init__(self, cfg: PlannerConfig, vehicle_cfg: VehicleConfig):
        self.cfg = cfg
        self.vehicle_cfg = vehicle_cfg

    def plan(
        self,
        state: VehicleState,
        global_path: GlobalPath,
        costmap: Costmap,
        obstacles: List[Obstacle],
    ) -> LocalTrajectory:
        dt = self.cfg.local_dt
        horizon = self.cfg.local_horizon_s
        n_steps = int(horizon / dt)

        speed = min(max(state.twist.vx, 0.0), self.cfg.target_speed)
        x = state.pose.x
        y = state.pose.y
        heading = state.pose.heading

        points: List[TrajectoryPoint] = []

        for i in range(n_steps + 1):
            t = i * dt
            points.append(
                TrajectoryPoint(
                    t=t,
                    pose=Pose2D(x=x, y=y, heading=heading),
                    twist=Twist2D(vx=speed, vy=0.0, yaw_rate=0.0),
                    curvature=0.0,
                    acceleration=0.0,
                )
            )

            x += speed * math.cos(heading) * dt
            y += speed * math.sin(heading) * dt

        return LocalTrajectory(
            header=Header(
                stamp=state.header.stamp,
                frame_id=FrameId.MAP,
                source="mock_local_planner",
            ),
            points=points,
            cost=0.0,
            is_safe=True,
            fallback_active=False,
            reason="mock trajectory",
        )

    def emergency_stop(self, state: VehicleState) -> LocalTrajectory:
        dt = 0.1
        n_steps = 30
        speed = max(state.twist.vx, 0.0)

        x = state.pose.x
        y = state.pose.y
        heading = state.pose.heading

        points: List[TrajectoryPoint] = []

        for i in range(n_steps + 1):
            t = i * dt
            v_curr = max(speed, 0.0)
            accel = -2.0 if v_curr > 1e-3 else 0.0

            points.append(
                TrajectoryPoint(
                    t=t,
                    pose=Pose2D(x=x, y=y, heading=heading),
                    twist=Twist2D(vx=v_curr, vy=0.0, yaw_rate=0.0),
                    curvature=0.0,
                    acceleration=accel,
                )
            )

            x += v_curr * math.cos(heading) * dt
            y += v_curr * math.sin(heading) * dt
            speed = max(0.0, speed - 2.0 * dt)

        return LocalTrajectory(
            header=Header(
                stamp=state.header.stamp,
                frame_id=FrameId.MAP,
                source="mock_local_planner",
            ),
            points=points,
            cost=0.0,
            is_safe=True,
            fallback_active=True,
            reason="emergency stop",
        )


class MockController:
    def command(
        self,
        state: VehicleState,
        trajectory: LocalTrajectory,
    ) -> ControlCommand:
        if trajectory.fallback_active:
            return ControlCommand(
                header=Header(
                    stamp=state.header.stamp,
                    frame_id=FrameId.VEHICLE,
                    source="mock_controller",
                ),
                steering_angle=state.steering_angle,
                steering_rate=0.0,
                acceleration=-2.0,
                brake=0.8,
                mode=ControlMode.EMERGENCY_STOP,
            )

        return ControlCommand(
            header=Header(
                stamp=state.header.stamp,
                frame_id=FrameId.VEHICLE,
                source="mock_controller",
            ),
            steering_angle=0.0,
            steering_rate=0.0,
            acceleration=0.5,
            brake=0.0,
            mode=ControlMode.NORMAL,
        )