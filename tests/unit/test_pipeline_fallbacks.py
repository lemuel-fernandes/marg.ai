import pytest
from src.common.types.base import FrameId, Header, Pose2D
from src.common.types.control import ControlCommand, ControlMode
from src.common.types.sensor import SensorFrame
from src.common.types.vehicle_state import VehicleState
from src.integration.contracts import LocalPlanner
from src.integration.system_runner import build_pipeline
from src.common.types.trajectory import LocalTrajectory

class FailingLocalPlanner:
    """Mock planner that crashes."""
    def plan(self, *args, **kwargs):
        raise RuntimeError("Simulated M4 Planner Crash")
        
    def emergency_stop(self, state):
        from src.common.types.trajectory import TrajectoryPoint
        from src.common.types.base import Twist2D
        return LocalTrajectory(
            header=Header(state.header.stamp, FrameId.MAP, "fallback"),
            points=[TrajectoryPoint(0.0, state.pose, Twist2D(), 0.0, -3.0)],
            is_safe=True, fallback_active=True
        )

def test_pipeline_failsafe_on_planner_crash(dummy_vehicle_state, dummy_sensor_frame):
    pipeline = build_pipeline()
    
    # Inject the failing planner
    pipeline.local_planner = FailingLocalPlanner()
    
    goal = Pose2D(x=20.0, y=0.0, heading=0.0)
    
    # Run the tick
    command = pipeline.tick(
        now=0.0,
        vehicle_state=dummy_vehicle_state,
        sensor_frame=dummy_sensor_frame,
        goal=goal
    )
    
    # The pipeline MUST catch the exception and return an EMERGENCY_STOP command
    assert isinstance(command, ControlCommand)
    assert command.mode == ControlMode.EMERGENCY_STOP
    assert command.brake > 0.0