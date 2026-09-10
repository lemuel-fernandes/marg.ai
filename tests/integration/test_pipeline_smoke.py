from src.common.types.base import FrameId, Header, Pose2D, Twist2D
from src.common.types.control import ControlCommand, ControlMode
from src.common.types.sensor import SensorFrame
from src.common.types.vehicle_state import VehicleState
from src.integration.system_runner import build_pipeline


def _make_inputs():
    now = 0.0

    header = Header(
        stamp=now,
        frame_id=FrameId.MAP,
        source="test",
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
            source="test",
        )
    )

    goal = Pose2D(x=20.0, y=0.0, heading=0.0)

    return now, vehicle_state, sensor_frame, goal


def test_pipeline_returns_control_command():
    pipeline = build_pipeline()
    now, vehicle_state, sensor_frame, goal = _make_inputs()

    command = pipeline.tick(
        now=now,
        vehicle_state=vehicle_state,
        sensor_frame=sensor_frame,
        goal=goal,
    )

    assert isinstance(command, ControlCommand)
    assert command.mode in {ControlMode.NORMAL, ControlMode.EMERGENCY_STOP}