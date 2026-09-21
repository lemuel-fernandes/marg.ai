import pytest
from src.integration.safety_monitor import EnvelopeSafetyMonitor
from src.common.types.base import FrameId, Header, Pose2D, Twist2D
from src.common.types.control import ControlCommand, ControlMode
from src.common.types.trajectory import LocalTrajectory, TrajectoryPoint
from src.common.types.vehicle_state import VehicleState

def test_safety_rejects_overspeed_trajectory(vehicle_cfg, dummy_vehicle_state):
    monitor = EnvelopeSafetyMonitor(vehicle_cfg)
    
    # Trajectory moving at 15 m/s (limit is 10 m/s)
    bad_traj = LocalTrajectory(
        header=Header(stamp=0.0, frame_id=FrameId.MAP, source="test"),
        points=[
            TrajectoryPoint(t=0.0, pose=Pose2D(0,0,0), twist=Twist2D(vx=15.0), curvature=0.0, acceleration=0.0)
        ]
    )
    
    decision = monitor.evaluate(dummy_vehicle_state, bad_traj, [], None)
    assert decision.allowed is False
    assert "speed" in decision.reason.lower()

def test_safety_clamps_control_command(vehicle_cfg, dummy_vehicle_state):
    monitor = EnvelopeSafetyMonitor(vehicle_cfg)
    
    # Command asks for 5.0 m/s^2 accel (limit is 2.0) and 1.0 rad steer (limit is 0.5)
    bad_cmd = ControlCommand(
        header=Header(stamp=0.0, frame_id=FrameId.VEHICLE, source="test"),
        steering_angle=1.0,
        steering_rate=0.0,
        acceleration=5.0,
        brake=0.0,
        mode=ControlMode.NORMAL
    )
    
    # Mock a valid decision just to pass limits to the clamp function
    valid_decision = monitor.evaluate(
        dummy_vehicle_state, 
        LocalTrajectory(header=Header(0.0, FrameId.MAP, "t"), points=[]), # empty is rejected, but we bypass evaluate here
        [], None
    )
    # Actually, let's just make a valid decision manually to test clamping:
    from src.common.types.safety import SafetyDecision
    valid_limits = SafetyDecision(
        allowed=True, reason="ok", 
        max_acceleration=vehicle_cfg.max_acceleration,
        min_acceleration=vehicle_cfg.min_acceleration,
        max_abs_steer=vehicle_cfg.max_steer_angle
    )

    clamped_cmd = monitor.limit_command(dummy_vehicle_state, bad_cmd, valid_limits)
    
    assert clamped_cmd.acceleration == vehicle_cfg.max_acceleration
    assert clamped_cmd.steering_angle == vehicle_cfg.max_steer_angle