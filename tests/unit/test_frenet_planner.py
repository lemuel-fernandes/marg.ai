import math
import numpy as np
import pytest

from src.common.coordinates.frenet import CartesianFrenetConverter
from src.local_planner.algorithms.indian_road_planner import IndianRoadPlanner
from src.local_planner.behavior_state_machine import BehaviorStateMachine, PlannerState
from src.perception.vision_pipeline import VisionPerceptionPipeline
from src.local_planner.vision_integrated_planner import VisionIntegratedPlanner


def test_cartesian_frenet_roundtrip():
    # Straight reference line along X-axis
    ref_pts = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
    converter = CartesianFrenetConverter(ref_pts)

    # Point at (25.0, 1.5) with velocity (4.0, 0.0)
    s, d, s_dot, d_dot = converter.to_frenet(25.0, 1.5, 4.0, 0.0)
    assert math.isclose(s, 25.0, abs_tol=1e-3)
    assert math.isclose(d, 1.5, abs_tol=1e-3)
    assert math.isclose(s_dot, 4.0, abs_tol=1e-3)
    assert math.isclose(d_dot, 0.0, abs_tol=1e-3)

    # Convert back to Cartesian
    cx, cy, c_th = converter.to_cartesian(s, d)
    assert math.isclose(cx, 25.0, abs_tol=1e-3)
    assert math.isclose(cy, 1.5, abs_tol=1e-3)
    assert math.isclose(c_th, 0.0, abs_tol=1e-3)


def test_indian_road_planner_polynomials():
    planner = IndianRoadPlanner(target_speed=10.0)

    # Lateral quintic test: start at d=0, finish at d=1.2 in T=2.0s
    lat_coeffs = planner._solve_quintic(0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 2.0)
    assert lat_coeffs is not None
    assert len(lat_coeffs) == 6
    # Evaluate at t=0 and t=2.0
    d_0 = lat_coeffs[0]
    t = 2.0
    d_T = sum(c * (t ** i) for i, c in enumerate(lat_coeffs))
    assert math.isclose(d_0, 0.0, abs_tol=1e-5)
    assert math.isclose(d_T, 1.2, abs_tol=1e-5)

    # Longitudinal quartic test: start at s=0, speed=5.0, reach speed=10.0 in T=2.0s
    long_coeffs = planner._solve_quartic(0.0, 5.0, 0.0, 10.0, 0.0, 2.0)
    assert long_coeffs is not None
    # Evaluate speed at t=0 and t=2.0
    v_0 = long_coeffs[1]
    v_T = long_coeffs[1] + 2*long_coeffs[2]*t + 3*long_coeffs[3]*(t**2) + 4*long_coeffs[4]*(t**3)
    assert math.isclose(v_0, 5.0, abs_tol=1e-5)
    assert math.isclose(v_T, 10.0, abs_tol=1e-5)


def test_behavior_state_machine_transitions():
    sm = BehaviorStateMachine(normal_target_speed=6.0, creep_speed=1.0)
    assert sm.state == PlannerState.LANE_KEEP

    # Step with clear road
    state, directives = sm.step(
        current_s=0.0, current_d=0.0, current_v=5.0,
        obstacles=[], gap_available=True, min_ttc=10.0
    )
    assert state == PlannerState.LANE_KEEP
    assert directives["target_speed"] == 6.0

    # Step with ahead obstacle -> transitions to NUDGE_CHECK -> NUDGE_PASSTHRU
    obs = [{'s': 15.0, 'd': 0.0, 'type': 'vehicle'}]
    state, directives = sm.step(
        current_s=0.0, current_d=0.0, current_v=5.0,
        obstacles=obs, gap_available=True, min_ttc=5.0
    )
    assert state == PlannerState.NUDGE_CHECK

    state, directives = sm.step(
        current_s=1.0, current_d=0.0, current_v=4.0,
        obstacles=obs, gap_available=True, min_ttc=4.5
    )
    assert state == PlannerState.NUDGE_PASSTHRU
    assert directives["allow_lateral_nudge"] is True

    # Emergency trigger
    state, directives = sm.step(
        current_s=10.0, current_d=0.0, current_v=4.0,
        obstacles=obs, gap_available=False, min_ttc=0.6
    )
    assert state == PlannerState.EMERGENCY_REACTIVE_STEER
    assert directives["target_speed"] == 0.0


def test_vision_pipeline_projection_and_fallbacks():
    # Setup standard pinhole camera intrinsic & extrinsic
    K = np.array([[800.0, 0.0, 320.0],
                  [0.0, 800.0, 240.0],
                  [0.0, 0.0, 1.0]])
    # Camera mounted 1.5m above ground, looking forward (+X ego)
    # Camera coordinate: X right, Y down, Z forward
    # Ego coordinate (base_link): X forward, Y left, Z up
    R_c2v = np.array([[0.0, 0.0, 1.0],
                     [-1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0]])
    T_c2v = np.array([2.0, 0.0, 1.5])
    RT = np.column_stack([R_c2v, T_c2v])

    pipeline = VisionPerceptionPipeline(K, RT)

    raw_dets = [
        {
            'id': 10,
            'class': 'two_wheeler',
            'bbox_3d_cam': [0.0, 0.0, 10.0],  # 10m forward in camera
            'velocity_cam': [0.0, -2.0]        # moving toward camera
        }
    ]

    spatial_obs = pipeline.process_3d_detections(raw_dets, current_time=0.0)
    assert len(spatial_obs) == 1
    obs = spatial_obs[0]
    # Forward distance in base_link: T_x (2.0) + Z_c (10.0) = 12.0m
    assert math.isclose(obs['x_v'], 12.0, abs_tol=1e-3)
    assert obs['type'] == 'two_wheeler'

    # Test tracking persistence fallback: obstacle disappears at t=0.5
    persisted_obs = pipeline.process_3d_detections([], current_time=0.5)
    assert len(persisted_obs) == 1
    assert persisted_obs[0]['id'] == 10
    assert persisted_obs[0]['is_persisted'] is True

    # Test occlusion speed cap fallback
    v_cap = pipeline.compute_occlusion_speed_cap(d_visible=10.0, a_comfortable=2.0)
    # v_cap = sqrt(2 * 2.0 * 10.0) = sqrt(40) ~= 6.32 m/s
    assert math.isclose(v_cap, math.sqrt(40.0), abs_tol=1e-2)
