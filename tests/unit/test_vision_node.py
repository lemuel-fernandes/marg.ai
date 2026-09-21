"""Unit tests for the synthetic camera -> vision pipeline -> map roundtrip."""
import math

import numpy as np
import pytest

from src.common.types.base import Covariance2D, FrameId, Header, Pose2D, Twist2D
from src.common.types.obstacle import Obstacle, ObstacleBehavior, ObstacleClass
from src.common.types.sensor import SensorFrame
from src.common.coordinates.transforms import TransformTree
from src.perception.synthetic_camera import SyntheticCameraAdapter
from src.perception.vision_node import VisionPerceptionNode
from src.perception.vision_pipeline import VisionPerceptionPipeline


K = np.array([[320.0, 0.0, 320.0],
              [0.0, 320.0, 240.0],
              [0.0, 0.0, 1.0]])
# Camera looking forward along base_link X, standard axis remap, T=0.
R_C2V = np.array([[0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]])
RT = np.column_stack([R_C2V, np.zeros(3)])


def _obs(tid, x, y, heading=0.0, vx=0.0, vy=0.0,
         cls=ObstacleClass.VEHICLE, L=4.0, W=2.0, dyn=True):
    return Obstacle(
        header=Header(0.0, FrameId.MAP, "test"), track_id=tid,
        class_label=cls, behavior=ObstacleBehavior.STATIC,
        pose=Pose2D(x, y, heading), length=L, width=W,
        velocity=Twist2D(vx=vx, vy=vy, yaw_rate=0.0),
        pose_covariance=Covariance2D(), velocity_covariance=Covariance2D(),
        confidence=0.9, is_dynamic=dyn)


EGO = Pose2D(10.0, 5.0, math.pi / 2)  # heading north


def test_roundtrip_obstacle_position_and_velocity():
    """map -> camera -> vision pipeline -> base_link == analytic vehicle frame."""
    adapter = SyntheticCameraAdapter(K)
    # Obstacle 20m ahead of ego (north), 3m to its left.
    obs = _obs(1, EGO.x - 3.0, EGO.y + 20.0, vx=0.0, vy=-1.0)
    dets = adapter.make_detections([obs], EGO)
    assert len(dets) == 1

    vp = VisionPerceptionPipeline(K, RT)
    spatial = vp.process_3d_detections(dets, current_time=0.0)
    assert len(spatial) == 1
    sp = spatial[0]

    # Analytic base_link: forward 20, left 3 (minus ego-size offset T=0).
    assert math.isclose(sp['x_v'], 20.0, abs_tol=1e-6)
    assert math.isclose(sp['y_v'], 3.0, abs_tol=1e-6)
    # Obstacle moving south while ego faces north -> backward in base_link:
    # vx_v = -1, vy_v = 0.
    assert math.isclose(sp['vx_v'], -1.0, abs_tol=1e-6)
    assert math.isclose(sp['vy_v'], 0.0, abs_tol=1e-6)
    # Class string survives.
    assert sp['type'] == 'vehicle'


def test_fov_and_range_gating():
    adapter = SyntheticCameraAdapter(K, fov_rad=0.5, max_range_m=30.0)
    ahead = _obs(1, EGO.x, EGO.y + 15.0)          # ahead, in FOV
    behind = _obs(2, EGO.x, EGO.y - 15.0)         # behind -> gated
    far = _obs(3, EGO.x, EGO.y + 80.0)            # beyond max range
    dets = adapter.make_detections([ahead, behind, far], EGO)
    assert [d['id'] for d in dets] == [1]


def test_fov_gating_at_edge():
    adapter = SyntheticCameraAdapter(K, fov_rad=0.4)
    # 16 deg off the forward axis: inside the 0.4 rad (~23 deg) half-FOV.
    off = _obs(1, EGO.x - 5.0, EGO.y + 17.32)
    dets = adapter.make_detections([off], EGO)
    assert len(dets) == 1


def test_vision_node_reconstructs_and_falls_back():
    tf = TransformTree()
    from src.common.types.vehicle_state import VehicleState
    tf.update_ego_state(VehicleState(
        header=Header(0.0, FrameId.MAP, "test"), pose=EGO,
        twist=Twist2D(vx=2.0), steering_angle=0.0))

    ahead = _obs(1, EGO.x, EGO.y + 20.0, heading=0.5)
    behind = _obs(2, EGO.x, EGO.y - 10.0, cls=ObstacleClass.POTHOLE,
                  L=0.8, W=0.8, dyn=False)
    provider = lambda: [ahead, behind]

    node = VisionPerceptionNode(provider, K, RT, tf)
    out = node.process(SensorFrame(header=Header(1.0, FrameId.SENSOR_FRONT, "t")))
    by_id = {o.track_id: o for o in out.obstacles}

    # Visible obstacle came through the vision path (VEHICLE frame).
    assert by_id[1].header.frame_id == FrameId.VEHICLE
    assert by_id[1].class_label == ObstacleClass.VEHICLE
    # Relative heading preserved (obstacle 0.5 - ego pi/2).
    assert math.isclose(by_id[1].pose.heading, 0.5 - math.pi / 2, abs_tol=1e-6)

    # Behind obstacle fell back to GT (MAP frame, untouched).
    assert by_id[2].header.frame_id == FrameId.MAP
    assert math.isclose(by_id[2].pose.x, behind.pose.x, abs_tol=1e-9)
    assert by_id[2].class_label == ObstacleClass.POTHOLE


def test_transform_tree_converts_vision_obstacles_to_map():
    """The full loop: VEHICLE-frame vision output == original map position."""
    tf = TransformTree()
    from src.common.types.vehicle_state import VehicleState
    tf.update_ego_state(VehicleState(
        header=Header(0.0, FrameId.MAP, "test"), pose=EGO,
        twist=Twist2D(vx=2.0), steering_angle=0.0))

    target = _obs(1, EGO.x - 3.0, EGO.y + 20.0)
    node = VisionPerceptionNode(lambda: [target], K, RT, tf)
    out = node.process(SensorFrame(header=Header(1.0, FrameId.SENSOR_FRONT, "t")))

    mapped = tf.obstacles_to_map(out.obstacles)
    assert len(mapped) == 1
    m = mapped[0]
    assert math.isclose(m.pose.x, target.pose.x, abs_tol=1e-6)
    assert math.isclose(m.pose.y, target.pose.y, abs_tol=1e-6)
    assert math.isclose(m.pose.heading, target.pose.heading, abs_tol=1e-6)
    # Velocity roundtrip
    assert math.isclose(m.velocity.vx, target.velocity.vx, abs_tol=1e-6)
    assert math.isclose(m.velocity.vy, target.velocity.vy, abs_tol=1e-6)


def test_persistence_compensates_ego_motion():
    """Lost-track persistence must anchor in the map frame, not base_link.

    A base_link snapshot goes stale as the ego drives: extrapolating it with
    only the obstacle's velocity makes a persisted track drift backward along
    the road at the ego's speed (a phantom cutting across the ego's path).
    """
    adapter = SyntheticCameraAdapter(K, fov_rad=0.5)
    vp = VisionPerceptionPipeline(K, RT)

    ego_t0 = Pose2D(0.0, 0.0, 0.0)            # facing +x
    obs = _obs(7, 10.0, 2.0, vx=1.0, vy=0.0)  # 10m ahead, 2m left, moving +x
    dets = adapter.make_detections([obs], ego_t0)
    assert len(dets) == 1
    vp.process_3d_detections(dets, current_time=0.0, ego_pose=ego_t0)

    # Ego drives 4m forward and turns 90 deg left; obstacle leaves FOV cone.
    ego_t1 = Pose2D(4.0, 0.0, math.pi / 2)
    out = vp.process_3d_detections([], current_time=1.0, ego_pose=ego_t1)
    assert len(out) == 1 and out[0]['is_persisted'] is True

    # True world position after 1s of CV extrapolation: (11.0, 2.0).
    # Re-projected into the new base_link (heading pi/2, facing +y_map):
    # map +x is to the right -> x_v = 2.0, y_v = -7.0.
    assert math.isclose(out[0]['x_v'], 2.0, abs_tol=1e-6)
    assert math.isclose(out[0]['y_v'], -7.0, abs_tol=1e-6)
    # Map velocity (1, 0) seen from heading pi/2 -> 1 m/s to the right.
    assert math.isclose(out[0]['vx_v'], 0.0, abs_tol=1e-6)
    assert math.isclose(out[0]['vy_v'], -1.0, abs_tol=1e-6)

    # Roundtrip through the transform tree at the new ego pose recovers truth.
    tf = TransformTree()
    from src.common.types.vehicle_state import VehicleState
    tf.update_ego_state(VehicleState(
        header=Header(1.0, FrameId.MAP, "test"), pose=ego_t1,
        twist=Twist2D(vx=2.0), steering_angle=0.0))
    node = VisionPerceptionNode(lambda: [], K, RT, tf)
    node.vision = vp  # share the warmed-up track store
    out2 = node.process(SensorFrame(header=Header(1.0, FrameId.SENSOR_FRONT, "t")))
    mapped = tf.obstacles_to_map(out2.obstacles)
    assert len(mapped) == 1
    assert math.isclose(mapped[0].pose.x, 11.0, abs_tol=1e-6)
    assert math.isclose(mapped[0].pose.y, 2.0, abs_tol=1e-6)
    assert math.isclose(mapped[0].velocity.vx, 1.0, abs_tol=1e-6)
    assert math.isclose(mapped[0].velocity.vy, 0.0, abs_tol=1e-6)


def test_near_pothole_rides_seg_depth_path_without_duplicates():
    """A segmentable pothole reaches the planner via seg/depth, exactly once."""
    tf = TransformTree()
    from src.common.types.vehicle_state import VehicleState
    tf.update_ego_state(VehicleState(
        header=Header(0.0, FrameId.MAP, "test"), pose=EGO,
        twist=Twist2D(vx=2.0), steering_angle=0.0))

    pothole = _obs(100, EGO.x, EGO.y + 12.0, cls=ObstacleClass.POTHOLE,
                   L=0.8, W=0.8, dyn=False)
    node = VisionPerceptionNode(lambda: [pothole], K, RT, tf)
    out = node.process(SensorFrame(header=Header(1.0, FrameId.SENSOR_FRONT, "t")))

    # Exactly ONE obstacle for the pothole (no 3D-path or GT duplicate).
    assert len(out.obstacles) == 1
    o = out.obstacles[0]
    assert o.header.source == "vision_seg_depth"
    assert o.header.frame_id == FrameId.VEHICLE
    assert o.class_label == ObstacleClass.POTHOLE
    assert not o.is_dynamic
    # Segmentation-derived size estimate: sane anomaly footprint.
    assert 0.3 <= o.length <= 1.5

    # Roundtrip through the transform tree lands at the true position
    # (blob centroid vs GT center; pixel-grid rounding only).
    mapped = tf.obstacles_to_map([o])[0]
    assert math.isclose(mapped.pose.x, pothole.pose.x, abs_tol=0.15)
    assert math.isclose(mapped.pose.y, pothole.pose.y, abs_tol=0.15)


def test_far_pothole_stays_on_3d_detection_path():
    """Beyond the depth-extraction band (25m), potholes fall back to 3D dets."""
    tf = TransformTree()
    from src.common.types.vehicle_state import VehicleState
    tf.update_ego_state(VehicleState(
        header=Header(0.0, FrameId.MAP, "test"), pose=EGO,
        twist=Twist2D(vx=2.0), steering_angle=0.0))

    pothole = _obs(101, EGO.x, EGO.y + 40.0, cls=ObstacleClass.POTHOLE,
                   L=0.8, W=0.8, dyn=False)  # visible, but depth-unextractable
    node = VisionPerceptionNode(lambda: [pothole], K, RT, tf)
    out = node.process(SensorFrame(header=Header(1.0, FrameId.SENSOR_FRONT, "t")))

    assert len(out.obstacles) == 1
    o = out.obstacles[0]
    assert o.header.source == "vision_pipeline"
    assert o.header.frame_id == FrameId.VEHICLE
    assert o.class_label == ObstacleClass.POTHOLE
    # True geometry via the 3D roundtrip, not a blob estimate.
    mapped = tf.obstacles_to_map([o])[0]
    assert math.isclose(mapped.pose.x, pothole.pose.x, abs_tol=1e-6)


def test_two_potholes_yield_two_distinct_blob_obstacles():
    tf = TransformTree()
    from src.common.types.vehicle_state import VehicleState
    tf.update_ego_state(VehicleState(
        header=Header(0.0, FrameId.MAP, "test"), pose=EGO,
        twist=Twist2D(vx=2.0), steering_angle=0.0))

    p1 = _obs(102, EGO.x, EGO.y + 10.0, cls=ObstacleClass.POTHOLE,
              L=0.8, W=0.8, dyn=False)
    p2 = _obs(103, EGO.x + 2.0, EGO.y + 10.0, cls=ObstacleClass.POTHOLE,
              L=0.8, W=0.8, dyn=False)
    node = VisionPerceptionNode(lambda: [p1, p2], K, RT, tf)
    out = node.process(SensorFrame(header=Header(1.0, FrameId.SENSOR_FRONT, "t")))

    blobs = [o for o in out.obstacles if o.header.source == "vision_seg_depth"]
    assert len(blobs) == 2
    assert len({o.track_id for o in blobs}) == 2
    # Lateral separation preserved in vehicle frame (y_v = left offset).
    y_vals = sorted(o.pose.y for o in blobs)
    assert math.isclose(y_vals[1] - y_vals[0], 2.0, abs_tol=0.3)


def test_seg_and_depth_render_potholes():
    adapter = SyntheticCameraAdapter(K)
    pothole = _obs(1, EGO.x, EGO.y + 12.0, cls=ObstacleClass.POTHOLE,
                   L=0.8, W=0.8, dyn=False)
    car = _obs(2, EGO.x, EGO.y + 25.0)
    seg, depth = adapter.make_seg_and_depth([pothole, car], EGO)
    assert seg is not None and depth is not None
    assert (seg == 2).sum() > 0          # pothole blob stamped
    assert (seg == 3).sum() == 0         # car not in seg mask
    # No anomalies -> (None, None)
    seg2, depth2 = adapter.make_seg_and_depth([car], EGO)
    assert seg2 is None and depth2 is None
