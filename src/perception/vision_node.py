"""Vision perception node: wires the synthetic camera + vision pipeline into
the IntegrationPipeline `PerceptionNode` contract (ROADMAP vision wiring).

Data path per tick:
    GT sim obstacles --SyntheticCameraAdapter--> camera-frame detections +
    seg/depth --VisionPerceptionPipeline--> base_link dicts --this node-->
    FrameId.VEHICLE obstacles --TransformTree--> map frame

Ground-truth fallback: obstacles outside the camera FOV/range (e.g. behind
the vehicle) are passed through from the GT provider as FrameId.MAP, so the
planner never loses non-visible obstacles. Camera-visible ones are
reconstructed through the full vision path (validating the geometry) and
emitted as FrameId.VEHICLE.
"""

from typing import Callable, List, Tuple

import numpy as np
import random

from src.common.types.base import Covariance2D, FrameId, Header, Pose2D, Twist2D
from src.common.types.obstacle import Obstacle, ObstacleBehavior, ObstacleClass
from src.common.types.perception import PerceptionOutput
from src.common.types.sensor import SensorFrame
from src.perception.synthetic_camera import SyntheticCameraAdapter
from src.perception.vision_pipeline import VisionPerceptionPipeline


_CLASS_BY_VALUE = {c.value: c for c in ObstacleClass}


class VisionPerceptionNode:
    """PerceptionNode implementation backed by the synthetic camera.

    The TransformTree reference is read-only: `process` pulls the current ego
    pose (updated by the pipeline before perception each tick) to drive the
    adapter, exactly as a real sensor driver would consume odom.
    """

    def __init__(self,
                 obstacles_provider: Callable[[], List[Obstacle]],
                 camera_intrinsic: np.ndarray,
                 camera_extrinsics_rt: np.ndarray,
                 transform_tree,
                 fov_rad: float = 1.2,
                 max_range_m: float = 60.0,
                 seg_shape: Tuple[int, int] = (480, 640),
                 pos_noise_std: float = 0.0,
                 vel_noise_std: float = 0.0):
        self.provider = obstacles_provider
        self.tf = transform_tree
        self.adapter = SyntheticCameraAdapter(
            camera_intrinsic, fov_rad=fov_rad, max_range_m=max_range_m)
        self.vision = VisionPerceptionPipeline(camera_intrinsic,
                                               camera_extrinsics_rt)
        self.seg_shape = seg_shape
        # Optional sensor-grade noise injected into vision-path reconstructions
        # (base_link position/velocity, mirrors NoisyPerception semantics).
        # GT passthrough stays clean: it models a fallback modality, not the
        # noisy camera. Dedicated seeded RNG keeps runs reproducible without
        # touching the global random state (scenarios seed it for layout).
        self.pos_noise_std = float(pos_noise_std)
        self.vel_noise_std = float(vel_noise_std)
        self._rng = random.Random(1234)

    def process(self, frame: SensorFrame) -> PerceptionOutput:
        t = frame.header.stamp
        ego_state = self.tf.ego_state
        if ego_state is None:
            raise RuntimeError(
                "VisionPerceptionNode requires ego state in TransformTree")
        ego_pose = ego_state.pose
        gt_obstacles = self.provider()

        # 0. Road-surface anomalies (potholes/speed breakers) the seg/depth
        # path can extract this tick. They are kept OFF the 3D-detection path
        # and OFF the GT passthrough so each one reaches the planner through
        # exactly one route — the segmentation branch.
        seg_anomaly_ids = {o.track_id for o in
                           self.adapter.anomalies_in_view(gt_obstacles, ego_pose)}

        # 1. GT -> camera frame (adapter) -> base_link (vision pipeline).
        # ego_pose anchors track persistence in the map frame so lost-track
        # extrapolation compensates for ego motion while the camera is blind
        # (e.g. an obstacle exiting the FOV cone alongside the vehicle).
        solid = [o for o in gt_obstacles if o.track_id not in seg_anomaly_ids]
        dets = self.adapter.make_detections(solid, ego_pose)
        seg, depth = self.adapter.make_seg_and_depth(
            gt_obstacles, ego_pose, width=self.seg_shape[1],
            height=self.seg_shape[0])
        spatial = self.vision.process_3d_detections(dets, current_time=t,
                                                    ego_pose=ego_pose)

        # 2. base_link dicts -> Obstacle records (FrameId.VEHICLE).
        vision_obstacles: List[Obstacle] = []
        for sp in spatial:
            # Sensor noise on the measured base_link state (no-op at std=0).
            if self.pos_noise_std > 0.0:
                sp['x_v'] += self._rng.gauss(0.0, self.pos_noise_std)
                sp['y_v'] += self._rng.gauss(0.0, self.pos_noise_std)
            if self.vel_noise_std > 0.0:
                sp['vx_v'] += self._rng.gauss(0.0, self.vel_noise_std)
                sp['vy_v'] += self._rng.gauss(0.0, self.vel_noise_std)
            cls = _CLASS_BY_VALUE.get(str(sp.get('type', 'unknown')),
                                      ObstacleClass.UNKNOWN)
            behavior = sp.get('behavior', ObstacleBehavior.STATIC)
            is_dynamic = bool(sp.get('is_dynamic',
                                     abs(sp.get('vx_v', 0.0))
                                     + abs(sp.get('vy_v', 0.0)) > 0.1))
            vision_obstacles.append(Obstacle(
                header=Header(t, FrameId.VEHICLE, "vision_pipeline",
                              seq=sp['id']),
                track_id=sp['id'],
                class_label=cls,
                behavior=behavior,
                pose=Pose2D(x=sp['x_v'], y=sp['y_v'],
                            heading=float(sp.get('rel_heading', 0.0))),
                length=sp.get('length', 2.0),
                width=sp.get('width', 1.5),
                velocity=Twist2D(vx=sp.get('vx_v', 0.0),
                                 vy=sp.get('vy_v', 0.0)),
                pose_covariance=Covariance2D(),
                velocity_covariance=Covariance2D(),
                confidence=float(sp.get('confidence', 0.9)),
                is_dynamic=is_dynamic,
            ))

        # 3. Road-surface anomalies via the seg/depth branch: cluster the
        # segmentation mask into blobs, unproject each to base_link, and emit
        # one Obstacle per blob. Synthetic ids (10_000+) are stable within a
        # tick: blobs are sorted by position, and static potholes keep their
        # relative order as the ego advances. Heading is -ego_heading so the
        # vehicle->map transform restores the GT convention (map heading 0).
        anomaly_obstacles: List[Obstacle] = []
        if seg is not None:
            blobs = self.vision.process_surface_anomalies(seg, depth)
            blobs.sort(key=lambda b: (round(b['x_v'], 1), round(b['y_v'], 1)))
            for i, b in enumerate(blobs):
                cls = _CLASS_BY_VALUE.get(str(b.get('type', 'unknown')),
                                          ObstacleClass.UNKNOWN)
                anomaly_obstacles.append(Obstacle(
                    header=Header(t, FrameId.VEHICLE, "vision_seg_depth",
                                  seq=10_000 + i),
                    track_id=10_000 + i,
                    class_label=cls,
                    behavior=ObstacleBehavior.STATIC,
                    pose=Pose2D(x=b['x_v'], y=b['y_v'],
                                heading=-ego_pose.heading),
                    length=b.get('length', 0.5),
                    width=b.get('width', 0.5),
                    velocity=Twist2D(),
                    pose_covariance=Covariance2D(),
                    velocity_covariance=Covariance2D(),
                    confidence=0.75,   # segmentation-derived, lower than 3D dets
                    is_dynamic=False,
                ))

        # 4. GT passthrough for obstacles the camera cannot see: not visible
        # to the camera at all (outside FOV/range, behind), not reconstructable
        # this tick, or an anomaly beyond the depth-extraction band. They keep
        # FrameId.MAP so the TransformTree passes them through untouched.
        seen_ids = {sp['id'] for sp in spatial}
        fallback = [o for o in gt_obstacles
                    if o.track_id not in seen_ids
                    and o.track_id not in seg_anomaly_ids]

        return PerceptionOutput(
            header=Header(t, FrameId.MAP, "vision_perception_node"),
            obstacles=vision_obstacles + anomaly_obstacles + fallback,
            latency_ms=1.0,
            status="OK",
        )
