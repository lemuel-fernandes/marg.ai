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

CV-based extension (optional): when use_cv_detectors=True, the node can also
run learned 2D detectors, segmentation, depth estimation, and optical flow
on synthetic camera images. This enables testing the full CV perception stack.
"""

from typing import Callable, List, Tuple, Optional, Dict, Any
import numpy as np
import random
import math

from src.common.types.base import Covariance2D, FrameId, Header, Pose2D, Twist2D
from src.common.types.obstacle import Obstacle, ObstacleBehavior, ObstacleClass
from src.common.types.perception import PerceptionOutput
from src.common.types.sensor import SensorFrame
from src.perception.synthetic_camera import SyntheticCameraAdapter
from src.perception.vision_pipeline import VisionPerceptionPipeline
from src.perception.vision_detector import (
    SyntheticDetector, SyntheticSegmenter, SyntheticDepthEstimator,
    SyntheticOpticalFlow, create_synthetic_vision_stack,
    compute_2d_bbox_from_3d, non_max_suppression,
    INDIAN_TRAFFIC_CLASSES
)
try:
    from src.perception.yolo_detector import YOLOv8Detector, YOLOv8Segmenter, YOLOConfig
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from src.perception.weather_degradation import WeatherDegradationModel, WeatherCondition, TimeOfDay
    WEATHER_AVAILABLE = True
except ImportError:
    WEATHER_AVAILABLE = False


_CLASS_BY_VALUE = {c.value: c for c in ObstacleClass}

# Map ObstacleClass to detector class names
OBSTACLE_CLASS_TO_DETECTOR_CLASS = {
    ObstacleClass.VEHICLE: 'car',
    ObstacleClass.PEDESTRIAN: 'pedestrian',
    ObstacleClass.ANIMAL: 'cattle',  # Default for animals
    ObstacleClass.POTHOLE: 'pothole',
    ObstacleClass.ENCROACHMENT: 'construction_barrier',
    ObstacleClass.PARKED_VEHICLE: 'car',
    ObstacleClass.TWO_WHEELER: 'two_wheeler',
    ObstacleClass.UNKNOWN: 'car',
}

# Map ObstacleClass to detector class names
OBSTACLE_CLASS_TO_DETECTOR_CLASS = {
    ObstacleClass.VEHICLE: 'car',
    ObstacleClass.PEDESTRIAN: 'pedestrian',
    ObstacleClass.ANIMAL: 'cattle',  # Default for animals
    ObstacleClass.POTHOLE: 'pothole',
    ObstacleClass.ENCROACHMENT: 'construction_barrier',
    ObstacleClass.PARKED_VEHICLE: 'car',
    ObstacleClass.TWO_WHEELER: 'two_wheeler',
    ObstacleClass.UNKNOWN: 'car',
}


class VisionPerceptionNode:
    """PerceptionNode implementation backed by the synthetic camera.

    The TransformTree reference is read-only: `process` pulls the current ego
    pose (updated by the pipeline before perception each tick) to drive the
    adapter, exactly as a real sensor driver would consume odom.

    When use_cv_detectors=True, also runs a synthetic CV stack (detector,
    segmenter, depth, optical flow) to simulate a learned perception pipeline.
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
                 vel_noise_std: float = 0.0,
                 use_cv_detectors: bool = False,
                 cv_detector_config: Optional[Dict[str, Any]] = None,
                 weather_condition: Optional[str] = None,
                 time_of_day: Optional[str] = None):
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

        # Weather/lighting degradation model
        self.weather_model = None
        if weather_condition and WEATHER_AVAILABLE:
            try:
                w = WeatherCondition(weather_condition)
                t = TimeOfDay(time_of_day) if time_of_day else TimeOfDay.DAY
                self.weather_model = WeatherDegradationModel(w, t)
                print(f"Weather degradation enabled: {weather_condition}, time: {time_of_day or 'day'}")
            except ValueError:
                print(f"Unknown weather condition: {weather_condition}")

        # CV-based detectors (optional)
        self.use_cv_detectors = use_cv_detectors
        self.use_yolo = cv_detector_config.get('use_yolo', False) if cv_detector_config else False
        self.cv_stack = None
        self.yolo_detector = None
        self.yolo_segmenter = None
        self.prev_image = None
        if use_cv_detectors:
            if self.use_yolo and YOLO_AVAILABLE:
                # Initialize YOLOv8 detector
                yolo_config = YOLOConfig(
                    model_path=cv_detector_config.get('yolo_model_path', 'yolov8n.pt'),
                    confidence_threshold=cv_detector_config.get('confidence_threshold', 0.25),
                    device=cv_detector_config.get('device', 'cpu'),
                )
                self.yolo_detector = YOLOv8Detector(yolo_config)
                # Optional segmenter
                if cv_detector_config.get('use_segmentation', False):
                    self.yolo_segmenter = YOLOv8Segmenter(yolo_config)
                print(f"YOLOv8 detector initialized (loaded: {self.yolo_detector.model_loaded})")
            else:
                self.cv_stack = create_synthetic_vision_stack(
                    class_names=cv_detector_config.get('class_names') if cv_detector_config else None,
                    image_shape=seg_shape
                )
                # Override with custom configs if provided
                if cv_detector_config:
                    if 'detection_noise' in cv_detector_config:
                        self.cv_stack['detector'].detection_noise = cv_detector_config['detection_noise']
                    if 'dropout_rate' in cv_detector_config:
                        self.cv_stack['detector'].dropout_rate = cv_detector_config['dropout_rate']
                    if 'false_positive_rate' in cv_detector_config:
                        self.cv_stack['detector'].false_positive_rate = cv_detector_config['false_positive_rate']

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
        
        # Apply weather/lighting degradation to sensor outputs
        if self.weather_model is not None:
            # Degrade 3D detections (camera frame)
            dets = self.weather_model.degrade_detections_3d(dets)
            # Degrade segmentation
            seg = self.weather_model.degrade_segmentation(seg)
            # Degrade depth map
            depth = self.weather_model.degrade_depth_map(depth)
        
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

        # 4. CV-based detector path (optional): simulate learned perception
        cv_obstacles: List[Obstacle] = []
        if self.use_cv_detectors and self.cv_stack is not None:
            cv_obstacles = self._process_cv_detectors(
                gt_obstacles, ego_pose, t, seg_anomaly_ids)

        # 5. GT passthrough for obstacles the camera cannot see: not visible
        # to the camera at all (outside FOV/range, behind), not reconstructable
        # this tick, or an anomaly beyond the depth-extraction band. They keep
        # FrameId.MAP so the TransformTree passes them through untouched.
        seen_ids = {sp['id'] for sp in spatial}
        if self.use_cv_detectors:
            seen_ids.update(o.track_id for o in cv_obstacles)
        fallback = [o for o in gt_obstacles
                    if o.track_id not in seen_ids
                    and o.track_id not in seg_anomaly_ids]

        return PerceptionOutput(
            header=Header(t, FrameId.MAP, "vision_perception_node"),
            obstacles=vision_obstacles + anomaly_obstacles + cv_obstacles + fallback,
            latency_ms=1.0,
            status="OK",
        )

    def _process_cv_detectors(self, gt_obstacles: List[Obstacle],
                              ego_pose: Pose2D, t: float,
                              seg_anomaly_ids: set) -> List[Obstacle]:
        """Process obstacles through the synthetic CV detector stack.
        
        This simulates a learned perception pipeline:
        1. Generate synthetic camera image from GT
        2. Run 2D object detector (YOLO-style)
        3. Run semantic segmentation
        4. Run monocular depth estimation
        5. Run optical flow (for velocity)
        6. Fuse detections + depth -> 3D positions
        7. Convert to Obstacle records
        """
        cv_obstacles = []
        
        # Generate synthetic camera image (placeholder - would be rendered in real system)
        # For synthetic testing, we use the adapter's projection to generate "detections"
        # from GT, simulating what a learned detector would output
        
        camera_intrinsic = self.adapter.K
        image_shape = self.seg_shape
        
        # Get solid obstacles (non-anomalies) that are visible
        visible_solid = [o for o in gt_obstacles 
                         if o.track_id not in seg_anomaly_ids
                         and self.adapter._visible(
                             *self.adapter.map_to_camera(o.pose.x, o.pose.y, 0.0, ego_pose))]
        
        detections_2d = []
        
        if self.yolo_detector and self.yolo_detector.model_loaded:
            # Use YOLOv8 detector
            # Render synthetic image from GT
            synthetic_image = self._render_synthetic_image(visible_solid, ego_pose)
            
            yolo_detections = self.yolo_detector.detect(synthetic_image)
            
            for det in yolo_detections:
                detections_2d.append({
                    'x1': det['x1'], 'y1': det['y1'],
                    'x2': det['x2'], 'y2': det['y2'],
                    'class_id': det['class_id'],
                    'class_name': det['class_name'],
                    'confidence': det['confidence'],
                    'track_id': -1,  # YOLO doesn't provide track IDs
                })
        else:
            # Simulate 2D detector output from GT (original synthetic method)
            for obs in visible_solid:
                x_c, y_c, z_c = self.adapter.map_to_camera(obs.pose.x, obs.pose.y, 0.0, ego_pose)
                
                # Compute 2D bbox
                x1, y1, x2, y2 = compute_2d_bbox_from_3d(
                    x_c, y_c, z_c, obs.length, obs.width, camera_intrinsic)
                
                if x1 < 0 or y1 < 0:
                    continue
                
                # Map obstacle class to detector class
                det_class = OBSTACLE_CLASS_TO_DETECTOR_CLASS.get(obs.class_label, 'car')
                class_id = list(INDIAN_TRAFFIC_CLASSES.keys())[
                    list(INDIAN_TRAFFIC_CLASSES.values()).index(det_class)
                ] if det_class in INDIAN_TRAFFIC_CLASSES.values() else 1
                
                # Add detection with some noise to simulate detector imperfections
                noise_x = self._rng.gauss(0, 2.0) if hasattr(self, '_rng') else 0
                noise_y = self._rng.gauss(0, 2.0) if hasattr(self, '_rng') else 0
                
                detections_2d.append({
                    'x1': x1 + noise_x, 'y1': y1 + noise_y,
                    'x2': x2 + noise_x, 'y2': y2 + noise_y,
                    'class_id': class_id,
                    'class_name': det_class,
                    'confidence': 0.85 + self._rng.uniform(-0.1, 0.1),
                    'track_id': obs.track_id,
                    'z_c': z_c,  # Depth from GT (simulated depth estimation)
                })
        
        # Apply NMS
        # Add weather degradation to detections before NMS
        if self.weather_model is not None:
            # Add distance info to detections for weather degradation
            for det in detections_2d:
                if 'z_c' in det:
                    det['distance_m'] = det['z_c']
                elif 'x1' in det and 'x2' in det:
                    # Estimate distance from bbox size (rough approximation)
                    bbox_area = (det['x2'] - det['x1']) * (det['y2'] - det['y1'])
                    if bbox_area > 0:
                        det['distance_m'] = 10000.0 / math.sqrt(bbox_area)  # Rough inverse sqrt
            detections_2d = self.weather_model.degrade_detections(detections_2d)
        
        detections_2d = non_max_suppression([
            type('Detection2D', (), d)() for d in detections_2d
        ])
        
        # Convert 2D detections + depth to 3D obstacles
        for det in detections_2d:
            # Unproject using depth (z_c from GT for synthetic testing)
            z_c = getattr(det, 'z_c', 10.0)
            x_c = (det.x1 + det.x2) / 2 - camera_intrinsic[0, 2]
            y_c = (det.y1 + det.y2) / 2 - camera_intrinsic[1, 2]
            x_c = x_c * z_c / camera_intrinsic[0, 0]
            y_c = y_c * z_c / camera_intrinsic[1, 1]
            
            # Transform to base_link
            # Camera: X right, Y down, Z forward
            # Base_link: X forward, Y left, Z up
            x_v = z_c
            y_v = -x_c
            z_v = -y_c
            
            # Get class
            cls = _CLASS_BY_VALUE.get(det.class_name, ObstacleClass.UNKNOWN)
            
            cv_obstacles.append(Obstacle(
                header=Header(t, FrameId.VEHICLE, "cv_detector", seq=getattr(det, 'track_id', -1)),
                track_id=getattr(det, 'track_id', -1),
                class_label=cls,
                behavior=ObstacleBehavior.STATIC,
                pose=Pose2D(x=x_v, y=y_v, heading=0.0),
                length=2.0, width=1.5,
                velocity=Twist2D(),
                pose_covariance=Covariance2D(),
                velocity_covariance=Covariance2D(),
                confidence=det.confidence * 0.8,  # Lower confidence for learned detector
                is_dynamic=False,
            ))
        
        return cv_obstacles

    def _render_synthetic_image(self, obstacles: List[Obstacle], ego_pose: Pose2D) -> np.ndarray:
        """Render a synthetic camera image from ground truth obstacles.
        
        Used as input to YOLO detector when real camera images not available.
        """
        h, w = self.seg_shape
        image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Sky gradient
        for y in range(h // 2):
            intensity = int(200 * (1 - y / (h // 2)))
            image[y, :] = (intensity, intensity, min(255, intensity + 50))
        
        # Road
        road_color = (80, 80, 80)
        image[h // 2:, :] = road_color
        
        # Lane markings (dashed)
        for x in range(0, w, 40):
            x1, y1, x2, y2 = x, h // 2 + 10, x + 20, h // 2 + 15
            image[y1:y2, x1:x2] = (180, 180, 180)
        
        # Project obstacles
        for obs in obstacles:
            x_c, y_c, z_c = self.adapter.map_to_camera(
                obs.pose.x, obs.pose.y, 0.0, ego_pose)
            
            if not self.adapter._visible(x_c, y_c, z_c):
                continue
            
            u, v, _ = self.adapter._project(x_c, y_c, z_c)
            
            if 0 <= u < w and 0 <= v < h:
                cls_id = self._obstacle_class_to_id(obs.class_label)
                # Color mapping
                colors = {
                    1: (0, 0, 255),      # car - blue
                    2: (255, 165, 0),    # auto_rickshaw - orange
                    3: (255, 0, 255),    # two_wheeler - magenta
                    4: (0, 255, 255),    # bus - cyan
                    5: (128, 0, 128),    # truck - purple
                    6: (139, 69, 19),    # tractor - brown
                    7: (0, 255, 0),      # bicycle - green
                    8: (255, 255, 0),    # pedestrian - yellow
                    9: (160, 82, 45),    # cattle - sienna
                    10: (255, 192, 203), # dog - pink
                    11: (100, 100, 100), # pothole - gray
                    12: (255, 255, 255), # speed_breaker - white
                    13: (255, 0, 0),     # construction_barrier - red
                    14: (255, 255, 0),   # traffic_cone - yellow
                    15: (0, 0, 128),     # level_crossing_gate - navy
                    16: (255, 0, 0),     # emergency_vehicle - red
                }
                color = colors.get(cls_id, (255, 255, 255))
                
                # Draw box
                box_size = max(5, int(30 * 2.0 / max(z_c, 1.0)))
                u0, u1 = max(0, u - box_size), min(w, u + box_size)
                v0, v1 = max(0, v - box_size), min(h, v + box_size)
                image[v0:v1, u0:u1] = color
        
        return image

    def _render_synthetic_image(self, obstacles: List[Obstacle], ego_pose: Pose2D) -> np.ndarray:
        """Render a synthetic camera image from ground truth obstacles.
        
        Used as input to YOLO detector when real camera images not available.
        Applies weather degradation if weather model is enabled.
        """
        h, w = self.seg_shape
        image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Sky gradient
        for y in range(h // 2):
            intensity = int(200 * (1 - y / (h // 2)))
            image[y, :] = (intensity, intensity, min(255, intensity + 50))
        
        # Road
        road_color = (80, 80, 80)
        image[h // 2:, :] = road_color
        
        # Lane markings (dashed)
        for x in range(0, w, 40):
            x1, y1, x2, y2 = x, h // 2 + 10, x + 20, h // 2 + 15
            image[y1:y2, x1:x2] = (180, 180, 180)
        
        # Project obstacles
        for obs in obstacles:
            x_c, y_c, z_c = self.adapter.map_to_camera(
                obs.pose.x, obs.pose.y, 0.0, ego_pose)
            
            if not self.adapter._visible(x_c, y_c, z_c):
                continue
            
            u, v, _ = self.adapter._project(x_c, y_c, z_c)
            
            if 0 <= u < w and 0 <= v < h:
                cls_id = self._obstacle_class_to_id(obs.class_label)
                # Color mapping
                colors = {
                    1: (0, 0, 255),      # car - blue
                    2: (255, 165, 0),    # auto_rickshaw - orange
                    3: (255, 0, 255),    # two_wheeler - magenta
                    4: (0, 255, 255),    # bus - cyan
                    5: (128, 0, 128),    # truck - purple
                    6: (139, 69, 19),    # tractor - brown
                    7: (0, 255, 0),      # bicycle - green
                    8: (255, 255, 0),    # pedestrian - yellow
                    9: (160, 82, 45),    # cattle - sienna
                    10: (255, 192, 203), # dog - pink
                    11: (100, 100, 100), # pothole - gray
                    12: (255, 255, 255), # speed_breaker - white
                    13: (255, 0, 0),     # construction_barrier - red
                    14: (255, 255, 0),   # traffic_cone - yellow
                    15: (0, 0, 128),     # level_crossing_gate - navy
                    16: (255, 0, 0),     # emergency_vehicle - red
                }
                color = colors.get(cls_id, (255, 255, 255))
                
                # Draw box
                box_size = max(5, int(30 * 2.0 / max(z_c, 1.0)))
                u0, u1 = max(0, u - box_size), min(w, u + box_size)
                v0, v1 = max(0, v - box_size), min(h, v + box_size)
                image[v0:v1, u0:u1] = color
        
        # Apply weather degradation if model is available
        if self.weather_model is not None:
            image = self.weather_model.degrade_camera_image(image)
        
        return image

    def _obstacle_class_to_id(self, obs_class) -> int:
        """Map ObstacleClass to Indian traffic class ID."""
        mapping = {
            'vehicle': 1,
            'pedestrian': 8,
            'animal': 9,
            'pothole': 11,
            'encroachment': 13,
            'parked_vehicle': 1,
            'two_wheeler': 3,
        }
        return mapping.get(obs_class.value if hasattr(obs_class, 'value') else str(obs_class), 1)