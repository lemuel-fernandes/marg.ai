"""Synthetic Indian Traffic Dataset Generator.

Generates realistic synthetic data for training/validation of perception models:
- 2D bounding boxes with Indian traffic classes
- Semantic segmentation masks
- Depth maps
- Optical flow
- Acoustic events with DOA

All data is generated from ground-truth simulation scenarios, ensuring
perfect labels and enabling systematic dataset creation.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import json
import os
from pathlib import Path

# Local imports
from src.sim.scenario_registry import REGISTRY
from src.sim.executor import ScenarioExecutor
from src.perception.vision_node import VisionPerceptionNode
from src.perception.audio_pipeline import AcousticPerceptionNode
from src.common.types.sensor import SensorFrame
from src.common.types.base import FrameId, Header
from src.common.coordinates.transforms import TransformTree
from src.sim.python_sim import PythonSimulator
from src.controller.controllers.pure_pursuit import PurePursuitController
from src.common.types.trajectory import LocalTrajectory, TrajectoryPoint


# Indian traffic class definitions
INDIAN_TRAFFIC_CLASSES = {
    0: 'background',
    1: 'car',
    2: 'auto_rickshaw',
    3: 'two_wheeler',
    4: 'bus',
    5: 'truck',
    6: 'tractor',
    7: 'bicycle',
    8: 'pedestrian',
    9: 'cattle',
    10: 'dog',
    11: 'pothole',
    12: 'speed_breaker',
    13: 'construction_barrier',
    14: 'traffic_cone',
    15: 'level_crossing_gate',
    16: 'emergency_vehicle',
}

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 0, 255),       # car - blue
    2: (255, 165, 0),     # auto_rickshaw - orange
    3: (255, 0, 255),     # two_wheeler - magenta
    4: (0, 255, 255),     # bus - cyan
    5: (128, 0, 128),     # truck - purple
    6: (139, 69, 19),     # tractor - brown
    7: (0, 255, 0),       # bicycle - green
    8: (255, 255, 0),     # pedestrian - yellow
    9: (160, 82, 45),     # cattle - sienna
    10: (255, 192, 203),  # dog - pink
    11: (100, 100, 100),  # pothole - gray
    12: (255, 255, 255),  # speed_breaker - white
    13: (255, 0, 0),      # construction_barrier - red
    14: (255, 255, 0),    # traffic_cone - yellow
    15: (0, 0, 128),      # level_crossing_gate - navy
    16: (255, 255, 255),  # emergency_vehicle - white/red
}


@dataclass
class SyntheticSample:
    """Single synthetic training sample."""
    image_id: str
    timestamp: float
    image_shape: Tuple[int, int]
    
    # Camera parameters
    camera_intrinsic: np.ndarray
    camera_extrinsic: np.ndarray
    ego_pose: Dict[str, float]  # x, y, heading
    
    # 2D detections (YOLO format: class, x_center, y_center, width, height - normalized)
    detections_2d: List[Dict]
    
    # Segmentation mask (class ID per pixel)
    segmentation: Optional[np.ndarray] = None
    
    # Depth map (meters)
    depth: Optional[np.ndarray] = None
    
    # Optical flow (pixels)
    flow_x: Optional[np.ndarray] = None
    flow_y: Optional[np.ndarray] = None
    
    # Acoustic events
    acoustic_events: List[Dict] = field(default_factory=list)


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    output_dir: str = "data/synthetic_indian_traffic"
    image_shape: Tuple[int, int] = (480, 640)
    camera_fov_deg: float = 90.0
    max_range_m: float = 60.0
    
    # Scenarios to include
    scenarios: List[str] = field(default_factory=lambda: [
        'free_world', 'indian_road', 'city_roads', 'static_obstacle',
        'crossing_animal', 'multi_animal', 'sensor_noise', 'occluded_siren'
    ])
    
    # Samples per scenario
    samples_per_scenario: int = 100
    
    # Augmentation
    enable_noise: bool = True
    pos_noise_std: float = 0.1
    vel_noise_std: float = 0.05
    dropout_rate: float = 0.1
    false_positive_rate: float = 0.02
    
    # Acoustic
    include_acoustic: bool = True
    mic_array_geometry: str = "4mic_square"  # or "6mic_circular", "8mic_circular"


class SyntheticDatasetGenerator:
    """Generates synthetic perception dataset from simulation scenarios."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        (self.output_dir / "segmentation").mkdir(exist_ok=True)
        (self.output_dir / "depth").mkdir(exist_ok=True)
        (self.output_dir / "flow").mkdir(exist_ok=True)
        (self.output_dir / "acoustic").mkdir(exist_ok=True)
        
        # Camera intrinsic
        h, w = config.image_shape
        fov_rad = math.radians(config.camera_fov_deg)
        fx = w / (2 * math.tan(fov_rad / 2))
        fy = fx
        cx, cy = w / 2, h / 2
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        
        # Camera extrinsic (base_link to camera)
        # Camera: X right, Y down, Z forward
        # Base_link: X forward, Y left, Z up
        self.R_c2v = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
        self.T_c2v = np.zeros(3, dtype=np.float32)
        
        self._rng = np.random.default_rng(42)
        self._sample_count = 0
    
    def generate_dataset(self) -> Dict[str, int]:
        """Generate complete dataset across all configured scenarios."""
        stats = {scenario: 0 for scenario in self.config.scenarios}
        
        for scenario_name in self.config.scenarios:
            print(f"Generating {self.config.samples_per_scenario} samples for {scenario_name}...")
            count = self._generate_scenario_samples(scenario_name)
            stats[scenario_name] = count
        
        # Write dataset manifest
        self._write_manifest(stats)
        
        print(f"Dataset generated: {sum(stats.values())} total samples")
        return stats
    
    def _generate_scenario_samples(self, scenario_name: str) -> int:
        """Generate samples for a specific scenario."""
        scenario_class = REGISTRY[scenario_name]
        scenario = scenario_class()
        
        v_cfg, c_cfg, dwa_cfg = scenario.configs()
        
        count = 0
        for seed in range(self.config.samples_per_scenario):
            # Create scenario with different seed
            if hasattr(scenario, 'seed'):
                scenario.seed = seed
                scenario._rng = random.Random(seed)
                scenario._events = scenario._generate_events()
                if hasattr(scenario, '_generate_acoustic_events'):
                    scenario._acoustic_events = scenario._generate_acoustic_events()
            
            # Setup perception with noise
            tf = TransformTree()
            
            def obstacle_provider():
                return scenario.obstacles_at(tf.ego_state.pose.x if tf.ego_state else 0)
            
            vision_node = VisionPerceptionNode(
                obstacles_provider=obstacle_provider,
                camera_intrinsic=self.K,
                camera_extrinsics_rt=np.column_stack([self.R_c2v, self.T_c2v]),
                transform_tree=tf,
                fov_rad=math.radians(self.config.camera_fov_deg),
                max_range_m=self.config.max_range_m,
                seg_shape=self.config.image_shape,
                pos_noise_std=self.config.pos_noise_std if self.config.enable_noise else 0.0,
                vel_noise_std=self.config.vel_noise_std if self.config.enable_noise else 0.0,
            )
            
            acoustic_node = None
            if self.config.include_acoustic and hasattr(scenario, 'acoustic_events'):
                acoustic_node = AcousticPerceptionNode(
                    events=scenario.acoustic_events(),
                    detect_snr_db=6.0,
                )
            
            # Run simulation and collect samples at intervals
            sim_duration = scenario.duration_s
            dt = 0.1
            sample_interval = 1.0  # Sample every 1 second
            
            state = scenario.initial_state()
            goal = scenario.goal()
            
            # Initialize simulator
            sim = PythonSimulator(v_cfg)
            
            clock = 0.0
            last_sample_t = -1.0
            
            while clock < sim_duration:
                # Update transform tree
                tf.update_ego_state(state)
                
                # Sample at intervals
                if clock - last_sample_t >= sample_interval:
                    sample = self._create_sample(
                        scenario_name=scenario_name,
                        clock=clock,
                        state=state,
                        vision_node=vision_node,
                        acoustic_node=acoustic_node,
                        tf=tf,
                    )
                    self._save_sample(sample)
                    count += 1
                    self._sample_count += 1
                    last_sample_t = clock
                
                # Step simulation
                sensor = SensorFrame(header=Header(clock, FrameId.SENSOR_FRONT, "sim"))
                perception_out = vision_node.process(sensor)
                map_obs = tf.obstacles_to_map(perception_out.obstacles)
                
                # Simple control for data collection (go straight)
                controller = PurePursuitController(v_cfg)
                
                # Dummy trajectory for control
                traj = LocalTrajectory(
                    header=Header(clock, FrameId.MAP, "dummy"),
                    points=[TrajectoryPoint(
                        t=clock, pose=state.pose, twist=state.twist,
                        curvature=0.0, acceleration=0.0
                    )],
                    cost=0.0, is_safe=True, fallback_active=False, reason="dummy"
                )
                cmd = controller.command(state, traj)
                
                state = sim.step(state, cmd, dt)
                clock += dt
        
        return count
    
    def _create_sample(self, scenario_name: str, clock: float, state,
                       vision_node, acoustic_node, tf) -> SyntheticSample:
        """Create a single synthetic sample from current simulation state."""
        # Get 2D detections from vision node's adapter
        detections_2d = self._extract_2d_detections(vision_node, state, tf)
        
        # Get segmentation
        gt_obstacles = vision_node.provider()
        ego_pose = state.pose
        seg, depth = vision_node.adapter.make_seg_and_depth(
            gt_obstacles, ego_pose, width=self.config.image_shape[1],
            height=self.config.image_shape[0])
        
        # Get acoustic events
        acoustic_events = []
        if acoustic_node is not None:
            sensor = SensorFrame(header=Header(clock, FrameId.SENSOR_FRONT, "sim"))
            cues = acoustic_node.process(sensor)
            for cue in cues:
                acoustic_events.append({
                    't': cue.t,
                    'acoustic_class': cue.acoustic_class,
                    'azimuth_rad': cue.azimuth_rad,
                    'confidence': cue.confidence,
                    'is_persisted': cue.is_persisted,
                })
        
        # Create sample
        sample = SyntheticSample(
            image_id=f"{scenario_name}_{self._sample_count:06d}",
            timestamp=clock,
            image_shape=self.config.image_shape,
            camera_intrinsic=self.K.copy(),
            camera_extrinsic=np.column_stack([self.R_c2v, self.T_c2v]),
            ego_pose={'x': state.pose.x, 'y': state.pose.y, 'heading': state.pose.heading},
            detections_2d=detections_2d,
            segmentation=seg,
            depth=depth,
            acoustic_events=acoustic_events,
        )
        
        return sample
    
    def _extract_2d_detections(self, vision_node, state, tf) -> List[Dict]:
        """Extract 2D detections in YOLO format."""
        detections = []
        gt_obstacles = vision_node.provider()
        ego_pose = state.pose
        
        for obs in gt_obstacles:
            if not vision_node.adapter._visible(
                *vision_node.adapter.map_to_camera(obs.pose.x, obs.pose.y, 0.0, ego_pose)):
                continue
            
            x_c, y_c, z_c = vision_node.adapter.map_to_camera(
                obs.pose.x, obs.pose.y, 0.0, ego_pose)
            
            x1, y1, x2, y2 = self._compute_2d_bbox(x_c, y_c, z_c, obs.length, obs.width)
            
            if x1 < 0 or y1 < 0:
                continue
            
            h, w = self.config.image_shape
            # YOLO format: class_id, x_center, y_center, width, height (normalized)
            x_center = (x1 + x2) / 2 / w
            y_center = (y1 + y2) / 2 / h
            box_w = (x2 - x1) / w
            box_h = (y2 - y1) / h
            
            cls_id = self._obstacle_class_to_id(obs.class_label)
            
            detections.append({
                'class_id': cls_id,
                'class_name': INDIAN_TRAFFIC_CLASSES.get(cls_id, 'unknown'),
                'x_center': float(x_center),
                'y_center': float(y_center),
                'width': float(box_w),
                'height': float(box_h),
                'confidence': 0.9,
                'track_id': obs.track_id,
                'distance_m': float(z_c),
            })
        
        return detections
    
    def _compute_2d_bbox(self, x_c: float, y_c: float, z_c: float,
                         length: float, width: float) -> Tuple[int, int, int, int]:
        """Compute 2D bounding box from 3D camera coordinates."""
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        if z_c <= 0:
            return -1, -1, -1, -1
        
        # Project center
        u = int(round(cx + fx * x_c / z_c))
        v = int(round(cy + fy * y_c / z_c))
        
        # Approximate box size in pixels
        w_pix = int(round(fx * width / z_c / 2))
        h_pix = int(round(fy * length / z_c / 2))
        
        x1 = max(0, u - w_pix)
        y1 = max(0, v - h_pix)
        x2 = min(self.config.image_shape[1], u + w_pix)
        y2 = min(self.config.image_shape[0], v + h_pix)
        
        return x1, y1, x2, y2
    
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
    
    def _save_sample(self, sample: SyntheticSample) -> None:
        """Save sample to disk."""
        base_name = sample.image_id
        
        # Save annotations (YOLO format)
        ann_path = self.output_dir / "annotations" / f"{base_name}.txt"
        with open(ann_path, 'w') as f:
            for det in sample.detections_2d:
                f.write(f"{det['class_id']} {det['x_center']:.6f} {det['y_center']:.6f} "
                        f"{det['width']:.6f} {det['height']:.6f}\n")
        
        # Save segmentation
        if sample.segmentation is not None:
            seg_path = self.output_dir / "segmentation" / f"{base_name}.npy"
            np.save(seg_path, sample.segmentation.astype(np.uint8))
        
        # Save depth
        if sample.depth is not None:
            depth_path = self.output_dir / "depth" / f"{base_name}.npy"
            np.save(depth_path, sample.depth.astype(np.float32))
        
        # Save acoustic events
        if sample.acoustic_events:
            ac_path = self.output_dir / "acoustic" / f"{base_name}.json"
            with open(ac_path, 'w') as f:
                json.dump(sample.acoustic_events, f, indent=2)
        
        # Save metadata
        meta_path = self.output_dir / "annotations" / f"{base_name}.json"
        meta = {
            'image_id': sample.image_id,
            'timestamp': sample.timestamp,
            'image_shape': sample.image_shape,
            'camera_intrinsic': sample.camera_intrinsic.tolist(),
            'camera_extrinsic': sample.camera_extrinsic.tolist(),
            'ego_pose': sample.ego_pose,
            'detections': sample.detections_2d,
            'acoustic_events': sample.acoustic_events,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
    
    def _write_manifest(self, stats: Dict[str, int]) -> None:
        """Write dataset manifest."""
        manifest = {
            'dataset_name': 'synthetic_indian_traffic',
            'version': '1.0',
            'config': {
                'image_shape': self.config.image_shape,
                'camera_fov_deg': self.config.camera_fov_deg,
                'max_range_m': self.config.max_range_m,
                'classes': INDIAN_TRAFFIC_CLASSES,
                'scenarios': stats,
            },
            'total_samples': sum(stats.values()),
            'splits': {
                'train': 0.7,
                'val': 0.15,
                'test': 0.15,
            },
        }
        
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create train/val/test split files
        self._create_split_files()
    
    def _create_split_files(self) -> None:
        """Create train/val/test split files."""
        ann_dir = self.output_dir / "annotations"
        all_files = sorted([f.stem for f in ann_dir.glob("*.txt")])
        
        n = len(all_files)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        
        splits = {
            'train': all_files[:n_train],
            'val': all_files[n_train:n_train + n_val],
            'test': all_files[n_train + n_val:],
        }
        
        for split_name, files in splits.items():
            split_path = self.output_dir / f"{split_name}.txt"
            with open(split_path, 'w') as f:
                for file in files:
                    f.write(f"{file}\n")


def create_coco_format_annotations(generator: SyntheticDatasetGenerator,
                                    output_path: str) -> None:
    """Convert generated dataset to COCO format for detection training."""
    from pathlib import Path
    
    ann_dir = Path(generator.output_dir) / "annotations"
    all_files = sorted(ann_dir.glob("*.json"))
    
    coco = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': k, 'name': v, 'supercategory': 'traffic'}
            for k, v in INDIAN_TRAFFIC_CLASSES.items() if k > 0
        ],
    }
    
    ann_id = 1
    for img_id, meta_file in enumerate(all_files):
        with open(meta_file) as f:
            meta = json.load(f)
        
        # Image entry
        coco['images'].append({
            'id': img_id,
            'file_name': f"{meta['image_id']}.jpg",
            'width': meta['image_shape'][1],
            'height': meta['image_shape'][0],
        })
        
        # Annotations
        for det in meta['detections']:
            w = det['width'] * meta['image_shape'][1]
            h = det['height'] * meta['image_shape'][0]
            x = (det['x_center'] - det['width'] / 2) * meta['image_shape'][1]
            y = (det['y_center'] - det['height'] / 2) * meta['image_shape'][0]
            
            coco['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': det['class_id'],
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
            })
            ann_id += 1
    
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)
    
    print(f"COCO annotations saved to {output_path}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")


if __name__ == "__main__":
    # Quick test generation
    config = DatasetConfig(
        output_dir="data/test_synthetic",
        samples_per_scenario=2,
        scenarios=['static_obstacle', 'free_world'],
    )
    
    generator = SyntheticDatasetGenerator(config)
    stats = generator.generate_dataset()
    print("Test generation complete:", stats)
    
    # Generate COCO format
    create_coco_format_annotations(generator, "data/test_synthetic/annotations_coco.json")