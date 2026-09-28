"""CV-based vision detectors for autonomous driving perception.

This module provides interfaces and implementations for learned vision models:
- 2D Object Detection (YOLO-style)
- Semantic Segmentation (DeepLabV3+ style)
- Monocular Depth Estimation (MiDaS/DepthAnything style)
- Optical Flow (RAFT style)

All implementations use synthetic data generation for training/validation
and provide a common interface for the vision pipeline.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class Detection2D:
    """2D bounding box detection with class and confidence."""
    x1: float      # Left pixel
    y1: float      # Top pixel
    x2: float      # Right pixel
    y2: float      # Bottom pixel
    class_id: int  # Class index
    class_name: str
    confidence: float
    track_id: Optional[int] = None


@dataclass
class SegmentationMask:
    """Semantic segmentation output."""
    mask: np.ndarray          # H x W, class IDs per pixel
    class_names: List[str]    # Index -> class name mapping
    confidence: np.ndarray    # H x W, per-pixel confidence


@dataclass
class DepthMap:
    """Monocular depth estimation output."""
    depth: np.ndarray         # H x W, depth in meters
    confidence: np.ndarray    # H x W, per-pixel confidence
    valid_mask: np.ndarray    # H x W, boolean validity


@dataclass
class OpticalFlow:
    """Optical flow output."""
    flow_x: np.ndarray        # H x W, horizontal flow (pixels)
    flow_y: np.ndarray        # H x W, vertical flow (pixels)
    confidence: np.ndarray    # H x W, per-pixel confidence


# Indian traffic class mapping
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

# COCO-style class mapping for pre-trained models
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic_light',
    10: 'fire_hydrant', 11: 'stop_sign', 12: 'parking_meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports_ball', 33: 'kite', 34: 'baseball_bat',
    35: 'baseball_glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis_racket',
    39: 'bottle', 40: 'wine_glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot_dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted_plant',
    59: 'bed', 60: 'dining_table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell_phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy_bear', 78: 'hair_drier',
    79: 'toothbrush',
}

# Map COCO classes to Indian traffic classes
COCO_TO_INDIAN = {
    0: 8,   # person -> pedestrian
    1: 7,   # bicycle -> bicycle
    2: 1,   # car -> car
    3: 3,   # motorcycle -> two_wheeler
    5: 4,   # bus -> bus
    7: 5,   # truck -> truck
    16: 9,  # dog -> dog
    17: 9,  # horse -> cattle (approximate)
    18: 9,  # sheep -> cattle
    19: 9,  # cow -> cattle
}


class BaseDetector(ABC):
    """Abstract base class for vision detectors."""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection2D]:
        """Run detection on image. Returns list of Detection2D."""
        pass
    
    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Return list of class names."""
        pass


class BaseSegmenter(ABC):
    """Abstract base class for semantic segmentation."""
    
    @abstractmethod
    def segment(self, image: np.ndarray) -> SegmentationMask:
        """Run segmentation on image."""
        pass


class BaseDepthEstimator(ABC):
    """Abstract base class for monocular depth estimation."""
    
    @abstractmethod
    def estimate_depth(self, image: np.ndarray) -> DepthMap:
        """Estimate depth from single image."""
        pass


class BaseOpticalFlow(ABC):
    """Abstract base class for optical flow estimation."""
    
    @abstractmethod
    def compute_flow(self, prev_image: np.ndarray, curr_image: np.ndarray) -> OpticalFlow:
        """Compute optical flow between two frames."""
        pass


class SyntheticDetector(BaseDetector):
    """Synthetic detector that generates detections from ground truth.
    
    Used for testing and validation without real model weights.
    Simulates YOLO-style output with configurable noise and dropouts.
    """
    
    def __init__(self, 
                 class_names: Optional[List[str]] = None,
                 detection_noise: float = 0.0,
                 dropout_rate: float = 0.0,
                 false_positive_rate: float = 0.0):
        self.class_names = class_names or [INDIAN_TRAFFIC_CLASSES[i] for i in range(len(INDIAN_TRAFFIC_CLASSES))]
        self.detection_noise = detection_noise
        self.dropout_rate = dropout_rate
        self.false_positive_rate = false_positive_rate
        self._rng = np.random.default_rng(42)
    
    def detect(self, image: np.ndarray) -> List[Detection2D]:
        """Generate synthetic detections. In real use, this would run a CNN."""
        # This is a placeholder - in practice this would be replaced by
        # a real model inference. For synthetic testing, we return empty
        # and rely on the SyntheticCameraAdapter for ground truth.
        return []
    
    def detect_from_gt(self, gt_obstacles: List, ego_pose, 
                       camera_intrinsic: np.ndarray,
                       image_shape: Tuple[int, int] = (480, 640)) -> List[Detection2D]:
        """Generate detections from ground truth obstacles (for synthetic testing)."""
        detections = []
        h, w = image_shape
        fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
        cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
        
        for obs in gt_obstacles:
            # Project to camera frame (simplified)
            # In reality, SyntheticCameraAdapter does this properly
            pass
        
        return detections
    
    def get_class_names(self) -> List[str]:
        return self.class_names


class SyntheticSegmenter(BaseSegmenter):
    """Synthetic semantic segmenter for road scene understanding."""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or [INDIAN_TRAFFIC_CLASSES[i] for i in range(len(INDIAN_TRAFFIC_CLASSES))]
    
    def segment(self, image: np.ndarray) -> SegmentationMask:
        """Generate synthetic segmentation. Placeholder for real model."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.int32)
        confidence = np.ones((h, w), dtype=np.float32)
        return SegmentationMask(mask=mask, class_names=self.class_names, confidence=confidence)
    
    def segment_from_gt(self, gt_anomalies: List, ego_pose,
                        camera_intrinsic: np.ndarray,
                        image_shape: Tuple[int, int] = (480, 640)) -> SegmentationMask:
        """Generate segmentation from ground truth anomalies."""
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.int32)
        confidence = np.zeros((h, w), dtype=np.float32)
        
        # Class IDs for anomalies
        class_id_map = {'pothole': 11, 'speed_breaker': 12}
        
        for obs in gt_anomalies:
            # Project to image (simplified)
            cls_id = class_id_map.get(obs.class_label.value, 0)
            if cls_id > 0:
                # Create a blob at projected location
                # This is simplified - real implementation uses SyntheticCameraAdapter
                pass
        
        return SegmentationMask(mask=mask, class_names=self.class_names, confidence=confidence)


class SyntheticDepthEstimator(BaseDepthEstimator):
    """Synthetic monocular depth estimator."""
    
    def estimate_depth(self, image: np.ndarray) -> DepthMap:
        """Generate synthetic depth. Placeholder for real model."""
        h, w = image.shape[:2]
        depth = np.full((h, w), 25.0, dtype=np.float32)
        confidence = np.zeros((h, w), dtype=np.float32)
        valid_mask = np.zeros((h, w), dtype=bool)
        return DepthMap(depth=depth, confidence=confidence, valid_mask=valid_mask)
    
    def estimate_from_gt(self, gt_obstacles: List, ego_pose,
                         camera_intrinsic: np.ndarray,
                         image_shape: Tuple[int, int] = (480, 640)) -> DepthMap:
        """Generate depth map from ground truth."""
        h, w = image_shape
        depth = np.full((h, w), 25.0, dtype=np.float32)
        confidence = np.zeros((h, w), dtype=np.float32)
        valid_mask = np.zeros((h, w), dtype=bool)
        
        # This would be filled by SyntheticCameraAdapter.make_seg_and_depth
        return DepthMap(depth=depth, confidence=confidence, valid_mask=valid_mask)


class SyntheticOpticalFlow(BaseOpticalFlow):
    """Synthetic optical flow estimator."""
    
    def compute_flow(self, prev_image: np.ndarray, curr_image: np.ndarray) -> OpticalFlow:
        """Generate synthetic flow. Placeholder for real model."""
        h, w = prev_image.shape[:2]
        return OpticalFlow(
            flow_x=np.zeros((h, w), dtype=np.float32),
            flow_y=np.zeros((h, w), dtype=np.float32),
            confidence=np.zeros((h, w), dtype=np.float32)
        )
    
    def compute_from_gt(self, gt_obstacles_prev: List, gt_obstacles_curr: List,
                        ego_pose_prev, ego_pose_curr,
                        camera_intrinsic: np.ndarray,
                        image_shape: Tuple[int, int] = (480, 640)) -> OpticalFlow:
        """Compute flow from ground truth obstacle tracks."""
        h, w = image_shape
        flow_x = np.zeros((h, w), dtype=np.float32)
        flow_y = np.zeros((h, w), dtype=np.float32)
        confidence = np.zeros((h, w), dtype=np.float32)
        
        # Match obstacles by track_id and compute pixel displacement
        # This would require proper camera projection
        return OpticalFlow(flow_x=flow_x, flow_y=flow_y, confidence=confidence)


def create_synthetic_vision_stack(
    class_names: Optional[List[str]] = None,
    image_shape: Tuple[int, int] = (480, 640)
) -> Dict[str, Any]:
    """Create a complete synthetic vision stack for testing."""
    return {
        'detector': SyntheticDetector(class_names=class_names),
        'segmenter': SyntheticSegmenter(class_names=class_names),
        'depth_estimator': SyntheticDepthEstimator(),
        'optical_flow': SyntheticOpticalFlow(),
        'class_names': class_names or [INDIAN_TRAFFIC_CLASSES[i] for i in range(len(INDIAN_TRAFFIC_CLASSES))],
        'image_shape': image_shape,
    }


def project_3d_to_2d(x_c: float, y_c: float, z_c: float, 
                     intrinsic: np.ndarray) -> Tuple[int, int]:
    """Project 3D camera point to 2D pixel coordinates."""
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    
    if z_c <= 0:
        return -1, -1
    
    u = int(round(cx + fx * x_c / z_c))
    v = int(round(cy + fy * y_c / z_c))
    return u, v


def compute_2d_bbox_from_3d(x_c: float, y_c: float, z_c: float,
                            length: float, width: float,
                            intrinsic: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute 2D bounding box from 3D box center and dimensions."""
    # Project center
    u, v = project_3d_to_2d(x_c, y_c, z_c, intrinsic)
    
    if u < 0 or v < 0:
        return -1, -1, -1, -1
    
    # Approximate box size in pixels
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    
    # Half-width/height in pixels at this depth
    w_pix = int(round(fx * width / z_c / 2))
    h_pix = int(round(fy * length / z_c / 2))
    
    x1 = max(0, u - w_pix)
    y1 = max(0, v - h_pix)
    x2 = u + w_pix
    y2 = v + h_pix
    
    return x1, y1, x2, y2


def non_max_suppression(detections: List[Detection2D], 
                        iou_threshold: float = 0.5) -> List[Detection2D]:
    """Non-maximum suppression for overlapping detections."""
    if not detections:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep = []
    
    for det in detections:
        keep_det = True
        for kept in keep:
            if compute_iou(det, kept) > iou_threshold:
                keep_det = False
                break
        if keep_det:
            keep.append(det)
    
    return keep


def compute_iou(det1: Detection2D, det2: Detection2D) -> float:
    """Compute IoU between two 2D detections."""
    x1 = max(det1.x1, det2.x1)
    y1 = max(det1.y1, det2.y1)
    x2 = min(det1.x2, det2.x2)
    y2 = min(det1.y2, det2.y2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    area1 = (det1.x2 - det1.x1) * (det1.y2 - det1.y1)
    area2 = (det2.x2 - det2.x1) * (det2.y2 - det2.y1)
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


# Indian-specific class weights for handling class imbalance
INDIAN_CLASS_WEIGHTS = {
    'background': 0.1,
    'car': 1.0,
    'auto_rickshaw': 2.0,       # Important, underrepresented
    'two_wheeler': 2.0,         # Very common in India
    'bus': 1.5,
    'truck': 1.5,
    'tractor': 1.5,             # Rural roads
    'bicycle': 1.5,
    'pedestrian': 2.0,          # Safety critical
    'cattle': 3.0,              # Unique to India, safety critical
    'dog': 1.5,
    'pothole': 3.0,             # Road condition critical
    'speed_breaker': 2.0,
    'construction_barrier': 1.5,
    'traffic_cone': 1.0,
    'level_crossing_gate': 3.0, # Safety critical
    'emergency_vehicle': 3.0,   # Safety critical
}


def get_class_weight(class_name: str) -> float:
    """Get loss weight for a class."""
    return INDIAN_CLASS_WEIGHTS.get(class_name, 1.0)