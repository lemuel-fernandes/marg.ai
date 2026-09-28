"""Perception metrics for autonomous driving evaluation.

Computes standard perception metrics:
- Detection: mAP, AP per class, precision/recall
- Segmentation: mIoU, per-class IoU
- Depth: RMSE, AbsRel, SqRel, delta thresholds
- Tracking: MOTA, MOTP, ID switches
- Acoustic: DOA accuracy, classification accuracy
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from collections import defaultdict


@dataclass
class DetectionMetrics:
    """2D object detection metrics."""
    map_50: float = 0.0          # mAP@0.5 IoU
    map_50_95: float = 0.0       # mAP@[0.5:0.95]
    ap_per_class: Dict[str, float] = field(default_factory=dict)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    
    # Per-class detail
    class_tp: Dict[str, int] = field(default_factory=dict)
    class_fp: Dict[str, int] = field(default_factory=dict)
    class_fn: Dict[str, int] = field(default_factory=dict)
    class_gt_count: Dict[str, int] = field(default_factory=dict)


@dataclass
class SegmentationMetrics:
    """Semantic segmentation metrics."""
    miou: float = 0.0
    iou_per_class: Dict[str, float] = field(default_factory=dict)
    pixel_accuracy: float = 0.0
    mean_accuracy: float = 0.0


@dataclass
class DepthMetrics:
    """Monocular depth estimation metrics."""
    rmse: float = 0.0
    abs_rel: float = 0.0
    sq_rel: float = 0.0
    log_rmse: float = 0.0
    delta_1: float = 0.0   # % pixels with max(pred/gt, gt/pred) < 1.25
    delta_2: float = 0.0   # < 1.25^2
    delta_3: float = 0.0   # < 1.25^3


@dataclass
class TrackingMetrics:
    """Multi-object tracking metrics."""
    mota: float = 0.0
    motp: float = 0.0
    id_switches: int = 0
    fragmentations: int = 0
    mostly_tracked: float = 0.0
    mostly_lost: float = 0.0


@dataclass
class AcousticMetrics:
    """Acoustic perception metrics."""
    doa_mae_deg: float = 0.0      # Mean absolute error in degrees
    doa_rmse_deg: float = 0.0
    classification_accuracy: float = 0.0
    detection_precision: float = 0.0
    detection_recall: float = 0.0


@dataclass
class PerceptionMetrics:
    """Complete perception metrics bundle."""
    detection: DetectionMetrics = field(default_factory=DetectionMetrics)
    segmentation: SegmentationMetrics = field(default_factory=SegmentationMetrics)
    depth: DepthMetrics = field(default_factory=DepthMetrics)
    tracking: TrackingMetrics = field(default_factory=TrackingMetrics)
    acoustic: AcousticMetrics = field(default_factory=AcousticMetrics)
    
    # Overall summary
    overall_score: float = 0.0


def compute_iou_2d(box1: Tuple[float, float, float, float],
                   box2: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


def compute_detection_metrics(gt_detections: List[Dict],
                              pred_detections: List[Dict],
                              iou_threshold: float = 0.5,
                              class_names: Optional[Dict[int, str]] = None) -> DetectionMetrics:
    """
    Compute detection metrics (mAP, precision, recall).
    
    Args:
        gt_detections: List of ground truth detections with keys:
            class_id, x1, y1, x2, y2 (or x_center, y_center, width, height)
        pred_detections: List of predicted detections with keys:
            class_id, x1, y1, x2, y2, confidence (or normalized format)
        iou_threshold: IoU threshold for positive match
        class_names: Mapping from class_id to class name
        
    Returns:
        DetectionMetrics object
    """
    if class_names is None:
        class_names = {}
    
    # Convert to (x1, y1, x2, y2) if needed
    def to_xyxy(det):
        if 'x_center' in det:
            w = det['width'] if 'width' in det else det['w']
            h = det['height'] if 'height' in det else det['h']
            x1 = det['x_center'] - w / 2
            y1 = det['y_center'] - h / 2
            x2 = det['x_center'] + w / 2
            y2 = det['y_center'] + h / 2
            return (x1, y1, x2, y2)
        return (det['x1'], det['y1'], det['x2'], det['y2'])
    
    gt_by_class = defaultdict(list)
    pred_by_class = defaultdict(list)
    
    for det in gt_detections:
        cid = det['class_id']
        gt_by_class[cid].append({'box': to_xyxy(det), 'matched': False})
    
    for det in pred_detections:
        cid = det['class_id']
        conf = det.get('confidence', det.get('conf', 1.0))
        pred_by_class[cid].append({'box': to_xyxy(det), 'conf': conf, 'matched': False})
    
    # Sort predictions by confidence
    for cid in pred_by_class:
        pred_by_class[cid].sort(key=lambda x: x['conf'], reverse=True)
    
    metrics = DetectionMetrics()
    all_tp, all_fp, all_fn = 0, 0, 0
    
    for cid in set(list(gt_by_class.keys()) + list(pred_by_class.keys())):
        gts = gt_by_class[cid]
        preds = pred_by_class[cid]
        
        class_name = class_names.get(cid, f'class_{cid}')
        metrics.class_gt_count[class_name] = len(gts)
        
        tp, fp = 0, 0
        
        for pred in preds:
            best_iou = 0.0
            best_gt_idx = -1
            
            for i, gt in enumerate(gts):
                if gt['matched']:
                    continue
                iou = compute_iou_2d(pred['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gts[best_gt_idx]['matched'] = True
                tp += 1
            else:
                fp += 1
        
        fn = sum(1 for gt in gts if not gt['matched'])
        
        metrics.class_tp[class_name] = tp
        metrics.class_fp[class_name] = fp
        metrics.class_fn[class_name] = fn
        
        all_tp += tp
        all_fp += fp
        all_fn += fn
        
        # Per-class AP (simplified: TP / (TP + FP) at this threshold)
        if tp + fp > 0:
            metrics.ap_per_class[class_name] = tp / (tp + fp)
        else:
            metrics.ap_per_class[class_name] = 0.0
    
    # Overall metrics
    if all_tp + all_fp > 0:
        metrics.precision = all_tp / (all_tp + all_fp)
    if all_tp + all_fn > 0:
        metrics.recall = all_tp / (all_tp + all_fn)
    if metrics.precision + metrics.recall > 0:
        metrics.f1 = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
    
    # mAP@0.5
    if metrics.ap_per_class:
        metrics.map_50 = sum(metrics.ap_per_class.values()) / len(metrics.ap_per_class)
    
    return metrics


def compute_segmentation_metrics(gt_seg: np.ndarray,
                                  pred_seg: np.ndarray,
                                  num_classes: int,
                                  class_names: Optional[Dict[int, str]] = None) -> SegmentationMetrics:
    """Compute segmentation metrics (mIoU, pixel accuracy)."""
    if class_names is None:
        class_names = {i: f'class_{i}' for i in range(num_classes)}
    
    metrics = SegmentationMetrics()
    
    # Flatten
    gt_flat = gt_seg.flatten()
    pred_flat = pred_seg.flatten()
    
    # Valid mask (ignore background class 0 if desired)
    valid = (gt_flat >= 0) & (gt_flat < num_classes) & (pred_flat >= 0) & (pred_flat < num_classes)
    gt_valid = gt_flat[valid]
    pred_valid = pred_flat[valid]
    
    # Pixel accuracy
    metrics.pixel_accuracy = np.mean(gt_valid == pred_valid)
    
    # Per-class IoU
    ious = []
    for c in range(num_classes):
        gt_c = (gt_valid == c)
        pred_c = (pred_valid == c)
        
        inter = np.sum(gt_c & pred_c)
        union = np.sum(gt_c | pred_c)
        
        if union > 0:
            iou = inter / union
        else:
            iou = 1.0 if not np.any(gt_c) else 0.0
        
        class_name = class_names.get(c, f'class_{c}')
        metrics.iou_per_class[class_name] = float(iou)
        ious.append(iou)
    
    metrics.miou = float(np.mean(ious))
    
    # Mean accuracy (per-class accuracy averaged)
    class_accs = []
    for c in range(num_classes):
        gt_c = (gt_valid == c)
        if np.sum(gt_c) > 0:
            pred_c = (pred_valid == c)
            acc = np.sum(gt_c & pred_c) / np.sum(gt_c)
            class_accs.append(acc)
    metrics.mean_accuracy = float(np.mean(class_accs)) if class_accs else 0.0
    
    return metrics


def compute_depth_metrics(gt_depth: np.ndarray,
                          pred_depth: np.ndarray,
                          valid_mask: Optional[np.ndarray] = None) -> DepthMetrics:
    """Compute depth estimation metrics."""
    metrics = DepthMetrics()
    
    # Flatten
    gt = gt_depth.flatten()
    pred = pred_depth.flatten()
    
    if valid_mask is not None:
        valid = valid_mask.flatten()
        gt = gt[valid]
        pred = pred[valid]
    else:
        # Valid where gt > 0
        valid = gt > 0
        gt = gt[valid]
        pred = pred[valid]
    
    if len(gt) == 0:
        return metrics
    
    # Clip predictions to valid range
    pred = np.clip(pred, 0.1, 100.0)
    
    # RMSE
    metrics.rmse = float(np.sqrt(np.mean((gt - pred) ** 2)))
    
    # AbsRel
    metrics.abs_rel = float(np.mean(np.abs(gt - pred) / gt))
    
    # SqRel
    metrics.sq_rel = float(np.mean((gt - pred) ** 2 / gt))
    
    # Log RMSE
    metrics.log_rmse = float(np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2)))
    
    # Delta thresholds
    ratio = np.maximum(gt / pred, pred / gt)
    metrics.delta_1 = float(np.mean(ratio < 1.25))
    metrics.delta_2 = float(np.mean(ratio < 1.25 ** 2))
    metrics.delta_3 = float(np.mean(ratio < 1.25 ** 3))
    
    return metrics


def compute_tracking_metrics(gt_tracks: List[Dict],
                             pred_tracks: List[Dict],
                             iou_threshold: float = 0.5) -> TrackingMetrics:
    """Compute MOT tracking metrics (simplified)."""
    metrics = TrackingMetrics()
    
    # This is a simplified version - full MOT metrics require track association over time
    # For now, compute frame-level association quality
    
    # Group by frame
    gt_by_frame = defaultdict(list)
    pred_by_frame = defaultdict(list)
    
    for t in gt_tracks:
        gt_by_frame[t['frame']].append(t)
    for t in pred_tracks:
        pred_by_frame[t['frame']].append(t)
    
    total_gt = sum(len(v) for v in gt_by_frame.values())
    total_pred = sum(len(v) for v in pred_by_frame.values())
    
    # Simple frame-level matching
    matches = 0
    fp = 0
    fn = 0
    id_switches = 0
    
    for frame_id in set(list(gt_by_frame.keys()) + list(pred_by_frame.keys())):
        gts = gt_by_frame[frame_id]
        preds = pred_by_frame[frame_id]
        
        # Match by IoU
        for pred in preds:
            best_iou = 0
            best_gt = None
            for gt in gts:
                iou = compute_iou_2d(
                    (pred['x1'], pred['y1'], pred['x2'], pred['y2']),
                    (gt['x1'], gt['y1'], gt['x2'], gt['y2'])
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
            
            if best_iou >= iou_threshold and best_gt:
                matches += 1
                # Check ID consistency
                if pred.get('track_id') != best_gt.get('track_id'):
                    id_switches += 1
            else:
                fp += 1
        
        fn += len(gts) - sum(1 for gt in gts if any(
            compute_iou_2d(
                (pred['x1'], pred['y1'], pred['x2'], pred['y2']),
                (gt['x1'], gt['y1'], gt['x2'], gt['y2'])
            ) >= iou_threshold for pred in preds
        ))
    
    if total_gt > 0:
        metrics.mota = 1 - (fp + fn + id_switches) / total_gt
    
    if matches > 0:
        metrics.motp = 1.0  # Placeholder
    
    metrics.id_switches = id_switches
    metrics.fragmentations = 0  # Would need temporal analysis
    
    return metrics


def compute_acoustic_metrics(gt_events: List[Dict],
                             pred_events: List[Dict],
                             azimuth_threshold_deg: float = 15.0) -> AcousticMetrics:
    """Compute acoustic perception metrics."""
    metrics = AcousticMetrics()
    
    if not gt_events and not pred_events:
        metrics.doa_mae_deg = 0.0
        metrics.doa_rmse_deg = 0.0
        metrics.classification_accuracy = 1.0
        metrics.detection_precision = 1.0
        metrics.detection_recall = 1.0
        return metrics
    
    if not gt_events:
        metrics.detection_precision = 0.0
        metrics.detection_recall = 0.0
        return metrics
    
    if not pred_events:
        metrics.detection_precision = 0.0
        metrics.detection_recall = 0.0
        return metrics
    
    # Match events by time and class
    matched_gt = set()
    matched_pred = set()
    az_errors = []
    class_correct = 0
    total_matched = 0
    
    for i, gt in enumerate(gt_events):
        best_match = None
        best_score = float('inf')
        
        for j, pred in enumerate(pred_events):
            if j in matched_pred:
                continue
            
            # Time difference
            dt = abs(gt['t'] - pred['t'])
            
            # Azimuth difference (in degrees)
            daz = abs(gt['azimuth_rad'] - pred['azimuth_rad']) * 180 / math.pi
            daz = min(daz, 360 - daz)
            
            # Class match
            class_match = gt['acoustic_class'] == pred['acoustic_class']
            
            # Score: weighted combination
            score = dt * 10 + daz
            if class_match:
                score *= 0.5
            
            if score < best_score and daz < azimuth_threshold_deg:
                best_score = score
                best_match = j
        
        if best_match is not None:
            matched_gt.add(i)
            matched_pred.add(best_match)
            total_matched += 1
            
            daz = abs(gt['azimuth_rad'] - pred_events[best_match]['azimuth_rad']) * 180 / math.pi
            daz = min(daz, 360 - daz)
            az_errors.append(daz)
            
            if gt['acoustic_class'] == pred_events[best_match]['acoustic_class']:
                class_correct += 1
    
    fp = len(pred_events) - len(matched_pred)
    fn = len(gt_events) - len(matched_gt)
    
    if len(pred_events) > 0:
        metrics.detection_precision = len(matched_pred) / len(pred_events)
    if len(gt_events) > 0:
        metrics.detection_recall = len(matched_gt) / len(gt_events)
    
    if az_errors:
        metrics.doa_mae_deg = float(np.mean(az_errors))
        metrics.doa_rmse_deg = float(np.sqrt(np.mean(np.array(az_errors) ** 2)))
    
    if total_matched > 0:
        metrics.classification_accuracy = class_correct / total_matched
    
    return metrics


def aggregate_perception_metrics(detection: DetectionMetrics,
                                  segmentation: SegmentationMetrics,
                                  depth: DepthMetrics,
                                  tracking: TrackingMetrics,
                                  acoustic: AcousticMetrics) -> PerceptionMetrics:
    """Aggregate all perception metrics into a single bundle."""
    overall = PerceptionMetrics(
        detection=detection,
        segmentation=segmentation,
        depth=depth,
        tracking=tracking,
        acoustic=acoustic,
    )
    
    # Compute overall score (weighted average)
    weights = {
        'detection': 0.3,
        'segmentation': 0.2,
        'depth': 0.2,
        'tracking': 0.15,
        'acoustic': 0.15,
    }
    
    scores = [
        weights['detection'] * detection.map_50,
        weights['segmentation'] * segmentation.miou,
        weights['depth'] * max(0, 1 - depth.abs_rel),  # Lower abs_rel is better
        weights['tracking'] * max(0, tracking.mota),
        weights['acoustic'] * (1 - acoustic.doa_mae_deg / 90),  # Normalize
    ]
    
    overall.overall_score = sum(scores)
    
    return overall


# Indian traffic class names for metrics
INDIAN_CLASS_NAMES = {
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