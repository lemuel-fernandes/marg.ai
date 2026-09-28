"""YOLOv8 detector wrapper for Indian traffic detection.

Provides a unified interface for YOLOv8nano/YOLOv8s models with
Indian traffic class mapping. Falls back to synthetic detector
when model weights are not available.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from pathlib import Path


@dataclass
class YOLOConfig:
    """Configuration for YOLO detector."""
    model_path: str = "yolov8n.pt"  # Will download if not found
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cpu"  # "cpu", "cuda", "mps"
    input_size: Tuple[int, int] = (640, 640)
    classes: Optional[List[int]] = None  # Filter by class IDs
    
    # Indian traffic class mapping from COCO
    coco_to_indian: Dict[int, int] = None
    
    def __post_init__(self):
        if self.coco_to_indian is None:
            # COCO class ID -> Indian traffic class ID
            self.coco_to_indian = {
                0: 8,   # person -> pedestrian
                1: 7,   # bicycle -> bicycle
                2: 1,   # car -> car
                3: 3,   # motorcycle -> two_wheeler
                5: 4,   # bus -> bus
                7: 5,   # truck -> truck
                16: 9,  # dog -> cattle (approximate)
                17: 9,  # horse -> cattle
                18: 9,  # sheep -> cattle
                19: 9,  # cow -> cattle
            }


class YOLOv8Detector:
    """YOLOv8 detector with Indian traffic class support."""
    
    def __init__(self, config: YOLOConfig):
        self.config = config
        self.model = None
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO
            
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                print(f"Model {model_path} not found, will download...")
            
            self.model = YOLO(str(model_path))
            self.model.to(self.config.device)
            self.model_loaded = True
            print(f"YOLOv8 model loaded on {self.config.device}")
            return True
        except ImportError:
            print("ultralytics not installed. Using synthetic detector fallback.")
            self.model_loaded = False
            return False
        except Exception as e:
            print(f"Failed to load YOLOv8 model: {e}")
            self.model_loaded = False
            return False
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Run detection on image.
        
        Args:
            image: H x W x 3 RGB or BGR image
            
        Returns:
            List of detections with keys:
            - class_id: Indian traffic class ID
            - class_name: Class name
            - x1, y1, x2, y2: Bounding box in pixels
            - confidence: Detection confidence
        """
        if not self.model_loaded:
            return self._synthetic_fallback(image)
        
        try:
            # Run inference
            results = self.model(
                image,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.input_size,
                classes=self.config.classes,
                verbose=False,
            )
            
            detections = []
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                
                xyxy = boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(xyxy)):
                    coco_class = cls[i]
                    
                    # Map COCO class to Indian traffic class
                    indian_class = self.config.coco_to_indian.get(coco_class)
                    if indian_class is None:
                        continue
                    
                    class_names = {
                        1: 'car', 2: 'auto_rickshaw', 3: 'two_wheeler',
                        4: 'bus', 5: 'truck', 6: 'tractor', 7: 'bicycle',
                        8: 'pedestrian', 9: 'cattle', 10: 'dog',
                        11: 'pothole', 12: 'speed_breaker',
                        13: 'construction_barrier', 14: 'traffic_cone',
                        15: 'level_crossing_gate', 16: 'emergency_vehicle'
                    }
                    
                    detections.append({
                        'class_id': indian_class,
                        'class_name': class_names.get(indian_class, f'class_{indian_class}'),
                        'x1': float(xyxy[i, 0]),
                        'y1': float(xyxy[i, 1]),
                        'x2': float(xyxy[i, 2]),
                        'y2': float(xyxy[i, 3]),
                        'confidence': float(conf[i]),
                    })
            
            return detections
            
        except Exception as e:
            print(f"YOLOv8 inference failed: {e}")
            return self._synthetic_fallback(image)
    
    def _synthetic_fallback(self, image: np.ndarray) -> List[Dict]:
        """Synthetic fallback when model not available."""
        return []
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """Run detection on batch of images."""
        if not self.model_loaded:
            return [self._synthetic_fallback(img) for img in images]
        
        try:
            results = self.model(
                images,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.input_size,
                classes=self.config.classes,
                verbose=False,
            )
            
            all_detections = []
            for r in results:
                detections = []
                boxes = r.boxes
                if boxes is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()
                    cls = boxes.cls.cpu().numpy().astype(int)
                    
                    for i in range(len(xyxy)):
                        coco_class = cls[i]
                        indian_class = self.config.coco_to_indian.get(coco_class)
                        if indian_class is None:
                            continue
                        
                        class_names = {
                            1: 'car', 2: 'auto_rickshaw', 3: 'two_wheeler',
                            4: 'bus', 5: 'truck', 6: 'tractor', 7: 'bicycle',
                            8: 'pedestrian', 9: 'cattle', 10: 'dog',
                            11: 'pothole', 12: 'speed_breaker',
                            13: 'construction_barrier', 14: 'traffic_cone',
                            15: 'level_crossing_gate', 16: 'emergency_vehicle'
                        }
                        
                        detections.append({
                            'class_id': indian_class,
                            'class_name': class_names.get(indian_class, f'class_{indian_class}'),
                            'x1': float(xyxy[i, 0]),
                            'y1': float(xyxy[i, 1]),
                            'x2': float(xyxy[i, 2]),
                            'y2': float(xyxy[i, 3]),
                            'confidence': float(conf[i]),
                        })
                all_detections.append(detections)
            
            return all_detections
            
        except Exception as e:
            print(f"YOLOv8 batch inference failed: {e}")
            return [self._synthetic_fallback(img) for img in images]


class YOLOv8Segmenter:
    """YOLOv8-seg for semantic segmentation."""
    
    def __init__(self, config: YOLOConfig):
        self.config = config
        self.model = None
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self) -> bool:
        try:
            from ultralytics import YOLO
            
            # Use segmentation model
            model_path = self.config.model_path.replace('.pt', '-seg.pt')
            if 'n' in model_path:
                model_path = model_path.replace('yolov8n', 'yolov8n-seg')
            elif 's' in model_path:
                model_path = model_path.replace('yolov8s', 'yolov8s-seg')
            
            self.model = YOLO(model_path)
            self.model.to(self.config.device)
            self.model_loaded = True
            return True
        except Exception as e:
            print(f"Failed to load YOLOv8-seg: {e}")
            self.model_loaded = False
            return False
    
    def segment(self, image: np.ndarray) -> np.ndarray:
        """Return segmentation mask with class IDs."""
        if not self.model_loaded:
            return np.zeros(image.shape[:2], dtype=np.int32)
        
        try:
            results = self.model(image, verbose=False)
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()  # N x H x W
                cls = results[0].boxes.cls.cpu().numpy().astype(int)
                
                # Combine masks by class (take highest confidence)
                h, w = image.shape[:2]
                seg_mask = np.zeros((h, w), dtype=np.int32)
                
                for i, c in enumerate(cls):
                    indian_class = self.config.coco_to_indian.get(c)
                    if indian_class is None:
                        continue
                    mask = masks[i]
                    if mask.shape != (h, w):
                        # Resize
                        from cv2 import resize
                        mask = resize(mask, (w, h))
                    seg_mask[mask > 0.5] = indian_class
                
                return seg_mask
            return np.zeros(image.shape[:2], dtype=np.int32)
        except Exception as e:
            print(f"YOLOv8-seg inference failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.int32)


def create_yolo_detector(
    model_size: str = 'n',  # 'n', 's', 'm', 'l', 'x'
    confidence: float = 0.25,
    device: str = 'cpu'
) -> YOLOv8Detector:
    """Factory function to create YOLOv8 detector."""
    config = YOLOConfig(
        model_path=f"yolov8{model_size}.pt",
        confidence_threshold=confidence,
        device=device,
    )
    return YOLOv8Detector(config)


def create_yolo_segmenter(
    model_size: str = 'n',
    confidence: float = 0.25,
    device: str = 'cpu'
) -> YOLOv8Segmenter:
    """Factory function to create YOLOv8 segmenter."""
    config = YOLOConfig(
        model_path=f"yolov8{model_size}.pt",
        confidence_threshold=confidence,
        device=device,
    )
    return YOLOv8Segmenter(config)


# Indian traffic specific class weights for training
INDIAN_TRAINING_CONFIG = {
    'data_yaml': '''
train: train.txt
val: val.txt
nc: 16
names: ['car', 'auto_rickshaw', 'two_wheeler', 'bus', 'truck', 
        'tractor', 'bicycle', 'pedestrian', 'cattle', 'dog',
        'pothole', 'speed_breaker', 'construction_barrier', 
        'traffic_cone', 'level_crossing_gate', 'emergency_vehicle']
''',
    'class_weights': {
        'car': 1.0,
        'auto_rickshaw': 2.0,
        'two_wheeler': 2.0,
        'bus': 1.5,
        'truck': 1.5,
        'tractor': 1.5,
        'bicycle': 1.5,
        'pedestrian': 2.0,
        'cattle': 3.0,
        'dog': 1.5,
        'pothole': 3.0,
        'speed_breaker': 2.0,
        'construction_barrier': 1.5,
        'traffic_cone': 1.0,
        'level_crossing_gate': 3.0,
        'emergency_vehicle': 3.0,
    },
    'augmentation': {
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.5,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 2.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
    }
}


def train_yolo_on_synthetic_data(
    data_dir: str = "data/synthetic_indian_traffic",
    model_size: str = 'n',
    epochs: int = 100,
    batch_size: int = 16,
    device: str = 'cpu',
) -> str:
    """
    Train YOLOv8 on synthetic Indian traffic dataset.
    
    Returns path to best model weights.
    """
    try:
        from ultralytics import YOLO
        
        # Write data.yaml
        import yaml
        data_yaml_path = Path(data_dir) / "data.yaml"
        with open(data_yaml_path, 'w') as f:
            f.write(INDIAN_TRAINING_CONFIG['data_yaml'])
        
        # Load model
        model = YOLO(f"yolov8{model_size}.pt")
        
        # Train
        results = model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            batch=batch_size,
            device=device,
            project="runs/train",
            name=f"yolov8{model_size}_indian_traffic",
            exist_ok=True,
            pretrained=True,
            optimizer='auto',
            verbose=True,
            # Augmentation
            mosaic=INDIAN_TRAINING_CONFIG['augmentation']['mosaic'],
            mixup=INDIAN_TRAINING_CONFIG['augmentation']['mixup'],
            copy_paste=INDIAN_TRAINING_CONFIG['augmentation']['copy_paste'],
            degrees=INDIAN_TRAINING_CONFIG['augmentation']['degrees'],
            translate=INDIAN_TRAINING_CONFIG['augmentation']['translate'],
            scale=INDIAN_TRAINING_CONFIG['augmentation']['scale'],
            shear=INDIAN_TRAINING_CONFIG['augmentation']['shear'],
            fliplr=INDIAN_TRAINING_CONFIG['augmentation']['fliplr'],
            hsv_h=INDIAN_TRAINING_CONFIG['augmentation']['hsv_h'],
            hsv_s=INDIAN_TRAINING_CONFIG['augmentation']['hsv_s'],
            hsv_v=INDIAN_TRAINING_CONFIG['augmentation']['hsv_v'],
        )
        
        # Return best model path
        best_model = Path("runs/train") / f"yolov8{model_size}_indian_traffic" / "weights" / "best.pt"
        return str(best_model)
        
    except Exception as e:
        print(f"Training failed: {e}")
        return ""


if __name__ == "__main__":
    # Test detector creation
    detector = create_yolo_detector('n', confidence=0.25)
    print(f"Detector created, model loaded: {detector.model_loaded}")
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_image)
    print(f"Detections on dummy image: {len(detections)}")