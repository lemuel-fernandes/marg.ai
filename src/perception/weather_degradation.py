"""Weather and lighting perception degradation models for Indian conditions.

Models realistic sensor degradation under:
- Monsoon/heavy rain
- Night/low-light
- Fog/mist
- Dust storms
- Glare (sunrise/sunset)
"""

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from enum import Enum


class WeatherCondition(Enum):
    CLEAR = "clear"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    MONSOON = "monsoon"
    FOG_LIGHT = "fog_light"
    FOG_DENSE = "fog_dense"
    DUST_STORM = "dust_storm"
    NIGHT = "night"
    NIGHT_RAIN = "night_rain"
    GLARE = "glare"


class TimeOfDay(Enum):
    DAY = "day"
    DAWN = "dawn"
    DUSK = "dusk"
    NIGHT = "night"


@dataclass
class WeatherParams:
    """Parameters defining weather/lighting effects on sensors."""
    # Visual sensor degradation
    visibility_range_m: float = 100.0          # Max visibility distance
    contrast_reduction: float = 1.0            # Contrast multiplier (0-1)
    brightness_shift: float = 0.0              # Brightness offset (-1 to 1)
    noise_std: float = 0.0                     # Added Gaussian noise std
    blur_kernel_size: int = 0                  # Motion/defocus blur
    lens_flare_intensity: float = 0.0          # Sun glare intensity
    raindrop_occlusion: float = 0.0            # Fraction of pixels occluded
    
    # LiDAR degradation (if applicable)
    lidar_range_reduction: float = 1.0
    lidar_false_positive_rate: float = 0.0
    lidar_dropout_rate: float = 0.0
    
    # Radar degradation
    radar_clutter_db: float = 0.0
    
    # Acoustic degradation
    acoustic_attenuation_db_per_m: float = 0.0
    acoustic_noise_floor_db: float = 40.0
    wind_noise_db: float = 0.0


# Preset weather parameters for Indian conditions
WEATHER_PRESETS = {
    WeatherCondition.CLEAR: WeatherParams(
        visibility_range_m=200.0,
        contrast_reduction=1.0,
        brightness_shift=0.0,
        noise_std=0.01,
        acoustic_noise_floor_db=35.0,
    ),
    
    WeatherCondition.LIGHT_RAIN: WeatherParams(
        visibility_range_m=80.0,
        contrast_reduction=0.8,
        brightness_shift=-0.1,
        noise_std=0.03,
        blur_kernel_size=3,
        raindrop_occlusion=0.05,
        acoustic_attenuation_db_per_m=0.1,
        acoustic_noise_floor_db=50.0,
        wind_noise_db=10.0,
    ),
    
    WeatherCondition.HEAVY_RAIN: WeatherParams(
        visibility_range_m=40.0,
        contrast_reduction=0.6,
        brightness_shift=-0.2,
        noise_std=0.05,
        blur_kernel_size=5,
        raindrop_occlusion=0.15,
        acoustic_attenuation_db_per_m=0.3,
        acoustic_noise_floor_db=60.0,
        wind_noise_db=20.0,
    ),
    
    WeatherCondition.MONSOON: WeatherParams(
        visibility_range_m=20.0,
        contrast_reduction=0.4,
        brightness_shift=-0.3,
        noise_std=0.08,
        blur_kernel_size=7,
        raindrop_occlusion=0.30,
        acoustic_attenuation_db_per_m=0.5,
        acoustic_noise_floor_db=70.0,
        wind_noise_db=30.0,
    ),
    
    WeatherCondition.FOG_LIGHT: WeatherParams(
        visibility_range_m=60.0,
        contrast_reduction=0.7,
        brightness_shift=0.1,
        noise_std=0.02,
        blur_kernel_size=5,
        acoustic_attenuation_db_per_m=0.05,
        acoustic_noise_floor_db=45.0,
    ),
    
    WeatherCondition.FOG_DENSE: WeatherParams(
        visibility_range_m=15.0,
        contrast_reduction=0.3,
        brightness_shift=0.2,
        noise_std=0.05,
        blur_kernel_size=9,
        acoustic_attenuation_db_per_m=0.1,
        acoustic_noise_floor_db=50.0,
    ),
    
    WeatherCondition.DUST_STORM: WeatherParams(
        visibility_range_m=30.0,
        contrast_reduction=0.5,
        brightness_shift=0.15,
        noise_std=0.06,
        blur_kernel_size=5,
        raindrop_occlusion=0.20,  # Dust particles
        acoustic_attenuation_db_per_m=0.2,
        acoustic_noise_floor_db=55.0,
        wind_noise_db=25.0,
    ),
    
    WeatherCondition.NIGHT: WeatherParams(
        visibility_range_m=50.0,  # Headlight range
        contrast_reduction=0.5,
        brightness_shift=-0.5,
        noise_std=0.08,
        lens_flare_intensity=0.1,  # Oncoming headlights
        acoustic_noise_floor_db=30.0,  # Quieter at night
    ),
    
    WeatherCondition.NIGHT_RAIN: WeatherParams(
        visibility_range_m=25.0,
        contrast_reduction=0.3,
        brightness_shift=-0.6,
        noise_std=0.12,
        blur_kernel_size=7,
        raindrop_occlusion=0.25,
        lens_flare_intensity=0.3,  # Headlight reflections
        acoustic_attenuation_db_per_m=0.3,
        acoustic_noise_floor_db=55.0,
        wind_noise_db=15.0,
    ),
    
    WeatherCondition.GLARE: WeatherParams(
        visibility_range_m=40.0,
        contrast_reduction=0.4,
        brightness_shift=0.5,
        noise_std=0.03,
        lens_flare_intensity=0.8,
        acoustic_noise_floor_db=40.0,
    ),
}


class WeatherDegradationModel:
    """Applies weather/lighting degradation to sensor data."""
    
    def __init__(self, weather: WeatherCondition = WeatherCondition.CLEAR,
                 time_of_day: TimeOfDay = TimeOfDay.DAY,
                 seed: int = 42):
        self.weather = weather
        self.time_of_day = time_of_day
        self.params = WEATHER_PRESETS[weather]
        self._rng = np.random.default_rng(seed)
    
    def set_weather(self, weather: WeatherCondition, time_of_day: TimeOfDay = None):
        """Change weather condition dynamically."""
        self.weather = weather
        self.params = WEATHER_PRESETS[weather]
        if time_of_day is not None:
            self.time_of_day = time_of_day
    
    def degrade_camera_image(self, image: np.ndarray) -> np.ndarray:
        """Apply weather effects to camera image."""
        if image is None:
            return None
        
        degraded = image.astype(np.float32) / 255.0
        h, w = degraded.shape[:2]
        
        # Brightness shift
        if self.params.brightness_shift != 0:
            degraded = np.clip(degraded + self.params.brightness_shift, 0, 1)
        
        # Contrast reduction
        if self.params.contrast_reduction != 1.0:
            mean = np.mean(degraded, axis=(0, 1), keepdims=True)
            degraded = mean + self.params.contrast_reduction * (degraded - mean)
            degraded = np.clip(degraded, 0, 1)
        
        # Gaussian noise
        if self.params.noise_std > 0:
            noise = self._rng.normal(0, self.params.noise_std, degraded.shape)
            degraded = np.clip(degraded + noise, 0, 1)
        
        # Blur (rain/fog)
        if self.params.blur_kernel_size > 1:
            from cv2 import GaussianBlur
            k = self.params.blur_kernel_size
            if k % 2 == 0:
                k += 1
            degraded = GaussianBlur(degraded, (k, k), 0)
        
        # Lens flare (glare/sunrise/sunset)
        if self.params.lens_flare_intensity > 0:
            # Add radial gradient flare from random position
            flare_center = (self._rng.integers(0, w), self._rng.integers(0, h//3))  # Upper portion
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - flare_center[0])**2 + (Y - flare_center[1])**2)
            max_dist = np.sqrt(w**2 + h**2)
            flare = self.params.lens_flare_intensity * np.exp(-dist / (max_dist * 0.3))
            flare = flare[:, :, np.newaxis]
            degraded = np.clip(degraded + flare, 0, 1)
        
        # Raindrop/dust occlusion
        if self.params.raindrop_occlusion > 0:
            mask = self._rng.random((h, w)) > self.params.raindrop_occlusion
            mask = mask[:, :, np.newaxis].astype(np.float32)
            degraded = degraded * mask
        
        return (degraded * 255).astype(np.uint8)
    
    def degrade_detections(self, detections: List[Dict]) -> List[Dict]:
        """Apply weather effects to 2D detections."""
        degraded = []
        
        for det in detections:
            # Distance-based dropout
            distance = det.get('distance_m', 10.0)
            if distance > self.params.visibility_range_m:
                # Object beyond visibility range - drop with probability
                dropout_prob = min(1.0, (distance - self.params.visibility_range_m) / 50.0)
                if self._rng.random() < dropout_prob:
                    continue
            
            # Confidence reduction based on distance and weather
            confidence = det.get('confidence', 0.9)
            dist_factor = min(1.0, distance / self.params.visibility_range_m)
            weather_factor = self.params.contrast_reduction
            new_confidence = confidence * weather_factor * (1 - 0.5 * dist_factor)
            
            # Add position noise (worse in bad weather)
            pos_noise = self.params.noise_std * 10 * dist_factor
            if 'x1' in det:
                det = det.copy()
                det['x1'] += self._rng.normal(0, pos_noise)
                det['y1'] += self._rng.normal(0, pos_noise)
                det['x2'] += self._rng.normal(0, pos_noise)
                det['y2'] += self._rng.normal(0, pos_noise)
            
            det = det.copy()
            det['confidence'] = max(0.01, min(1.0, new_confidence))
            degraded.append(det)
        
        return degraded
    
    def degrade_depth_map(self, depth: np.ndarray) -> np.ndarray:
        """Apply weather effects to depth map."""
        if depth is None:
            return None
        
        degraded = depth.copy().astype(np.float32)
        
        # Range clipping
        degraded[degraded > self.params.visibility_range_m] = 0
        
        # Add noise (worse at distance)
        if self.params.noise_std > 0:
            h, w = degraded.shape
            # Noise increases with distance
            y_coords = np.arange(h)[:, np.newaxis]
            dist_factor = 1 + y_coords / h  # Bottom of image = closer
            noise = self._rng.normal(0, self.params.noise_std * degraded * dist_factor)
            degraded = degraded + noise
        
        # Random dropouts (rain/fog scattering)
        if self.params.raindrop_occlusion > 0:
            mask = self._rng.random(degraded.shape) > self.params.raindrop_occlusion
            degraded[~mask] = 0
        
        return np.clip(degraded, 0, self.params.visibility_range_m)
    
    def degrade_segmentation(self, seg: np.ndarray) -> np.ndarray:
        """Apply weather effects to segmentation mask."""
        if seg is None:
            return None
        
        degraded = seg.copy()
        
        # Random class confusion (worse in bad weather)
        if self.params.contrast_reduction < 0.7:
            h, w = degraded.shape
            num_pixels = h * w
            num_confuse = int(num_pixels * (1 - self.params.contrast_reduction) * 0.1)
            
            for _ in range(num_confuse):
                y, x = self._rng.integers(0, h), self._rng.integers(0, w)
                # Confuse with neighboring class
                current_val = int(degraded[y, x])
                if self._rng.random() < 0.5:
                    degraded[y, x] = max(0, current_val - 1)
                else:
                    degraded[y, x] = min(16, current_val + 1)
        
        return degraded
    
    def degrade_acoustic_events(self, events: List[Dict]) -> List[Dict]:
        """Apply weather effects to acoustic events."""
        degraded = []
        
        for ev in events:
            distance = ev.get('distance_m', 10.0)  # If available
            
            # Attenuation
            attenuation = self.params.acoustic_attenuation_db_per_m * distance
            snr_loss = attenuation + self.params.wind_noise_db
            
            # Confidence reduction
            conf = ev.get('confidence', 0.9)
            new_conf = max(0.1, conf - snr_loss / 100.0)
            
            # Dropout for very low SNR
            if new_conf < 0.2 and self._rng.random() < 0.5:
                continue
            
            new_ev = ev.copy()
            new_ev['confidence'] = new_conf
            new_ev['snr_db'] = ev.get('snr_db', 20) - snr_loss
            degraded.append(new_ev)
        
        return degraded
    
    def get_effective_camera_range(self) -> float:
        """Get effective camera detection range in meters."""
        return self.params.visibility_range_m
    
    def get_effective_acoustic_range(self, source_level_db: float) -> float:
        """Get effective acoustic detection range for a source level."""
        # Solve: source_level - attenuation * range - noise_floor = detection_threshold
        threshold_db = 6.0  # Minimum detectable SNR
        available_db = source_level_db - self.params.acoustic_noise_floor_db - threshold_db
        if available_db <= 0:
            return 0.0
        return available_db / self.params.acoustic_attenuation_db_per_m

    def degrade_detections_3d(self, detections: List[Dict]) -> List[Dict]:
        """Apply weather effects to 3D camera-frame detections.
        
        Detections format: dict with keys like:
        - 'id', 'class', 'bbox_3d_cam': [X_c, Y_c, Z_c], 'velocity_cam': [Vx_c, Vz_c],
        - 'length', 'width', 'cls', 'behavior', 'is_dynamic', 'confidence', 'rel_heading'
        """
        if self.params.contrast_reduction >= 1.0 and self.params.noise_std == 0:
            return detections  # No degradation
        
        degraded = []
        
        for det in detections:
            # Distance-based dropout
            bbox_3d = det.get('bbox_3d_cam', [0, 0, 10])
            z_c = bbox_3d[2] if len(bbox_3d) > 2 else 10.0  # Forward distance
            
            if z_c > self.params.visibility_range_m:
                # Beyond visibility - probabilistic dropout
                dropout_prob = min(1.0, (z_c - self.params.visibility_range_m) / 50.0)
                if self._rng.random() < dropout_prob:
                    continue
            
            # Confidence reduction based on distance and weather
            confidence = det.get('confidence', 0.9)
            dist_factor = min(1.0, z_c / self.params.visibility_range_m)
            weather_factor = self.params.contrast_reduction
            new_confidence = confidence * weather_factor * (1 - 0.5 * dist_factor)
            
            # Position noise (worse in bad weather and at distance)
            pos_noise = self.params.noise_std * z_c * dist_factor * 2.0
            
            new_det = det.copy()
            new_det['confidence'] = max(0.01, min(1.0, new_confidence))
            
            # Add noise to 3D position
            if 'bbox_3d_cam' in new_det:
                bbox = list(new_det['bbox_3d_cam'])
                if len(bbox) >= 3:
                    bbox[0] += self._rng.normal(0, pos_noise)  # X_c (right)
                    bbox[1] += self._rng.normal(0, pos_noise)  # Y_c (down)
                    bbox[2] += self._rng.normal(0, pos_noise * 0.5)  # Z_c (forward) - less noise in depth
                    new_det['bbox_3d_cam'] = bbox
            
            # Add noise to velocity
            if 'velocity_cam' in new_det:
                vel = list(new_det['velocity_cam'])
                if len(vel) >= 2:
                    vel[0] += self._rng.normal(0, pos_noise * 0.5)
                    vel[1] += self._rng.normal(0, pos_noise * 0.5)
                    new_det['velocity_cam'] = vel
            
            degraded.append(new_det)
        
        return degraded


def create_weather_sequence(conditions: List[Tuple[WeatherCondition, float]],
                           time_of_day: TimeOfDay = TimeOfDay.DAY) -> List[WeatherDegradationModel]:
    """Create a sequence of weather models for scenario progression.
    
    Args:
        conditions: List of (WeatherCondition, duration_seconds)
        time_of_day: Base time of day
        
    Returns:
        List of WeatherDegradationModel instances
    """
    return [WeatherDegradationModel(w, time_of_day) for w, _ in conditions]


# Indian monsoon specific: progressive degradation
MONSOON_PROGRESSION = [
    (WeatherCondition.CLEAR, 300),       # 5 min clear
    (WeatherCondition.LIGHT_RAIN, 300),  # 5 min light rain
    (WeatherCondition.HEAVY_RAIN, 600),  # 10 min heavy rain
    (WeatherCondition.MONSOON, 600),     # 10 min monsoon
    (WeatherCondition.HEAVY_RAIN, 300),  # 5 min heavy rain
    (WeatherCondition.LIGHT_RAIN, 300),  # 5 min light rain
    (WeatherCondition.CLEAR, 300),       # 5 min clear
]

# Typical Indian day progression
DAY_NIGHT_CYCLE = [
    (WeatherCondition.CLEAR, 10800),     # 3h day
    (WeatherCondition.GLARE, 1800),      # 30min sunset glare
    (WeatherCondition.NIGHT, 7200),      # 2h night
    (WeatherCondition.NIGHT_RAIN, 3600), # 1h night rain
    (WeatherCondition.NIGHT, 3600),      # 1h night
    (WeatherCondition.GLARE, 1800),      # 30min sunrise glare
]


if __name__ == "__main__":
    # Test weather degradation
    import cv2
    
    # Create test image
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    for weather in [WeatherCondition.CLEAR, WeatherCondition.MONSOON, 
                    WeatherCondition.NIGHT, WeatherCondition.FOG_DENSE]:
        model = WeatherDegradationModel(weather)
        degraded = model.degrade_camera_image(test_img)
        print(f"{weather.value}: range={model.get_effective_camera_range():.0f}m, "
              f"mean_brightness={np.mean(degraded):.1f}")
    
    print("\nWeather degradation models ready!")