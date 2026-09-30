"""Synthetic camera adapter: bridge ground-truth sim state -> vision stack.

Produces the raw inputs `VisionPerceptionPipeline` and
`VisionIntegratedPlanner` consume (camera-frame 3D detections, semantic
segmentation mask, depth map) from ground-truth map-frame obstacles, by
reversing the camera projection. This lets the full vision path run inside
the deterministic scenario harness without a renderer:

    map obstacles --(adapter)--> camera frame --(vision pipeline)-->
    base_link --(transform tree)--> map frame

Roundtrip geometry is exact by construction (tests assert it); range/FOV
gating models real camera limits.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.common.types.obstacle import Obstacle


# Semantic segmentation class ids (mirrors vision_pipeline.py: 2=pothole,
# 3=speed_breaker; obstacles use 10+ObstacleClass index to avoid clashes).
SEG_POTHOLE = 2
SEG_SPEED_BREAKER = 3

# Obstacle classes rendered into the segmentation mask (road-surface
# anomalies). These reach the planner via the seg/depth unprojection path
# (VisionPerceptionPipeline.process_surface_anomalies), NOT the 3D-box
# detection path.
ANOMALY_CLASS_VALUES = ('pothole', 'speed_breaker')

# Camera model limits
DEFAULT_FOV_RAD = 1.2        # ~69 deg horizontal half-angle
DEFAULT_MAX_RANGE_M = 60.0   # detections beyond this are not reported

# Effective depth-sensing band for surface-anomaly extraction. Matches the
# validity window in VisionPerceptionPipeline.process_surface_anomalies
# (z_c <= 25 m); anomalies beyond it are not extractable from depth and stay
# on the 3D-detection / GT-passthrough path instead.
ANOMALY_MAX_DEPTH_M = 25.0


class SyntheticCameraAdapter:
    """Ground-truth -> camera-frame adapter for the vision pipeline."""

    def __init__(self, intrinsic_matrix: np.ndarray,
                 fov_rad: float = DEFAULT_FOV_RAD,
                 max_range_m: float = DEFAULT_MAX_RANGE_M):
        self.K = np.asarray(intrinsic_matrix, dtype=np.float64)
        self.cx = float(self.K[0, 2])
        self.cy = float(self.K[1, 2])
        self.fx = float(self.K[0, 0])
        self.fov_rad = fov_rad
        self.max_range_m = max_range_m

    # ---------- geometry helpers (camera frame: X right, Y down, Z fwd) ----

    def map_to_camera(self, x_map: float, y_map: float, z_map: float,
                      ego_pose) -> Tuple[float, float, float]:
        """Map point -> camera frame via base_link.

        base_link: X forward, Y left, Z up (matches TransformTree vehicle
        frame). Camera: X right, Y down, Z forward.
        """
        # map -> base_link (inverse of the tree's vehicle->map transform)
        dx, dy = x_map - ego_pose.x, y_map - ego_pose.y
        c, s = math.cos(ego_pose.heading), math.sin(ego_pose.heading)
        x_v = dx * c + dy * s
        y_v = -dx * s + dy * c
        z_v = z_map
        # base_link -> camera (fixed extrinsic: camera at origin of vehicle
        # frame looking forward, standard axis remap)
        x_c, y_c, z_c = -y_v, -z_v, x_v
        return x_c, y_c, z_c

    def _visible(self, x_c: float, y_c: float, z_c: float) -> bool:
        if z_c <= 0.5:  # behind or too close to project reliably
            return False
        if z_c > self.max_range_m:
            return False
        # Horizontal FOV gate (use atan2 on the azimuth in camera plane)
        if abs(math.atan2(x_c, z_c)) > self.fov_rad:
            return False
        return True

    def _project(self, x_c: float, y_c: float, z_c: float) -> Tuple[int, int, float]:
        u = self.cx + self.fx * (x_c / z_c)
        v = self.cy + self.fx * (y_c / z_c)
        return int(round(u)), int(round(v)), z_c

    # ---------- main API ---------------------------------------------------

    def make_detections(self, obstacles: List[Obstacle], ego_pose) -> List[Dict]:
        """Map-frame obstacles -> camera-frame 3D detection dicts.

        Format matches VisionPerceptionPipeline.process_3d_detections input:
        {'id', 'class', 'bbox_3d_cam': [X_c, Y_c, Z_c], 'velocity_cam': [Vx_c, Vz_c],
         'length', 'width', 'cls'}
        """
        dets: List[Dict] = []
        for obs in obstacles:
            x_c, y_c, z_c = self.map_to_camera(obs.pose.x, obs.pose.y, 0.0, ego_pose)
            if not self._visible(x_c, y_c, z_c):
                continue
            # Velocity: map -> base_link -> camera. base_link vel (vx_v fwd,
            # vy_v left) becomes camera vel (-vy_v, 0, vx_v) under the same
            # axis remap as positions (X_c right=-y_v, Z_c fwd=x_v). The
            # vision pipeline expects (Vx_c, Vz_c).
            c, s = math.cos(ego_pose.heading), math.sin(ego_pose.heading)
            vx_v = obs.velocity.vx * c + obs.velocity.vy * s
            vy_v = -obs.velocity.vx * s + obs.velocity.vy * c
            vel_cam = [-vy_v, vx_v]
            dets.append({
                'id': obs.track_id,
                'class': obs.class_label.value,
                'bbox_3d_cam': [x_c, y_c, z_c],
                'velocity_cam': vel_cam,
                'length': obs.length,
                'width': obs.width,
                'cls': obs.class_label,
                'behavior': obs.behavior,
                'is_dynamic': obs.is_dynamic,
                'confidence': obs.confidence,
                # Relative heading (obstacle - ego, CCW): base_link shares the
                # map frame's angular convention, so this survives the vehicle
                # -> map transform without passing through camera angles.
                'rel_heading': obs.pose.heading - ego_pose.heading,
            })
        return dets

    def anomalies_in_view(self, obstacles: List[Obstacle], ego_pose,
                          max_depth_m: float = ANOMALY_MAX_DEPTH_M
                          ) -> List[Obstacle]:
        """Road-surface anomalies the seg/depth path can actually extract.

        Mirrors the composite gate: camera visibility (forward of the vehicle,
        inside FOV/range) AND inside the depth-extraction band (z_c <=
        max_depth_m, matching process_surface_anomalies' validity window).
        The vision node uses this to keep anomalies OFF the 3D-detection path
        and OFF the GT passthrough while they are segmentable, so each
        pothole reaches the planner through exactly one path.
        """
        in_view = []
        for obs in obstacles:
            if obs.class_label.value not in ANOMALY_CLASS_VALUES:
                continue
            x_c, y_c, z_c = self.map_to_camera(obs.pose.x, obs.pose.y, 0.0, ego_pose)
            if z_c <= 0.5 or z_c > max_depth_m:
                continue
            if abs(math.atan2(x_c, z_c)) > self.fov_rad:
                continue
            in_view.append(obs)
        return in_view

    def make_seg_and_depth(self, obstacles: List[Obstacle], ego_pose,
                           width: int = 640, height: int = 480
                           ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Render potholes/speed breakers into a tiny semantic mask + depth map.

        Obstacles classified as POTHOLE/SPEED_BREAKER-pois are stamped as
        blobs at their projected pixels; everything else leaves the mask
        empty. Returns (None, None) when no anomalies exist so the vision
        node can skip segmentation entirely.
        """
        anomalies = [o for o in obstacles
                     if o.class_label.value in ANOMALY_CLASS_VALUES]
        if not anomalies:
            return None, None

        seg = np.zeros((height, width), dtype=np.uint8)
        depth = np.full((height, width), 25.0, dtype=np.float32)

        for obs in anomalies:
            x_c, y_c, z_c = self.map_to_camera(obs.pose.x, obs.pose.y, 0.0, ego_pose)
            if not self._visible(x_c, y_c, z_c):
                continue
            u, v, depth_m = self._project(x_c, y_c, z_c)
            cls_id = SEG_POTHOLE if obs.class_label.value == 'pothole' else SEG_SPEED_BREAKER
            # Blob radius grows with obstacle size, shrinks with distance.
            r = max(2, int(round(20.0 * max(obs.length, 0.4) / max(depth_m, 1.0))))
            u0, u1 = max(0, u - r), min(width, u + r + 1)
            v0, v1 = max(0, v - r), min(height, v + r + 1)
            seg[v0:v1, u0:u1] = cls_id
            depth[v0:v1, u0:u1] = depth_m
        return seg, depth
