import numpy as np
import math
from typing import List, Tuple, Dict, Optional


def _normalize(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _connected_components(pixels) -> List:
    """Group pixel coordinates into 4-connected components.

    `pixels`: Nx2 array of (row, col). Returns a list of arrays, one per
    component, in deterministic (row-major) seed order.
    """
    remaining = set(map(tuple, pixels))
    components = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            v, u = stack.pop()
            for nb in ((v - 1, u), (v + 1, u), (v, u - 1), (v, u + 1)):
                if nb in remaining:
                    remaining.discard(nb)
                    stack.append(nb)
                    comp.append(nb)
        components.append(np.array(comp, dtype=np.int64))
    return components


class VisionPerceptionPipeline:
    """
    Ingests multi-modal perception data (3D bounding boxes, semantic surface masks, depth)
    and transforms detections from camera coordinates to ego vehicle coordinates (base_link).
    Includes optical fallback mechanisms for occlusion, tracking loss, and miscalibration.
    """

    def __init__(self, intrinsic_matrix: np.ndarray, extrinsics_rt: np.ndarray):
        """
        Camera Intrinsic (K) and Extrinsic Transformation (R|T) Camera -> BaseLink
        intrinsic_matrix: 3x3 array
        extrinsics_rt: 3x4 or 4x4 array
        """
        self.K = np.array(intrinsic_matrix, dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)
        self.R_c2v = np.array(extrinsics_rt[:3, :3], dtype=np.float64)
        self.T_c2v = np.array(extrinsics_rt[:3, 3], dtype=np.float64).reshape(3, 1)

        # Temporal tracking memory: track_id -> {obs_dict, last_seen_t}
        self._tracks: Dict[int, Dict] = {}
        self.TRACK_PERSISTENCE_S = 1.5

        # Miscalibration compensation (online pitch/roll offset)
        self._pitch_offset_rad = 0.0
        self._roll_offset_rad = 0.0

    def process_3d_detections(self, raw_visual_detections: List[Dict], current_time: float = 0.0,
                              ego_pose=None) -> List[Dict]:
        """
        Converts raw visual bounding boxes (camera frame) to Ego Vehicle Frame (base_link).
        raw_visual_detections item format:
        {'id': 1, 'class': 'auto_rickshaw', 'bbox_3d_cam': [X_c, Y_c, Z_c], 'velocity_cam': [Vx_c, Vz_c]}

        ego_pose: optional Pose2D of the ego at `current_time`. When supplied,
        track persistence anchors observations in the map frame so lost-track
        extrapolation compensates for ego motion (a base_link snapshot goes
        stale as the ego drives; without compensation a persisted track drifts
        backward along the road at exactly the ego's speed — appearing to cut
        across the ego's path). When omitted, persistence falls back to the
        legacy base_link extrapolation (valid only for a stationary ego).
        """
        spatial_obstacles = []
        observed_ids = set()

        for idx, det in enumerate(raw_visual_detections):
            P_c = np.array(det['bbox_3d_cam'], dtype=np.float64).reshape(3, 1)
            
            # Apply miscalibration compensation
            R_calib = self._get_calibration_matrix()
            P_v = (self.R_c2v @ R_calib @ P_c) + self.T_c2v
            
            # Transform velocity vector to base_link
            vel_cam = det.get('velocity_cam', [0.0, 0.0])
            V_c = np.array([vel_cam[0], 0.0, vel_cam[1]], dtype=np.float64).reshape(3, 1)
            V_v = self.R_c2v @ R_calib @ V_c

            track_id = det.get('id', idx + 1)
            observed_ids.add(track_id)

            obs_data = {
                'id': track_id,
                'type': det['class'],
                'x_v': float(P_v[0, 0]),  # Forward distance (meters)
                'y_v': float(P_v[1, 0]),  # Lateral offset (meters)
                'z_v': float(P_v[2, 0]),
                'vx_v': float(V_v[0, 0]),
                'vy_v': float(V_v[1, 0]),
                'length': det.get('length', 2.0),
                'width': det.get('width', 1.5),
            }
            # Semantic passthrough: synthetic-camera detections carry the full
            # ground-truth metadata; preserve it so downstream nodes can
            # rebuild faithful Obstacle records (harmless for real det stacks
            # that do not provide these keys).
            for extra_key in ('behavior', 'is_dynamic', 'confidence',
                              'class_label', 'rel_heading'):
                if extra_key in det:
                    obs_data[extra_key] = det[extra_key]
            spatial_obstacles.append(obs_data)

            # Update temporal tracking memory
            track_entry = {
                'data': obs_data,
                'last_seen': current_time,
            }
            if ego_pose is not None:
                # Anchor the observation in the map frame so persistence can
                # extrapolate world-fixed (constant-velocity) while the ego
                # keeps moving between observations.
                c = math.cos(ego_pose.heading)
                s = math.sin(ego_pose.heading)
                track_entry['p_map'] = (
                    ego_pose.x + obs_data['x_v'] * c - obs_data['y_v'] * s,
                    ego_pose.y + obs_data['x_v'] * s + obs_data['y_v'] * c)
                track_entry['v_map'] = (
                    obs_data['vx_v'] * c - obs_data['vy_v'] * s,
                    obs_data['vx_v'] * s + obs_data['vy_v'] * c)
                track_entry['heading_map'] = (ego_pose.heading
                                              + float(obs_data.get('rel_heading', 0.0)))
            self._tracks[track_id] = track_entry

        # Temporal Persistence Fallback:
        # Keep tracking dynamic obstacles for at least 1.5s after visual loss
        expired_ids = []
        for tid, track_info in self._tracks.items():
            if tid not in observed_ids:
                dt_lost = current_time - track_info['last_seen']
                if dt_lost <= self.TRACK_PERSISTENCE_S:
                    last_obs = track_info['data'].copy()
                    if 'p_map' in track_info and ego_pose is not None:
                        # Ego-motion-compensated persistence: extrapolate the
                        # world-fixed map pose with the constant-velocity
                        # model, then re-project into the CURRENT base_link.
                        px = track_info['p_map'][0] + track_info['v_map'][0] * dt_lost
                        py = track_info['p_map'][1] + track_info['v_map'][1] * dt_lost
                        c = math.cos(ego_pose.heading)
                        s = math.sin(ego_pose.heading)
                        dx, dy = px - ego_pose.x, py - ego_pose.y
                        last_obs['x_v'] = dx * c + dy * s
                        last_obs['y_v'] = -dx * s + dy * c
                        last_obs['vx_v'] = track_info['v_map'][0] * c + track_info['v_map'][1] * s
                        last_obs['vy_v'] = -track_info['v_map'][0] * s + track_info['v_map'][1] * c
                        last_obs['rel_heading'] = _normalize(
                            track_info['heading_map'] - ego_pose.heading)
                    else:
                        # Legacy fallback (no ego pose available): stale
                        # base_link extrapolation — only correct for a
                        # stationary ego.
                        last_obs['x_v'] += last_obs['vx_v'] * dt_lost
                        last_obs['y_v'] += last_obs['vy_v'] * dt_lost
                    last_obs['is_persisted'] = True
                    spatial_obstacles.append(last_obs)
                else:
                    expired_ids.append(tid)

        for tid in expired_ids:
            del self._tracks[tid]

        return spatial_obstacles

    def process_surface_anomalies(
        self, 
        segmentation_mask: Optional[np.ndarray], 
        depth_map: Optional[np.ndarray]
    ) -> List[Dict]:
        """
        Extracts potholes and speed breakers from semantic segmentation + depth map.
        Class ID: 2 = Pothole, 3 = Speed Breaker

        Pixels are grouped into 4-connected components — one per road-surface
        anomaly — and each component is unprojected to a single base_link
        blob: centroid position plus a size estimate from the pixel spread
        (the synthetic depth stamp is constant per blob, so only the lateral
        spread is observable; the size is clamped to a sane anomaly footprint).
        Components whose depth falls entirely outside the validity band
        (0.2 < z <= 25 m) are dropped (too close/too far to trust).
        """
        if segmentation_mask is None or depth_map is None:
            return []

        anomalies = []
        R_calib = self._get_calibration_matrix()

        for class_id, label in [(2, 'pothole'), (3, 'speed_breaker')]:
            pixels = np.argwhere(segmentation_mask == class_id)
            if len(pixels) == 0:
                continue

            for comp in _connected_components(pixels):
                # Subsample unprojection: a close blob can span hundreds of
                # pixels and the centroid/spread estimate needs only a few.
                step = max(1, len(comp) // 64)
                pts = []
                for v, u in comp[::step]:
                    z_c = float(depth_map[v, u])
                    if z_c <= 0.2 or z_c > 25.0:  # Ignore invalid/far depth
                        continue
                    # Unproject pixel (u,v,z) -> 3D Camera -> 3D Vehicle Base Link
                    p_pixel = np.array([[u * z_c], [v * z_c], [z_c]])
                    P_c = self.K_inv @ p_pixel
                    P_v = (self.R_c2v @ R_calib @ P_c) + self.T_c2v
                    pts.append((float(P_v[0, 0]), float(P_v[1, 0])))

                if not pts:
                    continue
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                spread = max(math.hypot(p[0] - cx, p[1] - cy) for p in pts)
                size = min(1.5, max(0.3, 2.0 * spread + 0.2))
                anomalies.append({
                    'type': label,
                    'x_v': float(cx),
                    'y_v': float(cy),
                    'z_v': 0.0,
                    'length': size,
                    'width': size,
                })

        return anomalies

    def compute_occlusion_speed_cap(self, d_visible: float, a_comfortable: float = 2.0) -> float:
        """
        Optical Fallback: Occlusion-Aware Speed Scaling
        v_max = sqrt(2 * a_comfortable * d_visible)
        """
        if d_visible <= 0.5:
            return 0.5  # minimum crawl speed
        return float(math.sqrt(2.0 * a_comfortable * d_visible))

    def update_imu_miscalibration(self, imu_pitch_rad: float, imu_roll_rad: float):
        """
        Dynamically adapts online rotation for speed bumps, pitch tilt, and miscalibration.
        """
        self._pitch_offset_rad = imu_pitch_rad
        self._roll_offset_rad = imu_roll_rad

    def _get_calibration_matrix(self) -> np.ndarray:
        cp = math.cos(self._pitch_offset_rad)
        sp = math.sin(self._pitch_offset_rad)
        cr = math.cos(self._roll_offset_rad)
        sr = math.sin(self._roll_offset_rad)

        R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return R_pitch @ R_roll
