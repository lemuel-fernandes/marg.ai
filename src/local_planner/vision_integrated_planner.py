import numpy as np
import math
from typing import List, Tuple, Dict, Optional

from src.local_planner.algorithms.indian_road_planner import IndianRoadPlanner
from src.perception.vision_pipeline import VisionPerceptionPipeline
from src.common.coordinates.frenet import CartesianFrenetConverter


class VisionIntegratedPlanner(IndianRoadPlanner):
    """
    Wraps the vision perception pipeline, converts visual detections into Frenet obstacles,
    and runs the adaptive Indian road trajectory optimization engine.
    """

    def __init__(
        self,
        intrinsic_matrix: np.ndarray,
        camera_extrinsics: np.ndarray,
        target_speed: float = 11.11,
        max_accel: float = 2.5,
        max_jerk: float = 2.0,
        car_width: float = 1.8,
    ):
        super().__init__(
            target_speed=target_speed,
            max_accel=max_accel,
            max_jerk=max_jerk,
            car_width=car_width,
        )
        self.vision_pipe = VisionPerceptionPipeline(intrinsic_matrix, camera_extrinsics)

    def cartesian_to_frenet_frame(
        self, 
        x_v: float, 
        y_v: float, 
        ref_line_waypoints: np.ndarray, 
        ego_s: float
    ) -> Tuple[float, float]:
        """
        Maps a 2D position in base_link (x_v: forward, y_v: left) to Frenet (s, d).
        ref_line_waypoints: Nx2 or Nx3 array [[x, y] or [x, y, heading_rad]]
        """
        converter = CartesianFrenetConverter(ref_line_waypoints)
        s, d, _, _ = converter.to_frenet(x_v, y_v)
        return s, d

    def plan_from_vision(
        self,
        current_state: Dict,
        raw_detections: List[Dict],
        seg_mask: Optional[np.ndarray],
        depth_map: Optional[np.ndarray],
        ref_line_waypoints: np.ndarray,
        d_visible: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        End-to-End Execution step: Vision Ingestion -> Frenet Conversion -> Cost Minimization
        current_state: {'s': float, 'd': float, 'v': float, 'd_d': float, 'd_dd': float}
        """
        # 1. Transform Camera detections into Vehicle Space (base_link)
        spatial_obstacles = self.vision_pipe.process_3d_detections(raw_detections)
        surface_anomalies = self.vision_pipe.process_surface_anomalies(seg_mask, depth_map)

        # 2. Convert spatial obstacles to Frenet Frame (s, d)
        frenet_obstacles = []
        for obs in spatial_obstacles:
            obs_s, obs_d = self.cartesian_to_frenet_frame(
                obs['x_v'], obs['y_v'], ref_line_waypoints, current_state['s']
            )
            frenet_obstacles.append({
                'type': obs['type'],
                's': obs_s,
                'd': obs_d,
                'v_s': obs.get('vx_v', 0.0),  # Longitudinal speed component
                'v_d': obs.get('vy_v', 0.0),  # Lateral speed component
                'length': obs.get('length', 2.0),
                'width': obs.get('width', 1.5),
            })

        # 3. Convert anomalies to Frenet Frame
        frenet_anomalies = []
        for anom in surface_anomalies:
            anom_s, anom_d = self.cartesian_to_frenet_frame(
                anom['x_v'], anom['y_v'], ref_line_waypoints, current_state['s']
            )
            frenet_anomalies.append({
                'type': anom['type'],
                's': anom_s,
                'd': anom_d
            })

        # 4. Optical Fallback: Occlusion-Aware Speed Scaling
        target_v = self.TARGET_SPEED
        if d_visible is not None and d_visible < 30.0:
            target_v = min(target_v, self.vision_pipe.compute_occlusion_speed_cap(d_visible))

        # 5. Generate & Rank Trajectories using IndianRoadPlanner logic
        candidate_trajectories = self.generate_frenet_trajectories(
            c_speed=current_state['v'],
            c_d=current_state.get('d', 0.0),
            c_d_d=current_state.get('d_d', 0.0),
            c_d_dd=current_state.get('d_dd', 0.0),
            s_0=current_state['s'],
            target_speed=target_v,
        )

        best_trajectory = self.evaluate_trajectories(
            candidate_trajectories,
            frenet_obstacles,
            frenet_anomalies,
            target_speed=target_v,
        )

        return best_trajectory
