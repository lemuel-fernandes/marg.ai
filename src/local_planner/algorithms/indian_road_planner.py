import numpy as np
import math
from typing import List, Tuple, Dict, Optional


class IndianRoadPlanner:
    def __init__(
        self,
        target_speed: float = 11.11,
        max_accel: float = 2.5,
        max_jerk: float = 2.0,
        car_width: float = 1.8,
        min_side_margin: float = 0.35,
        pothole_cost: float = 500.0,
        pedestrian_safe_dist: float = 1.5,
    ):
        # Motion profile constants
        self.TARGET_SPEED = target_speed  # ~40 km/h in m/s
        self.MAX_ACCEL = max_accel        # m/s^2
        self.MAX_JERK = max_jerk          # m/s^3
        self.CAR_WIDTH = car_width        # meters
        
        # Indian road parameters
        self.MIN_SIDE_MARGIN = min_side_margin  # Tighter lateral tolerance for narrow gaps (meters)
        self.POTHOLE_COST = pothole_cost
        self.PEDESTRIAN_SAFE_DIST = pedestrian_safe_dist

    def generate_frenet_trajectories(
        self, 
        c_speed: float, 
        c_d: float, 
        c_d_d: float, 
        c_d_dd: float, 
        s_0: float,
        target_speed: Optional[float] = None,
        d_range: Tuple[float, float, float] = (-1.8, 1.9, 0.3),
    ) -> List[Dict]:
        """
        Generates candidate trajectories in Frenet Frame (s, d).
        """
        trajectories = []
        v_target = target_speed if target_speed is not None else self.TARGET_SPEED

        # Lateral sampling offsets (allow assertive nudging up to 1.8m off-center)
        d_samples = np.arange(d_range[0], d_range[1], d_range[2])
        # Prediction horizon sampling (1.5 to 3.5 seconds)
        t_samples = np.arange(1.5, 3.6, 0.5)
        # Target velocity sampling
        step_v = max(1.0, v_target / 4.0) if v_target > 0.0 else 1.0
        v_samples = np.arange(0.0, v_target + 0.1, step_v)
        if len(v_samples) == 0 or v_samples[-1] < v_target:
            v_samples = np.append(v_samples, v_target)
        # Escape sampling: always include the current speed (and a faster option)
        # so the planner can thread a fast-crossing obstacle when coming to a
        # stop is itself unsafe (e.g. an animal sweeping through the stopped
        # position). The velocity cost still prefers v_target when feasible.
        for v_esc in (c_speed, c_speed + 1.0):
            if v_esc > v_samples[-1] and v_esc <= self.TARGET_SPEED + 1e-6:
                v_samples = np.append(v_samples, v_esc)

        for Ti in t_samples:
            for di in d_samples:
                # Quintic polynomial for lateral motion: d(t)
                lat_coeffs = self._solve_quintic(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
                if lat_coeffs is None:
                    continue
                
                for vi in v_samples:
                    # Quartic polynomial for longitudinal motion: s(t)
                    long_coeffs = self._solve_quartic(s_0, c_speed, 0.0, vi, 0.0, Ti)
                    if long_coeffs is None:
                        continue
                    
                    traj = self._build_trajectory(lat_coeffs, long_coeffs, Ti, s_0, c_d)
                    trajectories.append(traj)
                    
        return trajectories

    def evaluate_trajectories(
        self, 
        trajectories: List[Dict], 
        obstacles: List[Dict], 
        road_anomalies: List[Dict],
        target_speed: Optional[float] = None,
        escape_allowance_s: float = 0.0,
    ) -> Optional[Dict]:
        """
        Ranks trajectories using cost functions tuned for dense, unorganized traffic.

        ``escape_allowance_s`` skips the veto for the first points of each
        candidate (deadlock-escape retries only); see _calc_obstacle_cost.
        """
        best_traj = None
        min_cost = float('inf')
        v_target = target_speed if target_speed is not None else self.TARGET_SPEED

        for traj in trajectories:
            # Quick check: if longitudinal jerk or end acceleration exceeds limits by large margin
            if traj['jerk_s'] > 2000.0:
                continue

            # 1. Smoothness Cost (Jerk minimization). Weighted far below the velocity
            # cost: with equal weights a small jerk saving can outweigh reaching a
            # usable speed, which stalls progress in slow-traffic states.
            cost_jerk = (traj['jerk_s'] + traj['jerk_d']) * 0.02
            
            # 2. Velocity Tracking Cost
            cost_vel = (v_target - traj['v_end']) ** 2
            
            # 3. Lateral Deviation Cost (Penalize driving off-center unless necessary)
            cost_lat = (traj['d_end']) ** 2 * 0.5
            
            # 4. Indian Driving Custom Logic: Dynamic Obstacle & Nudge Cost
            cost_obs = self._calc_obstacle_cost(
                traj, obstacles, escape_allowance_s=escape_allowance_s)
            if math.isinf(cost_obs):
                continue

            # 5. Surface Anomaly Cost (Potholes, Speed Breakers)
            cost_surface = self._calc_surface_cost(traj, road_anomalies)

            total_cost = cost_jerk + cost_vel + cost_lat + cost_obs + cost_surface

            if total_cost < min_cost:
                min_cost = total_cost
                best_traj = traj

        return best_traj

    def _calc_obstacle_cost(self, traj: Dict, obstacles: List[Dict],
                            escape_allowance_s: float = 0.0) -> float:
        cost = 0.0
        s_start = traj['s'][0]
        s_end = traj['s'][-1]

        # Fast spatial pre-filter: obstacles within [s_start - 3.0, s_end + 5.0]
        relevant_obstacles = [
            obs for obs in obstacles
            if (s_start - 3.0) <= obs['s'] <= (s_end + 8.0)
        ]

        if not relevant_obstacles:
            return 0.0

        for t, (s, d, v) in enumerate(zip(traj['s'], traj['d'], traj['v'])):
            # Dynamic safety margin shrinks at low speeds to allow squeezing through bottlenecks
            dynamic_margin = self.MIN_SIDE_MARGIN + 0.1 * v
            t_delta = t * 0.1

            # Escape allowance (ONLY active for the deadlock-escape retry):
            # skip the first points. When the ego is already stopped inside a
            # fattened veto box (e.g. held behind a blocker by an acoustic
            # hold), vetoing trajectories at t=0 makes escape geometrically
            # impossible (every candidate starts at the current pose). The
            # downstream safety monitor is the hard gate with true OBB
            # geometry. Regular evaluation keeps the full veto from t=0.
            if t_delta < escape_allowance_s:
                continue

            for obs in relevant_obstacles:
                obs_s_pred = obs['s'] + obs.get('v_s', 0.0) * t_delta
                obs_d_pred = obs['d'] + obs.get('v_d', 0.0) * t_delta
                
                dist_s = abs(s - obs_s_pred)
                dist_d = abs(d - obs_d_pred)
                
                obs_type = str(obs.get('type', '')).lower()
                obs_w = obs.get('width', 1.0)
                obs_l = obs.get('length', 2.0)
                half_widths = 0.5 * (self.CAR_WIDTH + obs_w)

                # Pedestrian, Cattle & Two-Wheeler Priority Override.
                # Strong repulsive cost instead of a hard veto: rejecting every candidate
                # leaves the planner without a plan and deadlocks the vehicle; the safety
                # monitor downstream still enforces the hard space-time envelope.
                if any(k in obs_type for k in ['pedestrian', 'cattle', 'two_wheeler', 'animal']):
                    if dist_s < self.PEDESTRIAN_SAFE_DIST and dist_d < (half_widths + 0.8):
                        cost += 500.0

                # Collision check with dynamic clearance box (obstacle-width aware).
                # Longitudinal pad is small: the downstream safety monitor is the
                # hard gate; an oversized pad engulfs the ego in the veto zone
                # during close encounters and starves the planner of any plan.
                if dist_s < (obs_l / 2.0 + 0.6) and dist_d < (half_widths + dynamic_margin - 0.2):
                    if v > 0.05:
                        return float('inf')
                    # Stationary point inside a predicted sweep: waiting cannot
                    # CAUSE a collision, so keep it available as a heavy-cost
                    # option instead of vetoing it. Hard-vetoing every candidate
                    # (including all-stop) deadlocks a stopped ego whose stopped
                    # position the crossing actor's CV corridor sweeps through —
                    # the planner returns no plan, the monitor rejects the brake
                    # ramp, and the pipeline emergency-stops every tick (372
                    # estops observed in city_roads). The safety monitor's 0.1 m
                    # margin at v=0 remains the hard gate on real proximity.
                    cost += 300.0
                
                # Proximity cost field
                dist = math.hypot(dist_s, dist_d)
                if dist < 3.0:
                    cost += math.exp(-0.8 * dist) * 100.0

        return cost

    def _calc_surface_cost(self, traj: Dict, road_anomalies: List[Dict]) -> float:
        if not road_anomalies:
            return 0.0
        cost = 0.0
        s_start = traj['s'][0]
        s_end = traj['s'][-1]
        relevant_anomalies = [
            a for a in road_anomalies
            if (s_start - 1.0) <= a['s'] <= (s_end + 1.0)
        ]

        for s, d, v in zip(traj['s'], traj['d'], traj['v']):
            for anomaly in relevant_anomalies:
                anom_type = str(anomaly.get('type', '')).lower()
                if 'pothole' in anom_type:
                    # Smooth cost field: full POTHOLE_COST at the anomaly
                    # center, tapering linearly to zero at the 0.8 m boundary.
                    # A hard box makes the cost a step function of perception
                    # noise — a ~2 cm centroid jitter flips a boundary-grazing
                    # candidate by the full +500, which destabilizes trajectory
                    # selection (and cascades via chaotic divergence).
                    ds = abs(s - anomaly['s'])
                    dd = abs(d - anomaly['d'])
                    if ds < 0.8 and dd < 0.8:
                        cost += self.POTHOLE_COST * (1.0 - max(ds, dd) / 0.8)
                elif 'speed_breaker' in anom_type:
                    # Slow down requirement instead of full evasion
                    if abs(s - anomaly['s']) < 0.8 and abs(d - anomaly['d']) < 0.8:
                        if v > 4.16:  # > 15 km/h over speed breaker
                            cost += 200.0 * (v - 4.16)
        return cost

    def _solve_quintic(self, xs, vxs, axs, xe, vxe, axe, T) -> Optional[np.ndarray]:
        """
        Analytic closed-form solution for quintic polynomial:
        x(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        """
        if T <= 1e-4:
            return None
        
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        b0 = xe - (xs + vxs * T + 0.5 * axs * T2)
        b1 = vxe - (vxs + axs * T)
        b2 = axe - axs

        # Closed-form inverse of A:
        # [10/T^3, -4/T^2, 1/(2*T)]
        # [-15/T^4, 7/T^3, -1/T^2]
        # [6/T^5, -3/T^4, 1/(2*T^3)]
        a3 = (10.0 / T3) * b0 - (4.0 / T2) * b1 + (0.5 / T) * b2
        a4 = (-15.0 / T4) * b0 + (7.0 / T3) * b1 - (1.0 / T2) * b2
        a5 = (6.0 / T5) * b0 - (3.0 / T4) * b1 + (0.5 / T3) * b2

        return np.array([xs, vxs, 0.5 * axs, a3, a4, a5])

    def _solve_quartic(self, xs, vxs, axs, vxe, axe, T) -> Optional[np.ndarray]:
        """
        Analytic closed-form solution for quartic polynomial with terminal velocity constraint:
        s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4
        """
        if T <= 1e-4:
            return None
        
        T2 = T * T
        T3 = T2 * T

        b0 = vxe - (vxs + axs * T)
        b1 = axe - axs

        # Closed-form inverse:
        # [1/T^2, -1/(3*T)]
        # [-1/(2*T^3), 1/(4*T^2)]
        a3 = (1.0 / T2) * b0 - (1.0 / (3.0 * T)) * b1
        a4 = (-0.5 / T3) * b0 + (0.25 / T2) * b1

        return np.array([xs, vxs, 0.5 * axs, a3, a4])

    def _build_trajectory(self, a_lat, a_long, T, s_0, d_0) -> Dict:
        t_vec = np.arange(0.0, T, 0.1)
        t_vec2 = t_vec * t_vec
        t_vec3 = t_vec2 * t_vec
        t_vec4 = t_vec3 * t_vec
        t_vec5 = t_vec4 * t_vec
        
        # Evaluate lateral position, speed, acceleration
        d = a_lat[0] + a_lat[1]*t_vec + a_lat[2]*t_vec2 + a_lat[3]*t_vec3 + a_lat[4]*t_vec4 + a_lat[5]*t_vec5
        d_d = a_lat[1] + 2*a_lat[2]*t_vec + 3*a_lat[3]*t_vec2 + 4*a_lat[4]*t_vec3 + 5*a_lat[5]*t_vec4
        d_dd = 2*a_lat[2] + 6*a_lat[3]*t_vec + 12*a_lat[4]*t_vec2 + 20*a_lat[5]*t_vec3
        
        # Evaluate longitudinal position, speed, acceleration
        s = a_long[0] + a_long[1]*t_vec + a_long[2]*t_vec2 + a_long[3]*t_vec3 + a_long[4]*t_vec4
        v = a_long[1] + 2*a_long[2]*t_vec + 3*a_long[3]*t_vec2 + 4*a_long[4]*t_vec3
        a_long_val = 2*a_long[2] + 6*a_long[3]*t_vec + 12*a_long[4]*t_vec2
        
        jerk_s = float(np.sum((6*a_long[3] + 24*a_long[4]*t_vec)**2))
        jerk_d = float(np.sum((6*a_lat[3] + 24*a_lat[4]*t_vec + 60*a_lat[5]*t_vec2)**2))

        return {
            't': t_vec,
            's': s, 'd': d,
            'v': v, 'a': a_long_val,
            'd_d': d_d, 'd_dd': d_dd,
            'v_end': float(v[-1]), 'd_end': float(d[-1]),
            'jerk_s': jerk_s, 'jerk_d': jerk_d
        }
