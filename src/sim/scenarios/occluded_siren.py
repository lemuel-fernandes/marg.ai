"""Occluded-siren scenario validating the acoustic attention coupling (#27b).

Setup: an ambulance (siren, SNR 20 dB) approaches from behind at 7 m/s but is
occluded (it only becomes a visible obstacle at t=6 while its siren is audible
from t=2). A parked truck blocks the lane ahead; the only egress is the narrow
right pass band. Correct behavior per the #27b contract:

- t<2s: no cues, ego drives/creeps toward the blocker (anti-deadlock logic).
- t in [2,6): siren audible, NOT visible, no visual track inside its azimuth
  cone -> unseen-cue hold suppresses the blind creep -> ego HOLDS at ~0 speed
  instead of pulling out into the path of the unseen overtaking ambulance.
- t>=6s: the ambulance visually overtakes in the pass band; the cue matches a
  visual track / expires -> hold releases -> ego overtakes the parked truck
  and proceeds to the goal.
"""
import math
from typing import List, Tuple

from src.common.types.base import Covariance2D, FrameId, Header, Pose2D, Twist2D
from src.common.types.config import CostmapConfig, DWAConfig, VehicleConfig
from src.common.types.obstacle import Obstacle, ObstacleBehavior, ObstacleClass
from src.common.types.vehicle_state import VehicleState
from src.mapping.costmap import RoadNetwork
from src.perception.audio_pipeline import AcousticEvent, CLASS_SIREN

from .base import Scenario, ScenarioResult, min_obstacle_clearance, boundary_violation

ROAD = [(0, 0), (60, 0)]
ROAD_HALF_WIDTH = 4.0    # 8 m road: leaves a genuine (narrow) pass band
AMBU_SPEED = 7.0
AMBU_START_X = -12.0     # starts behind the ego
AMBU_SPAWN_T = 6.0       # becomes visible (obstacle) only at t=6


class OccludedSirenScenario(Scenario):
    name = "occluded_siren"
    duration_s = 30.0
    min_ttc_gate_s = 0.25

    def acoustic_events(self):
        # Siren from directly behind (azimuth ~pi in base_link), 20 dB SNR.
        return [AcousticEvent(t_onset=2.0, t_end=6.0, acoustic_class=CLASS_SIREN,
                              azimuth_rad=math.pi, snr_db=20.0)]

    def configs(self):
        return (
            VehicleConfig(max_speed=6.0, max_steer_angle=0.6, wheelbase=2.5),
            CostmapConfig(width_m=120, height_m=60, resolution=0.5,
                          inflation_radius=1.0, fixed_origin=(-15.0, -15.0)),
            DWAConfig(obstacle_margin=1.5),
        )

    def road_network(self):
        return RoadNetwork(polylines=[ROAD], half_width=ROAD_HALF_WIDTH)

    def initial_state(self):
        return VehicleState(header=Header(0.0, FrameId.MAP, "scenario"),
                            pose=Pose2D(4.0, 0.0, 0.0),
                            twist=Twist2D(3.0, 0.0, 0.0), steering_angle=0.0)

    def goal(self):
        return Pose2D(32.0, 0.0, 0.0)

    def obstacles_at(self, t: float) -> List[Obstacle]:
        obs = []

        def add(tid, cls, x, y, hdg, vx, vy, L, W, dyn):
            obs.append(Obstacle(header=Header(t, FrameId.MAP, "scenario"), track_id=tid,
                                class_label=cls, behavior=ObstacleBehavior.STATIC,
                                pose=Pose2D(x, y, hdg), length=L, width=W,
                                velocity=Twist2D(vx=vx, vy=vy, yaw_rate=0.0),
                                pose_covariance=Covariance2D(), velocity_covariance=Covariance2D(),
                                confidence=0.9, is_dynamic=dyn))

        # Parked truck ahead in the same lane: the blocker the ego must not
        # pull out around while the unseen ambulance approaches from behind.
        truck_x = 20.0
        add(60, ObstacleClass.VEHICLE, truck_x, 0.0, 0.0, 0.0, 0.0, 6.0, 2.0, False)

        # Ambulance: siren source. It only becomes a VISIBLE obstacle at
        # t=AMBU_SPAWN_T when it "overtakes" — before that it exists
        # acoustically only (this is the occlusion the camera cannot see).
        if t >= AMBU_SPAWN_T:
            ambu_x = AMBU_START_X + AMBU_SPEED * (t - AMBU_SPAWN_T)
            add(62, ObstacleClass.VEHICLE, ambu_x, 2.5, 0.0,
                AMBU_SPEED, 0.0, 4.5, 2.0, True)

        return obs

    def get_local_planner(self, v_cfg, dwa_cfg):
        from src.local_planner.frenet_local_planner import AdaptiveFrenetLocalPlanner
        return AdaptiveFrenetLocalPlanner(v_cfg, target_speed=4.5, creep_speed=1.5)

    def plot_tracks(self):
        return [{"points": ROAD, "markers": [], "radius": 0.3, "color": "gray",
                 "ls": "-", "label": "Road"},
                {"points": [(-12.0, 0.0), (50.0, 0.0)], "markers": [],
                 "radius": 0.3, "color": "red", "ls": ":", "label": "Ambulance path"}]

    def evaluate(self, log, v_cfg) -> ScenarioResult:
        g = self.goal()
        final = log.states[-1].pose
        dist = math.hypot(g.x - final.x, g.y - final.y)

        # --- Acoustic coupling assertions (the point of this scenario) ---
        # Siren window [2, 6): ego must hold (speed ~0) while the siren is
        # audible but unseen, instead of escalating the blind creep. The
        # window starts at 4.5s: from 4.5 m/s the certified stop takes ~2.3s,
        # so the braking tail after the t=2.0 hold demand is excluded.
        hold_speeds = []
        for t, st in zip(log.times, log.states):
            if 4.5 <= t <= 5.8:
                hold_speeds.append(st.twist.vx)
        max_hold_speed = max(hold_speeds) if hold_speeds else 0.0

        # After the siren ends and the ambulance has passed, the ego must
        # release the hold, overtake the parked truck and make progress.
        x_at = dict((round(t, 1), st.pose.x)
                    for t, st in zip(log.times, log.states))
        x8 = x_at.get(8.0)
        x_end = log.states[-1].pose.x
        progress = (x_end - x8) if x8 is not None else 0.0
        progressed = progress > 5.0

        clear = min_obstacle_clearance(log, self.obstacles_at)
        bviol = boundary_violation(log, self.road_network())
        min_accel = min((c.acceleration for c in log.commands), default=0.0)
        metrics = {
            "final_dist_m": dist,
            "min_clearance_m": clear,
            "min_accel_mps2": min_accel,
            "emergency_stops": float(log.emergency_stops),
            "max_hold_speed_mps": max_hold_speed,
            "post_siren_progress_m": progress,
        }
        failures = []
        if clear < 0.2:
            failures.append(f"collision: min clearance {clear:.2f} m")
        if dist > 5.0:
            failures.append(f"did not reach goal (dist={dist:.2f} m)")
        if max_hold_speed > 0.5:
            failures.append(
                f"blind creep during unseen-siren hold (max {max_hold_speed:.2f} m/s)")
        if not progressed:
            failures.append(
                f"deadlocked after siren ended (progress {progress:.2f} m)")
        if bviol is not None:
            failures.append(f"boundary violation: ego left road corridor by {bviol:.2f} m")
        return ScenarioResult(self.name, not failures, metrics, failures)
