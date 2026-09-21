import math
from typing import List, Optional
import numpy as np

from src.common.types.base import FrameId, Header, Pose2D, Twist2D, normalize_angle
from src.common.types.config import VehicleConfig
from src.common.types.costmap import Costmap
from src.common.types.obstacle import Obstacle, ObstacleClass, is_surface_anomaly
from src.common.types.path import GlobalPath
from src.common.types.trajectory import LocalTrajectory, TrajectoryPoint
from src.common.types.vehicle_state import VehicleState
from src.common.coordinates.frenet import CartesianFrenetConverter
from src.integration.contracts import LocalPlanner

from .algorithms.indian_road_planner import IndianRoadPlanner
from .behavior_state_machine import BehaviorStateMachine, PlannerState


class AdaptiveFrenetLocalPlanner(LocalPlanner):
    """
    Adaptive Frenet-frame local trajectory planner tuned for Indian road dynamics.
    Decouples lateral nudging d(t) from longitudinal speed profiling s(t) via quintic/quartic
    polynomials, with behavioral state machine governing intersection creeping and obstacle nudging.
    """

    def __init__(
        self,
        vehicle_cfg: VehicleConfig,
        target_speed: Optional[float] = None,
        creep_speed: float = 1.0,
    ):
        self.vcfg = vehicle_cfg
        self.target_speed = target_speed if target_speed is not None else vehicle_cfg.max_speed
        self.creep_speed = creep_speed

        self.planner = IndianRoadPlanner(
            target_speed=self.target_speed,
            max_accel=vehicle_cfg.max_acceleration,
            max_jerk=2.0,
            car_width=vehicle_cfg.width,
            min_side_margin=0.35,
            pothole_cost=500.0,
            pedestrian_safe_dist=1.8,
        )
        self.state_machine = BehaviorStateMachine(
            normal_target_speed=self.target_speed,
            creep_speed=self.creep_speed,
        )
        # ROADMAP #17 (global<->local handshake): the local planner may only
        # deviate from the global reference within this lateral half-width.
        # The A* reference itself meanders up to ~0.5 m from the road
        # centerline, and the scenario gate allows half_width - 0.5 m of road
        # distance: d_bound must satisfy d_bound + 0.5 < 3.0 with margin, so
        # 1.8 m (worst-case road distance ~2.3 m) — not the road half-width.
        # The frenet-d proxy is intentionally conservative; the road mask
        # remains the hard outer boundary.
        self.CORRIDOR_MAX_D = 1.8

        self._prev_d = 0.0
        self._prev_d_d = 0.0
        self._creep_steps = 0

        # Acoustic attention coupling (ROADMAP #27b): the pipeline pushes the
        # latest AttentionCues here each tick (see set_acoustic_cues).
        from src.perception.audio_pipeline import AudioPlannerPolicy
        self.audio_policy = AudioPlannerPolicy()
        self._acoustic_cues: List = []

    def set_acoustic_cues(self, cues: List) -> None:
        """Receive the current tick's acoustic attention cues (may be empty)."""
        self._acoustic_cues = list(cues)

    def plan(
        self,
        state: VehicleState,
        global_path: GlobalPath,
        costmap: Costmap,
        obstacles: List[Obstacle],
    ) -> LocalTrajectory:
        pts = global_path.points
        if not pts or len(pts) < 2:
            return self.emergency_stop(state, reason="No valid global path")

        # Check arrival at global goal. Threshold must comfortably exceed the
        # stopping distance at the approach speed limit so the arrival brake
        # engages before the A* path degenerates to a couple of grid points
        # (a degenerate path makes Frenet obstacle projections meaningless).
        goal = pts[-1].pose
        dist_to_goal = math.hypot(goal.x - state.pose.x, goal.y - state.pose.y)
        if dist_to_goal < 4.5:
            return self._arrival_stop(state, goal)

        # Build reference line from global path. A* grid paths are jagged and
        # obstacle-hugging, but heavy smoothing is unsafe on curved roads: it
        # pulls the reference across the inner edge and off the road. Keep the
        # raw polyline and only extend it past the goal so obstacles beyond the
        # end (e.g. traffic continuing down the road) keep meaningful
        # longitudinal projections while the ego approaches.
        ref_coords = np.array([[p.pose.x, p.pose.y] for p in pts], dtype=np.float64)
        ref_coords = self._extend_reference(ref_coords, extra_length=8.0)
        # Extend the reference BACKWARD too: after a failed replan the ego can
        # legitimately sit behind the retained path's start (e.g. blocked
        # goal cell). Without a back-extension the Frenet frame degenerates —
        # to_frenet clamps every ego position to s=0 with wild d swings, and
        # candidates generated in that broken frame drive the vehicle off the
        # road (real boundary violation observed at city_roads' corner).
        ref_coords = self._extend_reference_backward(ref_coords, extra_length=10.0)
        converter = CartesianFrenetConverter(ref_coords)

        # Current ego Frenet state
        vx = max(state.twist.vx, 0.0)
        vy = state.twist.vy
        ego_s, ego_d, ego_s_dot, ego_d_dot = converter.to_frenet(
            state.pose.x, state.pose.y, vx, vy
        )

        # Estimate d_dd from previous step
        ego_d_dd = (ego_d_dot - self._prev_d_d) / 0.1
        self._prev_d = ego_d
        self._prev_d_d = ego_d_dot

        # Separate solid obstacles and road anomalies (potholes, speed breakers)
        frenet_obstacles = []
        frenet_anomalies = []
        min_ttc = float('inf')

        # Ego-centric vehicle-frame basis for the TTC/proximity rules below.
        # Path-Frenet coordinates distort true geometry when the ego is far off
        # the reference line (e.g. after intersection avoidance): along-path
        # distance then wildly overestimates proximity and lateral offsets of
        # two co-linearly-displaced actors cancel out. Vehicle-frame relative
        # geometry is always physically true. forward = along ego heading;
        # lateral = left-positive (matches FrameId.VEHICLE convention).
        c_h = math.cos(state.pose.heading)
        s_h = math.sin(state.pose.heading)

        for obs in obstacles:
            obs_s, obs_d, obs_vs, obs_vd = converter.to_frenet(
                obs.pose.x, obs.pose.y, obs.velocity.vx, obs.velocity.vy
            )

            # Vehicle-frame relative pose/velocity of the obstacle
            dx = obs.pose.x - state.pose.x
            dy = obs.pose.y - state.pose.y
            v_f = dx * c_h + dy * s_h            # forward separation
            v_l = -dx * s_h + dy * c_h           # lateral separation (left +)
            ov_f = obs.velocity.vx * c_h + obs.velocity.vy * s_h
            ov_l = -obs.velocity.vx * s_h + obs.velocity.vy * c_h

            is_pothole = is_surface_anomaly(obs)

            if is_pothole:
                frenet_anomalies.append({
                    'type': 'pothole',
                    's': obs_s,
                    'd': obs_d,
                })
            else:
                obs_dict = {
                    'type': str(obs.class_label.value),
                    's': obs_s,
                    'd': obs_d,
                    'v_s': obs_vs,
                    'v_d': obs_vd,
                    'length': obs.length,
                    'width': obs.width,
                    # Vehicle-frame relative geometry (forward/lateral, ego
                    # heading basis). Frenet (s, d) distorts when the ego is
                    # far from the reference line (e.g. corner cutting on
                    # arcs): co-linear displacements there map to large spurious
                    # d offsets. Keep both; use vehicle-frame for proximity and
                    # yield decisions, frenet for trajectory-space reasoning.
                    'v_f': v_f,
                    'v_l': v_l,
                }
                frenet_obstacles.append(obs_dict)

                # 1. Longitudinal TTC for in-corridor obstacles ahead (vehicle frame)
                dv = vx - ov_f
                if v_f > 0.0 and dv > 0.1 and abs(v_l) < 1.6:
                    ttc_long = v_f / dv
                    if ttc_long < min_ttc:
                        min_ttc = ttc_long

                # 2. Spatio-temporal TTC for crossing dynamic actors (animals,
                # bikes, pedestrians): time until the obstacle reaches the ego's
                # lateral line (v_l = 0), then check longitudinal overlap at
                # that instant.
                if abs(ov_l) > 0.2:
                    t_cross = (0.0 - v_l) / ov_l
                    if 0.0 < t_cross < 4.0:
                        pred_v_f = v_f + ov_f * t_cross
                        if abs(pred_v_f) < 0.5 * (self.vcfg.length + obs.length) + 1.2:
                            if t_cross < min_ttc:
                                min_ttc = t_cross

                # 3. Immediate spatial proximity: VRU currently inside / entering
                # the ego's corridor (vehicle frame).
                if any(k in str(obs.class_label.value) for k in ("pedestrian", "animal")):
                    if -1.0 < v_f < 6.0:
                        gap_d = abs(v_l) - 0.5 * (self.vcfg.width + obs.width)
                        if gap_d < 0.9:
                            t_impact = max(v_f, 0.0) / max(vx, 0.5)
                            if t_impact < min_ttc:
                                min_ttc = t_impact

        # Canonical ordering: trajectory-cost summation is float-order-sensitive
        # (a+b+c != c+b+a in floats), so the planner must evaluate obstacles and
        # anomalies in a deterministic order regardless of how perception emitted
        # them (3D track order vs blob-cluster order). Same set -> same decision.
        frenet_obstacles.sort(key=lambda o: (o['s'], o['d']))
        frenet_anomalies.sort(key=lambda a: (a['s'], a['d']))

        # State machine step
        gap_available = True
        sm_state, directives = self.state_machine.step(
            current_s=ego_s,
            current_d=ego_d,
            current_v=vx,
            obstacles=frenet_obstacles,
            gap_available=gap_available,
            min_ttc=min_ttc,
        )

        target_v = directives["target_speed"]
        # If spatio-temporal TTC is low, yield/slow down immediately
        if min_ttc < 1.8:
            target_v = min(target_v, 0.5)
        # More aggressive yielding for crossing dynamic actors (animals, bikes, pedestrians)
        # When the crossing TTC is very low and lateral distance is small, limit speed further
        if min_ttc < 1.0:
            target_v = min(target_v, 0.3)

        # Follow a slower moving leader in-corridor instead of creeping behind
        # it forever (state-machine yield targets are for stopped blockers and
        # crossing conflicts, not for car-following). In-corridor test uses
        # VEHICLE-frame lateral offset (true geometry), not frenet d, which
        # distorts on curved reference lines and misclassified crossing bikes
        # as followers (driving the ego into an oncoming bike at the
        # intersection = the t~10-18s single-tick estop cluster).
        leader_vs = None
        leader_ds = float('inf')
        for o in frenet_obstacles:
            ds_o = o['v_f']
            if 0.0 < ds_o < 20.0 and abs(o['v_l']) < 1.2 and o.get('v_s', 0.0) > 0.8:
                if ds_o < leader_ds:
                    leader_ds, leader_vs = ds_o, o['v_s']
        if leader_vs is not None and min_ttc > 2.0:
            target_v = max(target_v, min(leader_vs + 0.5, self.target_speed))

        # Acoustic attention coupling (ROADMAP #27b):
        # - Siren audible -> defensive speed cap (yield to the emergency
        #   vehicle even before it is positively identified visually).
        # - Audible cue with NO matching visual track in its azimuth cone ->
        #   something is approaching unseen: HOLD (do not creep blind). The
        #   anti-deadlock crawl below is suppressed while the cue persists.
        siren_cap = self.audio_policy.siren_cap(self._acoustic_cues)
        if siren_cap is not None:
            target_v = min(target_v, siren_cap)
        audible_unseen = self.audio_policy.creep_release(
            self._acoustic_cues,
            [self.audio_policy.track_azimuth((o.pose.x, o.pose.y), state.pose)
             for o in obstacles],
        )

        # Anti-deadlock: while creeping/stopped behind a blocking obstacle, progressively
        # widen the lateral search range and relax the yield speed so the planner can
        # commit to a crawl-past trajectory instead of oscillating forever.
        # (Never applied to the emergency directive — a true stop demand stays a stop;
        # escape-speed sampling below handles the "stopping is unsafe" case.)
        creep_window = 8.0
        if audible_unseen:
            # Audible but unseen (e.g. siren from an occluded overtaker):
            # HOLD position. This must be cue-driven, not creep-threshold-
            # driven — a 2.0 m/s siren cap above the creep threshold would
            # otherwise let the ego drive up to the blocker blind.
            # Lateral range is pinned near the current offset: letting the
            # proximity field drift the ego toward the road edge during the
            # hold wedges it into a position with no kinematic escape later.
            target_v = 0.0
            self._creep_steps = 0
            d_range = (ego_d - 0.3, ego_d + 0.3, 0.1)
        elif 0.0 < target_v <= 1.5:
            # Lateral tolerance 1.6: the blocker shadow after a hold includes
            # offsets around d=+-1.2; a 1.2 boundary makes blocked_ahead False
            # exactly when the escape must begin.
            blocked_ahead = any(
                0.0 < o['s'] - ego_s < creep_window and abs(o['d'] - ego_d) < 1.6
                for o in frenet_obstacles
            )
            crossing_wait = False
            if blocked_ahead:
                self._creep_steps = min(self._creep_steps + 1, 200)
                ds_ahead = min(
                    o['s'] - ego_s for o in frenet_obstacles
                    if 0.0 < o['s'] - ego_s < creep_window and abs(o['d'] - ego_d) < 1.6
                )
                # Crawl to a near-stop approaching the blocker: the lateral
                # escape must begin while there is still longitudinal room
                # (once alongside the tail, no forward candidate escapes).
                target_v = min(target_v, max(0.4, 0.5 * (ds_ahead - 3.5)))
                # A stopped blocker is deadlocked traffic; a CROSSING one is a
                # conflict to wait out, not one to squeeze past. Forcing the
                # crawl speed floor toward a crossing pedestrian/animal
                # overrides the TTC yield above and walks the ego into the
                # threat's path — the safety monitor then emergency-stops the
                # intrusion every tick (8 estops observed in city_roads).
                # Crossing conflicts must remain pure waits: no creep floor,
                # no lateral widening.
                crossing_wait = any(
                    0.0 < o['s'] - ego_s < creep_window
                    and abs(o['d'] - ego_d) < 1.6
                    and abs(o['v_d']) > 0.2
                    for o in frenet_obstacles
                )
                if crossing_wait:
                    self._creep_steps = 0
            if crossing_wait:
                target_v = 0.0
                d_range = directives.get("d_range", (-1.8, 1.9, 0.3))
            elif blocked_ahead:
                d_lo, d_hi, d_step = directives.get("d_range", (-1.8, 1.9, 0.3))
                widen = min(1.0, 0.1 * self._creep_steps)
                d_range = (min(d_lo, -1.8 - widen), max(d_hi, 1.9 + widen), d_step)
                target_v = max(target_v, 0.8 + 0.02 * self._creep_steps)
            else:
                d_range = directives.get("d_range", (-1.8, 1.9, 0.3))
        else:
            self._creep_steps = 0
            d_range = directives.get("d_range", (-1.8, 1.9, 0.3))

        if dist_to_goal < 8.0:
            target_v = min(target_v, math.sqrt(2.0 * 1.5 * max(dist_to_goal - 0.2, 0.0)))

        # Generate candidates
        trajectories = self.planner.generate_frenet_trajectories(
            c_speed=vx,
            c_d=ego_d,
            c_d_d=ego_d_dot,
            c_d_dd=ego_d_dd,
            s_0=ego_s,
            target_speed=target_v,
            d_range=d_range,
        )

        # Corridor admissibility: sample every candidate against the costmap's
        # static road-mask layer and reject those that leave the drivable
        # region. The road boundary is a hard constraint, not a soft
        # preference — the trajectory cost terms know nothing about the road
        # edge, so without this filter the planner can pick an off-corridor
        # line (e.g. cutting a curve's inside) that no downstream module
        # would catch.
        CORRIDOR_MAX_D = self.CORRIDOR_MAX_D

        # Corridor bound: candidates may not exceed this ABSOLUTE lateral
        # distance from the reference line (ROADMAP #17 safe corridor). It
        # must be absolute, not relative to the ego's current offset: a
        # relative bound legalizes the ratchet where each tick's fallback
        # accepts a candidate slightly further out than the last, walking the
        # ego meters off-road while every per-tick choice looks locally fine.
        corridor_bound = CORRIDOR_MAX_D

        def _in_corridor(traj) -> bool:
            res = costmap.resolution
            # Lateral distance from the reference line: the road-mask filter
            # below passes anywhere ON the road polylines — including a point
            # on a DIFFERENT road than the reference (e.g. off the main road
            # but across the cross-street). What keeps the ego near its route
            # is the reference-line d-bound (ROADMAP #17: local deviation must
            # stay within a safe corridor).
            for si, di in zip(traj['s'][::2], traj['d'][::2]):
                if abs(di) > corridor_bound:
                    return False
            for si, di in zip(traj['s'][::2], traj['d'][::2]):
                cx, cy, _ = converter.to_cartesian(si, di)
                col = int(round((cx - costmap.origin_x) / res))
                row = int(round((cy - costmap.origin_y) / res))
                if not (0 <= row < costmap.height and 0 <= col < costmap.width):
                    return False
                if float(costmap.data[row, col]) >= 0.9:
                    return False
            return True

        admissible = [tr for tr in trajectories if _in_corridor(tr)]
        if admissible:
            trajectories = admissible
        else:
            # EVERY candidate was rejected. Two distinct situations:
            #  - Ego already off-corridor (e.g. wedged outside after a
            #    degraded episode): a strict filter would deadlock it — allow
            #    recovery candidates that END inside the corridor.
            #  - Ego on-corridor but the crossing actor blocks it entirely:
            #    the correct answer is a full stop-and-wait, not an off-road
            #    creep. Any all-stop candidate is acceptable there; fall back
            #    to the unfiltered set only as a last resort.
            def _recovers(traj) -> bool:
                # Recovery = ends inside the corridor AND at-or-inward of the
                # ego's current offset. The absolute bound alone still permits
                # dwelling at |d| = bound; requiring the candidate to end no
                # further out than the ego already is forces convergence back
                # toward the reference line instead of an outward ratchet.
                return (abs(traj['d'][-1]) <= corridor_bound
                        and abs(traj['d'][-1]) <= abs(ego_d) + 0.05)

            def _is_allstop(traj) -> bool:
                # All-stop candidates MUST also respect the corridor bound:
                # the frenet parametrization can translate laterally even at
                # zero longitudinal speed, so an unbounded "stop" candidate
                # still walks the ego sideways off the road, tick by tick.
                return (max(traj['v'][:6]) <= 0.05
                        and all(abs(di) <= corridor_bound
                                for di in traj['d'][::2]))

            # Wait OUT the conflict instead of forcing an off-road creep:
            # brake along the reference line and re-evaluate next tick. A
            # slow-forward candidate is NOT acceptable here — the safety
            # monitor then vetoes it near the crossing actor (predicted
            # overlap) and the pipeline escalates to a true emergency stop
            # (observed: 4-8 single-tick estops per city_roads run).
            recovering = [tr for tr in trajectories if _recovers(tr)]
            stopping = [tr for tr in trajectories if _is_allstop(tr)]
            if recovering:
                trajectories = recovering
            elif stopping:
                trajectories = stopping
            else:
                return self._strong_brake(state, reason="Corridor blocked: stop and wait")

        best_frenet = self.planner.evaluate_trajectories(
            trajectories=trajectories,
            obstacles=frenet_obstacles,
            road_anomalies=frenet_anomalies,
            target_speed=target_v,
        )

        if best_frenet is None and vx <= 0.5:
            # Escape retry (blocked-road deadlock escape), ONLY when the ego
            # is (nearly) stopped: a stopped ego wedged inside a fattened veto
            # box has no candidate that starts outside it, so the veto would
            # deadlock it forever. A MOVING ego with no candidates is a
            # transient yield — strong brake is the correct response there,
            # not a crawl-past attempt.
            escape_trajectories = self.planner.generate_frenet_trajectories(
                c_speed=vx,
                c_d=ego_d,
                c_d_d=ego_d_dot,
                c_d_dd=ego_d_dd,
                s_0=ego_s,
                target_speed=0.8,
                d_range=(-2.6, 2.6, 0.2),
            )
            escape_trajectories = [
                tr for tr in escape_trajectories
                if abs(tr['d'][-1]) <= corridor_bound
            ] or escape_trajectories
            best_frenet = self.planner.evaluate_trajectories(
                trajectories=escape_trajectories,
                obstacles=frenet_obstacles,
                road_anomalies=frenet_anomalies,
                target_speed=0.8,
                # Allow escape from inside the fattened veto box (the ego is
                # already stopped there); the safety monitor remains the hard
                # gate with true OBB geometry.
                escape_allowance_s=0.4,
            )

        if best_frenet is None:
            # No candidate survived (transient yield under time pressure).
            # Return a controlled strong-brake ramp instead of an emergency
            # fallback: the safety monitor space-time-validates it like any
            # other trajectory and the controller executes a normal (harsh)
            # stop. If the ramp itself is genuinely unsafe, the monitor still
            # escalates to a true emergency stop.
            return self._strong_brake(state, reason="Yield: no viable candidate, strong brake")

        # Convert Frenet trajectory to Cartesian coordinates
        t_arr = best_frenet['t']
        s_arr = best_frenet['s']
        d_arr = best_frenet['d']
        v_arr = best_frenet['v']
        a_arr = best_frenet['a']

        raw_pts = []
        for si, di in zip(s_arr, d_arr):
            cx, cy, _ = converter.to_cartesian(si, di)
            raw_pts.append((cx, cy))

        max_curv = math.tan(self.vcfg.max_steer_angle) / max(self.vcfg.wheelbase, 1e-4)

        points: List[TrajectoryPoint] = []
        for i in range(len(raw_pts)):
            ti = float(t_arr[i])
            vi = float(v_arr[i])
            ai = float(a_arr[i])
            cx, cy = raw_pts[i]

            if i < len(raw_pts) - 1:
                dx = raw_pts[i+1][0] - cx
                dy = raw_pts[i+1][1] - cy
                heading = math.atan2(dy, dx)
            elif i > 0:
                heading = points[-1].pose.heading
            else:
                heading = state.pose.heading

            if i > 0:
                prev_h = points[-1].pose.heading
                dth = normalize_angle(heading - prev_h)
                prev_ds = math.hypot(cx - raw_pts[i-1][0], cy - raw_pts[i-1][1])
                curv = dth / prev_ds if prev_ds > 1e-3 else 0.0
                curv = max(-max_curv, min(max_curv, curv))
            else:
                curv = 0.0

            yaw_rate = curv * vi

            points.append(TrajectoryPoint(
                t=ti,
                pose=Pose2D(x=float(cx), y=float(cy), heading=float(heading)),
                twist=Twist2D(vx=vi, vy=0.0, yaw_rate=float(yaw_rate)),
                curvature=float(curv),
                acceleration=float(max(self.vcfg.min_acceleration, min(self.vcfg.max_acceleration, ai))),
            ))

        return LocalTrajectory(
            header=Header(state.header.stamp, FrameId.MAP, "adaptive_frenet_planner"),
            points=points,
            cost=0.0,
            is_safe=True,
            fallback_active=False,
            reason=f"Frenet state: {sm_state.value}",
        )

    @staticmethod
    def _extend_reference(ref: np.ndarray, extra_length: float = 10.0) -> np.ndarray:
        """Continue the polyline past its end along the final segment heading."""
        if len(ref) < 2 or extra_length <= 0.0:
            return ref
        last, prev = ref[-1], ref[-2]
        hx, hy = last[0] - prev[0], last[1] - prev[1]
        h = math.hypot(hx, hy)
        if h < 1e-6:
            return ref
        ux, uy = hx / h, hy / h
        n_extra = max(1, int(round(extra_length / h)))
        extension = np.array([[last[0] + ux * h * i, last[1] + uy * h * i]
                              for i in range(1, n_extra + 1)])
        return np.vstack([ref, extension])

    @staticmethod
    def _extend_reference_backward(ref: np.ndarray, extra_length: float = 10.0) -> np.ndarray:
        """Extend the polyline backward along the first segment heading.

        Mirrors _extend_reference for the path-start side: an ego trailing
        the retained path start (after a failed replan, or early in a run
        before the first path arrives) must still get a well-defined Frenet
        frame instead of a clamped s=0 degenerate one.
        """
        if len(ref) < 2 or extra_length <= 0.0:
            return ref
        first, second = ref[0], ref[1]
        hx, hy = first[0] - second[0], first[1] - second[1]
        h = math.hypot(hx, hy)
        if h < 1e-6:
            return ref
        ux, uy = hx / h, hy / h
        n_extra = max(1, int(round(extra_length / h)))
        extension = np.array([[first[0] + ux * h * i, first[1] + uy * h * i]
                              for i in range(n_extra, 0, -1)])
        return np.vstack([extension, ref])

    @staticmethod
    def _smooth_reference(ref: np.ndarray, passes: int = 3, window: int = 2) -> np.ndarray:
        """Centered moving-average smoothing of a polyline with endpoint clamping."""
        sm = ref.copy()
        n = len(sm)
        for _ in range(passes):
            xs, ys = sm[:, 0].copy(), sm[:, 1].copy()
            for i in range(n):
                lo, hi = max(0, i - window), min(n - 1, i + window)
                sm[i, 0] = float(xs[lo:hi + 1].mean())
                sm[i, 1] = float(ys[lo:hi + 1].mean())
        return sm

    def _arrival_stop(self, state: VehicleState, goal: Pose2D) -> LocalTrajectory:
        """Brake from the current speed to a stop aimed at the goal.

        A frozen pose profile relies on the controller's P-term for braking and
        overshoots; this bakes the deceleration ramp into the trajectory.
        """
        dt = 0.1
        v0 = max(state.twist.vx, 0.0)
        a_brake = min(2.0, abs(self.vcfg.min_acceleration))
        n = int(math.ceil((v0 / a_brake + 0.5) / dt))
        dx, dy = goal.x - state.pose.x, goal.y - state.pose.y
        L = math.hypot(dx, dy)
        ux, uy = (dx / L, dy / L) if L > 1e-6 else (math.cos(state.pose.heading), math.sin(state.pose.heading))
        points = []
        rem = L
        for i in range(n + 1):
            v = max(v0 - a_brake * i * dt, 0.0)
            step = min(v * dt, max(rem, 0.0))
            px = state.pose.x + (L - rem) * ux + step * ux
            py = state.pose.y + (L - rem) * uy + step * uy
            rem -= step
            points.append(TrajectoryPoint(
                t=i * dt,
                pose=Pose2D(x=px, y=py, heading=math.atan2(uy, ux)),
                twist=Twist2D(vx=v, vy=0.0, yaw_rate=0.0),
                curvature=0.0,
                acceleration=-a_brake if v > 1e-3 else 0.0,
            ))
        return LocalTrajectory(
            header=Header(state.header.stamp, FrameId.MAP, "adaptive_frenet_planner"),
            points=points,
            cost=0.0,
            is_safe=True,
            fallback_active=False,
            reason="Arrived at goal",
        )

    def _strong_brake(self, state: VehicleState, reason: str = "strong brake") -> LocalTrajectory:
        """Controlled strong-brake ramp along the current heading.

        Unlike `emergency_stop`, this is NOT flagged fallback_active, so the
        safety monitor evaluates it against predicted obstacle motion and the
        controller runs it through the normal path (no emergency-stop mode
        latch). Bakes a comfortable-envelope deceleration so the controller
        cannot overrun the stop.
        """
        dt = 0.1
        v0 = max(state.twist.vx, 0.0)
        a_brake = 2.0  # strong but within the comfort envelope
        n = int(math.ceil(v0 / a_brake / dt)) + 2
        x, y, th = state.pose.x, state.pose.y, state.pose.heading
        ux, uy = math.cos(th), math.sin(th)
        points: List[TrajectoryPoint] = []
        s = 0.0
        for i in range(n + 1):
            v = max(v0 - a_brake * i * dt, 0.0)
            points.append(TrajectoryPoint(
                t=i * dt,
                pose=Pose2D(x=x + s * ux, y=y + s * uy, heading=th),
                twist=Twist2D(vx=v, vy=0.0, yaw_rate=0.0),
                curvature=0.0,
                acceleration=-a_brake if v > 1e-3 else 0.0,
            ))
            s += v * dt
        return LocalTrajectory(
            header=Header(state.header.stamp, FrameId.MAP, "adaptive_frenet_planner"),
            points=points,
            cost=0.0,
            is_safe=True,
            fallback_active=False,
            reason=reason,
        )

    def emergency_stop(self, state: VehicleState, reason: str = "emergency stop") -> LocalTrajectory:
        dt, n = 0.1, 30
        speed = max(state.twist.vx, 0.0)
        x, y, th = state.pose.x, state.pose.y, state.pose.heading
        points = []
        for i in range(n + 1):
            v = max(speed, 0.0)
            points.append(TrajectoryPoint(
                t=i * dt, pose=Pose2D(x, y, th), twist=Twist2D(vx=v),
                curvature=0.0, acceleration=-3.0 if v > 1e-3 else 0.0,
            ))
            x += v * math.cos(th) * dt
            y += v * math.sin(th) * dt
            speed -= 3.0 * dt
        return LocalTrajectory(
            header=Header(state.header.stamp, FrameId.MAP, "adaptive_frenet_planner"),
            points=points, is_safe=True, fallback_active=True, reason=reason,
        )
