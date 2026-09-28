from typing import Optional, List, Dict, Any
import numpy as np

from src.common.coordinates.transforms import TransformTree
from src.common.types.base import FrameId, Header
from src.common.types.sensor import SensorFrame
from src.controller.controllers.pure_pursuit import PurePursuitController
from src.global_planner.planner import AStarGlobalPlanner
from src.integration.message_bus import Topic, TypedMessageBus
from src.integration.pipeline import IntegrationPipeline
from src.integration.safety_monitor import EnvelopeSafetyMonitor
from src.integration.scenario_runner import ScenarioRunner
from src.local_planner.local_planner import DWALocalPlanner
from src.mapping.costmap import LocalGridCostmapBuilder
from src.perception.audio_pipeline import (AcousticEvent,
                                           AcousticPerceptionNode)
from src.sim.python_sim import PythonSimulator
from src.sim.scenarios.base import RunLog, Scenario, ScenarioResult

# Advanced Metrics Imports
from src.sim.metrics import (calculate_path_efficiency, calculate_avg_jerk,
                             calculate_min_ttc, calculate_min_ttc_overall,
                             ttc_gate_failure)
from src.sim.perception_metrics import (compute_detection_metrics,
                                         compute_acoustic_metrics,
                                         aggregate_perception_metrics,
                                         INDIAN_CLASS_NAMES,
                                         DetectionMetrics, AcousticMetrics)


class ScenarioExecutor:
    def __init__(self, scenario: Scenario, live: bool = False,
                 compute_perception_metrics: bool = True):
        self.scenario = scenario
        self.live = live
        self.compute_perception_metrics = compute_perception_metrics

    def run(self, save_plot: Optional[str] = None,
            on_bus=None) -> ScenarioResult:
        sc = self.scenario
        v_cfg, c_cfg, dwa_cfg = sc.configs()

        sim = PythonSimulator(v_cfg)
        clock = {"t": 0.0}

        tf = TransformTree()
        bus = TypedMessageBus()
        # Optional observer hook: lets recorders/dashboard replay tools
        # subscribe to telemetry before the pipeline starts publishing.
        if on_bus is not None:
            on_bus(bus)

        latest_path = {}
        bus.subscribe(Topic.GLOBAL_PATH, lambda p: latest_path.update({"path": p}))

        # Perception metrics collection
        gt_detections_per_frame: List[List[Dict]] = []
        pred_detections_per_frame: List[List[Dict]] = []
        gt_acoustic_per_frame: List[List[Dict]] = []
        pred_acoustic_per_frame: List[List[Dict]] = []

        # Use the scenario's perception module (supports ground truth, noisy,
        # or — when the scenario opts in via `use_vision_perception` — the
        # synthetic camera -> vision pipeline path with GT passthrough for
        # non-visible obstacles).
        provider = lambda: sc.obstacles_at(clock["t"])
        if getattr(sc, "use_vision_perception", False):
            import math as _math
            import numpy as _np
            from src.perception.vision_node import VisionPerceptionNode

            # 640x480 pinhole, ~90 deg horizontal FOV, camera co-located with
            # base_link looking forward (T = 0).
            K = _np.array([[320.0, 0.0, 320.0],
                           [0.0, 320.0, 240.0],
                           [0.0, 0.0, 1.0]])
            R_c2v = _np.array([[0.0, 0.0, 1.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0]])
            RT = _np.column_stack([R_c2v, _np.zeros(3)])
            perception_module = VisionPerceptionNode(
                obstacles_provider=provider,
                camera_intrinsic=K,
                camera_extrinsics_rt=RT,
                transform_tree=tf,
                fov_rad=_math.atan2(K[0, 2], K[0, 0]),
                # Optional sensor-grade noise (scenario opt-in, e.g.
                # sensor_noise): injected into vision-path reconstructions.
                pos_noise_std=getattr(sc, "vision_pos_noise_std", 0.0),
                vel_noise_std=getattr(sc, "vision_vel_noise_std", 0.0),
                use_cv_detectors=getattr(sc, "use_cv_detectors", False),
                cv_detector_config=getattr(sc, "cv_detector_config", None),
                weather_condition=getattr(sc, "weather_condition", None),
                time_of_day=getattr(sc, "time_of_day", None),
            )
        else:
            perception_module = sc.get_perception_module(provider)

        pipeline = IntegrationPipeline(
            perception=perception_module,
            costmap_builder=LocalGridCostmapBuilder(
                c_cfg,
                road_network=sc.road_network() if hasattr(sc, "road_network") else None,
            ),
            global_planner=AStarGlobalPlanner(target_speed=v_cfg.max_speed),
            local_planner=(
                sc.get_local_planner(v_cfg, dwa_cfg)
                if hasattr(sc, "get_local_planner")
                else DWALocalPlanner(v_cfg, dwa_cfg)
            ),
            controller=PurePursuitController(v_cfg),
            safety_monitor=EnvelopeSafetyMonitor(v_cfg),
            transform_tree=tf,
            message_bus=bus,
            # ROADMAP #27b: opt-in acoustic attention via a scenario-level
            # `acoustic_events()` provider (synthetic event injection — no
            # real audio hardware needed).
            acoustic_node=(
                AcousticPerceptionNode(sc.acoustic_events())
                if hasattr(sc, "acoustic_events") else None
            ),
        )

        # Setup Dashboard if live=True
        dashboard = None
        if self.live:
            from src.dashboard.live_dashboard import LiveDashboard
            dashboard = LiveDashboard(bus)

        def on_tick(t, state, cmd):
            if dashboard:
                dashboard.tick(state)

        # Pass the on_tick callback to the ScenarioRunner
        runner = ScenarioRunner(pipeline, dt=0.1, on_tick=on_tick)

        def get_sensor(t: float) -> SensorFrame:
            clock["t"] = t  # perception reads the same sim clock
            return SensorFrame(header=Header(t, FrameId.SENSOR_FRONT, "sim"))

        # Custom runner that collects perception data
        if self.compute_perception_metrics:
            metrics = self._run_with_metrics(
                runner, get_sensor, sc, clock, sim, v_cfg,
                gt_detections_per_frame, pred_detections_per_frame,
                gt_acoustic_per_frame, pred_acoustic_per_frame
            )
        else:
            metrics = runner.run(
                initial_state=sc.initial_state(),
                goal=sc.goal(),
                sensor_provider=get_sensor,
                state_updater=lambda s, c, dt: sim.step(s, c, dt),
                duration_s=sc.duration_s,
            )

        log = RunLog(
            times=[e[0] for e in runner.log],
            states=[e[1] for e in runner.log],
            commands=[e[2] for e in runner.log],
            global_path=latest_path.get("path"),
            emergency_stops=metrics.emergency_stops,
        )
        self.last_run_log = log  # retained for post-run metric analysis

        result = sc.evaluate(log, v_cfg)

        # --- Advanced Metrics ---
        path_eff = calculate_path_efficiency(log)
        avg_jerk = calculate_avg_jerk(log)
        min_ttc = calculate_min_ttc(log, sc.obstacles_at)
        
        result.metrics["path_efficiency"] = path_eff
        result.metrics["avg_jerk_mps3"] = avg_jerk
        if min_ttc > 0:
            result.metrics["min_ttc_s"] = min_ttc
        # Unfiltered classic TTC minimum, for context alongside the
        # response-aware min_ttc_s (which excludes samples during a correct
        # braking response).
        min_ttc_all = calculate_min_ttc_overall(log, sc.obstacles_at)
        if min_ttc_all > 0:
            result.metrics["min_ttc_overall_s"] = min_ttc_all

        # Response-aware TTC gate: a low unresponded min TTC means the system
        # failed to react to a closing threat. Opt-in per scenario via the
        # ``min_ttc_gate_s`` attribute (None/absent disables the gate).
        gate_threshold = getattr(sc, "min_ttc_gate_s", None)
        gate_failure = ttc_gate_failure(min_ttc, gate_threshold)
        if gate_failure:
            result.failures.append(gate_failure)
            result.passed = False

        # --- Perception Metrics ---
        if self.compute_perception_metrics:
            self._compute_and_add_perception_metrics(
                result, gt_detections_per_frame, pred_detections_per_frame,
                gt_acoustic_per_frame, pred_acoustic_per_frame
            )

        print(f"[Scenario:{result.name}] {'PASS' if result.passed else 'FAIL'}")
        for k, v in result.metrics.items():
            if "efficiency" in k:
                print(f"    {k}: {v:.2%}")
            else:
                print(f"    {k}: {v:.2f}")
        for f in result.failures:
            print(f"    FAILURE: {f}")

        if save_plot:
            sim.plot_run(sc.goal().x, sc.goal().y, save_path=save_plot,
                         global_path=log.global_path, tracks=sc.plot_tracks())

        return result

    def _run_with_metrics(self, runner, get_sensor, sc, clock, sim, v_cfg,
                          gt_detections_per_frame, pred_detections_per_frame,
                          gt_acoustic_per_frame, pred_acoustic_per_frame):
        """Run scenario while collecting perception ground truth and predictions."""
        # We need to manually step through to collect perception data
        initial_state = sc.initial_state()
        goal = sc.goal()
        duration_s = sc.duration_s
        dt = 0.1
        
        state = initial_state
        t = 0.0
        
        emergency_stops = 0
        
        while t < duration_s:
            sensor = get_sensor(t)
            
            # Update transform tree with current state BEFORE perception
            runner.pipeline.tf.update_ego_state(state)
            
            # Get GT obstacles
            gt_obstacles = sc.obstacles_at(t)
            
            # Process perception
            perception_out = runner.pipeline.perception.process(sensor)
            
            # Convert GT to detection format (for metrics)
            gt_dets = self._obstacles_to_detections(gt_obstacles, t)
            gt_detections_per_frame.append(gt_dets)
            
            # Convert perception output to detection format
            pred_dets = self._perception_to_detections(perception_out, t)
            pred_detections_per_frame.append(pred_dets)
            
            # Acoustic GT
            if hasattr(sc, 'acoustic_events'):
                gt_acoustic = []
                for ev in sc.acoustic_events():
                    if ev.t_onset <= t <= ev.t_end:
                        gt_acoustic.append({
                            't': t,
                            'acoustic_class': ev.acoustic_class,
                            'azimuth_rad': ev.azimuth_rad,
                        })
                gt_acoustic_per_frame.append(gt_acoustic)
            
            # Acoustic predictions
            if runner.pipeline.acoustic_node is not None:
                cues = runner.pipeline.acoustic_node.process(sensor)
                pred_acoustic = []
                for cue in cues:
                    pred_acoustic.append({
                        't': cue.t,
                        'acoustic_class': cue.acoustic_class,
                        'azimuth_rad': cue.azimuth_rad,
                    })
                pred_acoustic_per_frame.append(pred_acoustic)
            else:
                pred_acoustic_per_frame.append([])
            
            # Run pipeline tick
            cmd = runner.pipeline.tick(t, state, sensor, goal)
            
            # Check for emergency stop
            if cmd.mode.name == "EMERGENCY_STOP":
                emergency_stops += 1
            
            # Step simulation
            state = sim.step(state, cmd, dt)
            
            # Log
            runner.log.append((t, state, cmd))
            
            t += dt
        
        # Return metrics object compatible with existing code
        class SimpleMetrics:
            def __init__(self, emergency_stops):
                self.emergency_stops = emergency_stops
        
        return SimpleMetrics(emergency_stops)

    def _obstacles_to_detections(self, obstacles, timestamp: float) -> List[Dict]:
        """Convert ground truth obstacles to detection format."""
        detections = []
        for obs in obstacles:
            # Project to camera frame (simplified - use map coordinates)
            # In reality, would project through camera model
            detections.append({
                'class_id': self._obstacle_class_to_id(obs.class_label),
                'x1': obs.pose.x - obs.length/2,
                'y1': obs.pose.y - obs.width/2,
                'x2': obs.pose.x + obs.length/2,
                'y2': obs.pose.y + obs.width/2,
                'confidence': 1.0,
                'track_id': obs.track_id,
            })
        return detections

    def _perception_to_detections(self, perception_out, timestamp: float) -> List[Dict]:
        """Convert perception output to detection format."""
        detections = []
        for obs in perception_out.obstacles:
            detections.append({
                'class_id': self._obstacle_class_to_id(obs.class_label),
                'x1': obs.pose.x - obs.length/2,
                'y1': obs.pose.y - obs.width/2,
                'x2': obs.pose.x + obs.length/2,
                'y2': obs.pose.y + obs.width/2,
                'confidence': obs.confidence,
                'track_id': obs.track_id,
            })
        return detections

    def _obstacle_class_to_id(self, obs_class) -> int:
        """Map ObstacleClass to Indian traffic class ID."""
        mapping = {
            'vehicle': 1,
            'pedestrian': 8,
            'animal': 9,
            'pothole': 11,
            'encroachment': 13,
            'parked_vehicle': 1,
            'two_wheeler': 3,
        }
        return mapping.get(obs_class.value if hasattr(obs_class, 'value') else str(obs_class), 1)

    def _compute_and_add_perception_metrics(self, result, gt_dets_per_frame,
                                             pred_dets_per_frame,
                                             gt_acoustic_per_frame,
                                             pred_acoustic_per_frame):
        """Compute perception metrics and add to result."""
        # Flatten all frames
        all_gt_dets = []
        all_pred_dets = []
        for i, (gt_frame, pred_frame) in enumerate(zip(gt_dets_per_frame, pred_dets_per_frame)):
            for gt in gt_frame:
                gt['frame'] = i
                all_gt_dets.append(gt)
            for pred in pred_frame:
                pred['frame'] = i
                all_pred_dets.append(pred)
        
        # Compute detection metrics
        det_metrics = compute_detection_metrics(
            all_gt_dets, all_pred_dets,
            iou_threshold=0.5,
            class_names=INDIAN_CLASS_NAMES
        )
        
        # Add to result metrics
        result.metrics["detection_map_50"] = det_metrics.map_50
        result.metrics["detection_precision"] = det_metrics.precision
        result.metrics["detection_recall"] = det_metrics.recall
        result.metrics["detection_f1"] = det_metrics.f1
        
        # Per-class AP
        for cls, ap in det_metrics.ap_per_class.items():
            result.metrics[f"ap_{cls}"] = ap
        
        # Acoustic metrics
        if gt_acoustic_per_frame and any(gt_acoustic_per_frame):
            all_gt_acoustic = []
            all_pred_acoustic = []
            for i, (gt_frame, pred_frame) in enumerate(zip(gt_acoustic_per_frame, pred_acoustic_per_frame)):
                for gt in gt_frame:
                    all_gt_acoustic.append(gt)
                for pred in pred_frame:
                    all_pred_acoustic.append(pred)
            
            ac_metrics = compute_acoustic_metrics(all_gt_acoustic, all_pred_acoustic)
            
            result.metrics["acoustic_doa_mae_deg"] = ac_metrics.doa_mae_deg
            result.metrics["acoustic_doa_rmse_deg"] = ac_metrics.doa_rmse_deg
            result.metrics["acoustic_classification_acc"] = ac_metrics.classification_accuracy
            result.metrics["acoustic_detection_precision"] = ac_metrics.detection_precision
            result.metrics["acoustic_detection_recall"] = ac_metrics.detection_recall