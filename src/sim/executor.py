from typing import Optional

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


class ScenarioExecutor:
    def __init__(self, scenario: Scenario, live: bool = False):
        self.scenario = scenario
        self.live = live

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