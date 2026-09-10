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
from src.sim.python_sim import PythonSimulator
from src.sim.scenarios.base import RunLog, Scenario, ScenarioResult

# Advanced Metrics Imports
from src.sim.metrics import calculate_path_efficiency, calculate_avg_jerk, calculate_min_ttc


class ScenarioExecutor:
    def __init__(self, scenario: Scenario, live: bool = False):
        self.scenario = scenario
        self.live = live

    def run(self, save_plot: Optional[str] = None) -> ScenarioResult:
        sc = self.scenario
        v_cfg, c_cfg, dwa_cfg = sc.configs()

        sim = PythonSimulator(v_cfg)
        clock = {"t": 0.0}

        tf = TransformTree()
        bus = TypedMessageBus()
        
        latest_path = {}
        bus.subscribe(Topic.GLOBAL_PATH, lambda p: latest_path.update({"path": p}))

        # Use the scenario's perception module (supports ground truth or noisy)
        perception_module = sc.get_perception_module(lambda: sc.obstacles_at(clock["t"]))

        pipeline = IntegrationPipeline(
            perception=perception_module,
            costmap_builder=LocalGridCostmapBuilder(c_cfg),
            global_planner=AStarGlobalPlanner(target_speed=v_cfg.max_speed),
            local_planner=DWALocalPlanner(v_cfg, dwa_cfg),
            controller=PurePursuitController(v_cfg),
            safety_monitor=EnvelopeSafetyMonitor(v_cfg),
            transform_tree=tf,
            message_bus=bus,
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

        result = sc.evaluate(log, v_cfg)

        # --- Advanced Metrics ---
        path_eff = calculate_path_efficiency(log)
        avg_jerk = calculate_avg_jerk(log)
        min_ttc = calculate_min_ttc(log, sc.obstacles_at)
        
        result.metrics["path_efficiency"] = path_eff
        result.metrics["avg_jerk_mps3"] = avg_jerk
        if min_ttc > 0:
            result.metrics["min_ttc_s"] = min_ttc

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