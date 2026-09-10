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
from src.perception.ground_truth_perception import GroundTruthPerception
from src.sim.python_sim import PythonSimulator
from src.sim.scenarios.base import RunLog, Scenario, ScenarioResult
from src.local_planner.algorithms.dwa import DWA_CODE_VERSION
from src.local_planner.local_planner import LP_CODE_VERSION

class ScenarioExecutor:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario

    def run(self, save_plot: Optional[str] = None) -> ScenarioResult:
        sc = self.scenario
        v_cfg, c_cfg, dwa_cfg = sc.configs()

        sim = PythonSimulator(v_cfg)
        clock = {"t": 0.0}

        tf = TransformTree()
        bus = TypedMessageBus()
        
        # CHANGED: Capture the LATEST global path. 
        # This ensures we always have a valid path for deviation calculations,
        # even if the initial path was blocked and A* had to replan.
        latest_path = {}
        bus.subscribe(Topic.GLOBAL_PATH, lambda p: latest_path.update({"path": p}))

        pipeline = IntegrationPipeline(
            perception=GroundTruthPerception(lambda: sc.obstacles_at(clock["t"])),
            costmap_builder=LocalGridCostmapBuilder(c_cfg),
            global_planner=AStarGlobalPlanner(target_speed=v_cfg.max_speed),
            local_planner=DWALocalPlanner(v_cfg, dwa_cfg),
            controller=PurePursuitController(v_cfg),
            safety_monitor=EnvelopeSafetyMonitor(v_cfg),
            transform_tree=tf,
            message_bus=bus,
        )
        print(f"[CODE] executor built pipeline with dwa={DWA_CODE_VERSION} lp={LP_CODE_VERSION}")
        runner = ScenarioRunner(pipeline, dt=0.1)

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
            global_path=latest_path.get("path"),  # <--- CHANGED
            emergency_stops=metrics.emergency_stops,
        )

        result = sc.evaluate(log, v_cfg)

        print(f"[Scenario:{result.name}] {'PASS' if result.passed else 'FAIL'}")
        for k, v in result.metrics.items():
            print(f"    {k}: {v:.2f}")
        for f in result.failures:
            print(f"    FAILURE: {f}")

        if save_plot:
            sim.plot_run(sc.goal().x, sc.goal().y, save_path=save_plot,
                         global_path=log.global_path, tracks=sc.plot_tracks())

        return result