from src.common.coordinates.transforms import TransformTree
from src.common.types.base import Covariance2D, FrameId, Header, Pose2D, Twist2D
from src.common.types.config import CostmapConfig, DWAConfig, PlannerConfig, VehicleConfig
from src.common.types.control import ControlCommand
from src.common.types.obstacle import Obstacle, ObstacleBehavior, ObstacleClass
from src.common.types.path import GlobalPath
from src.common.types.sensor import SensorFrame
from src.common.types.vehicle_state import VehicleState
from src.controller.controllers.pure_pursuit import PurePursuitController
from src.integration.message_bus import Topic, TypedMessageBus
from src.integration.pipeline import IntegrationPipeline
from src.integration.safety_monitor import EnvelopeSafetyMonitor
from src.integration.scenario_runner import ScenarioRunner
from src.local_planner.local_planner import DWALocalPlanner
from src.mapping.costmap import LocalGridCostmapBuilder
from src.perception.ground_truth_perception import GroundTruthPerception
from src.sim.python_sim import PythonSimulator


def main():
    print("=== Sprint 2: A* + DWA + Pure Pursuit End-to-End ===")

    v_cfg = VehicleConfig(max_speed=8.0, max_steer_angle=0.6, wheelbase=2.5)
    c_cfg = CostmapConfig(width_m=100, height_m=100, resolution=0.5, inflation_radius=2.0)
    dwa_cfg = DWAConfig()

    sim = PythonSimulator(v_cfg)

    # World-fixed obstacle directly in the driving lane
    dummy_obs = Obstacle(
        header=Header(0.0, FrameId.MAP, "sim"), track_id=99,
        class_label=ObstacleClass.ANIMAL, behavior=ObstacleBehavior.STATIC,
        pose=Pose2D(12.0, 1.0, 0.0), length=2.0, width=2.0,
        velocity=Twist2D(), pose_covariance=Covariance2D(), velocity_covariance=Covariance2D(),
        confidence=1.0, is_dynamic=False,
    )
    sim.set_obstacles([dummy_obs])

    tf = TransformTree()
    bus = TypedMessageBus()

    latest_global_path = {}
    bus.subscribe(Topic.GLOBAL_PATH, lambda p: latest_global_path.update(path=p))

    pipeline = IntegrationPipeline(
        perception=GroundTruthPerception(lambda: sim.obstacles),
        costmap_builder=LocalGridCostmapBuilder(c_cfg),
        global_planner=__import__("src.global_planner.planner", fromlist=["AStarGlobalPlanner"]).AStarGlobalPlanner(target_speed=v_cfg.max_speed),
        local_planner=DWALocalPlanner(v_cfg, dwa_cfg),
        controller=PurePursuitController(v_cfg),
        safety_monitor=EnvelopeSafetyMonitor(v_cfg),
        transform_tree=tf,
        message_bus=bus,
    )

    runner = ScenarioRunner(pipeline, dt=0.1)

    start_state = VehicleState(
        header=Header(0.0, FrameId.MAP, "sim"),
        pose=Pose2D(0.0, 0.0, 0.0),
        twist=Twist2D(5.0, 0.0, 0.0),
        steering_angle=0.0,
    )
    goal = Pose2D(25.0, 5.0, 0.0)

    metrics = runner.run(
        initial_state=start_state,
        goal=goal,
        sensor_provider=lambda t: SensorFrame(header=Header(t, FrameId.SENSOR_FRONT, "sim")),
        state_updater=lambda s, c, dt: sim.step(s, c, dt),
        duration_s=12.0,
    )

    sim.plot_run(goal.x, goal.y, save_path="sprint2_dwa_demo.png",
                 global_path=latest_global_path.get("path"))
    print("=== Simulation Complete ===")


if __name__ == "__main__":
    main()