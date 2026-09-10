import math

from src.common.coordinates.transforms import TransformTree
from src.common.types.base import Covariance2D, FrameId, Header, Pose2D, Twist2D
from src.common.types.config import CostmapConfig, DWAConfig, VehicleConfig
from src.common.types.control import ControlCommand
from src.common.types.obstacle import Obstacle, ObstacleBehavior, ObstacleClass
from src.common.types.sensor import SensorFrame
from src.common.types.vehicle_state import VehicleState
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


def main():
    print("=== Sprint 2: A* + DWA + Pure Pursuit End-to-End ===")

    # 1. Configs
    v_cfg = VehicleConfig(max_speed=8.0, max_steer_angle=0.6, wheelbase=2.5)
    c_cfg = CostmapConfig(width_m=100, height_m=100, resolution=0.5, inflation_radius=2.0)
    dwa_cfg = DWAConfig()

    # 2. Simulator + world-fixed obstacle (directly in the driving lane)
    sim = PythonSimulator(v_cfg)
    dummy_obs = Obstacle(
        header=Header(0.0, FrameId.MAP, "sim"), track_id=99,
        class_label=ObstacleClass.ANIMAL, behavior=ObstacleBehavior.STATIC,
        pose=Pose2D(12.0, 1.0, 0.0), length=2.0, width=2.0,
        velocity=Twist2D(), pose_covariance=Covariance2D(), velocity_covariance=Covariance2D(),
        confidence=1.0, is_dynamic=False,
    )
    sim.set_obstacles([dummy_obs])

    # 3. Integration stack
    tf = TransformTree()
    bus = TypedMessageBus()

    latest_global_path = {}
    bus.subscribe(Topic.GLOBAL_PATH, lambda p: latest_global_path.update(path=p))

    pipeline = IntegrationPipeline(
        perception=GroundTruthPerception(lambda: sim.obstacles),
        costmap_builder=LocalGridCostmapBuilder(c_cfg),
        global_planner=AStarGlobalPlanner(target_speed=v_cfg.max_speed),
        local_planner=DWALocalPlanner(v_cfg, dwa_cfg),
        controller=PurePursuitController(v_cfg),
        safety_monitor=EnvelopeSafetyMonitor(v_cfg),
        transform_tree=tf,
        message_bus=bus,
    )

    runner = ScenarioRunner(pipeline, dt=0.1)

    # 4. Scenario definition
    start_state = VehicleState(
        header=Header(0.0, FrameId.MAP, "sim"),
        pose=Pose2D(0.0, 0.0, 0.0),
        twist=Twist2D(5.0, 0.0, 0.0),
        steering_angle=0.0,
    )
    goal = Pose2D(25.0, 5.0, 0.0)

    # 5. Simulator callbacks
    stall_ticks = {"n": 0}

    def get_sensor(t: float) -> SensorFrame:
        return SensorFrame(header=Header(t, FrameId.SENSOR_FRONT, "sim"))

    def update_physics(state: VehicleState, cmd: ControlCommand, dt: float) -> VehicleState:
        next_state = sim.step(state, cmd, dt)

        # Watchdog: flag stopping ONLY if we are NOT at the goal
        dist_goal = math.hypot(goal.x - state.pose.x, goal.y - state.pose.y)
        if next_state.twist.vx < 0.2 and dist_goal > 3.0:
            stall_ticks["n"] += 1
            if stall_ticks["n"] == 20:  # 2 seconds stopped mid-run
                print(f"[WATCHDOG] Stall at t={state.header.stamp:.1f}s "
                      f"pos=({state.pose.x:.1f},{state.pose.y:.1f}) dist_goal={dist_goal:.1f}m")
        else:
            stall_ticks["n"] = 0

        return next_state

    # 6. Run scenario
    metrics = runner.run(
        initial_state=start_state,
        goal=goal,
        sensor_provider=get_sensor,
        state_updater=update_physics,
        duration_s=12.0,
    )

    # 7. Acceptance check + visualization
    final_x, final_y, _ = sim.history[-1] if sim.history else (0.0, 0.0, 0.0)
    dist = math.hypot(goal.x - final_x, goal.y - final_y)
    print(f"[Result] Final pos=({final_x:.2f}, {final_y:.2f}) | dist_to_goal={dist:.2f} m")

    if dist <= 1.5 and metrics.emergency_stops == 0:
        print("[Result] PASS: reached goal, stopped cleanly, zero emergency stops")
    else:
        print(f"[Result] FAIL: dist={dist:.2f} m, estops={metrics.emergency_stops}")

    sim.plot_run(
        goal.x, goal.y,
        save_path="sprint2_dwa_demo.png",
        global_path=latest_global_path.get("path"),
    )
    print("=== Simulation Complete ===")


if __name__ == "__main__":
    main()