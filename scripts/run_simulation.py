import math

from src.common.types.base import FrameId, Header, Pose2D, Twist2D, Covariance2D
from src.common.types.config import CostmapConfig, PlannerConfig, VehicleConfig
from src.common.types.control import ControlCommand
from src.common.types.obstacle import Obstacle, ObstacleBehavior, ObstacleClass
from src.common.types.sensor import SensorFrame
from src.common.types.trajectory import LocalTrajectory
from src.common.types.vehicle_state import VehicleState
from src.integration.mocks import MockController, MockGlobalPlanner, MockPerception
from src.integration.pipeline import IntegrationPipeline
from src.integration.safety_monitor import EnvelopeSafetyMonitor
from src.integration.scenario_runner import ScenarioRunner
from src.integration.system_runner import build_pipeline
from src.mapping.costmap import LocalGridCostmapBuilder
from src.sim.python_sim import PythonSimulator
from src.common.coordinates.transforms import TransformTree
from src.integration.message_bus import TypedMessageBus


class GoToGoalLocalPlanner:
    """
    Sprint 1 Dummy Local Planner.
    Just points the car at the goal to prove the pipeline moves.
    M4 will replace this with DWA/MPC in Sprint 2.
    """
    def plan(self, state, global_path, costmap, obstacles):
        goal = global_path.points[-1].pose
        dx = goal.x - state.pose.x
        dy = goal.y - state.pose.y
        target_heading = math.atan2(dy, dx)
        
        # Simple P-controller for steering
        heading_error = target_heading - state.pose.heading
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        
        steer = max(-0.5, min(0.5, heading_error * 1.5))
        
        from src.common.types.trajectory import TrajectoryPoint
        pt = TrajectoryPoint(
            t=0.1, pose=state.pose, twist=Twist2D(vx=5.0), curvature=0.0, acceleration=0.5
        )
        return LocalTrajectory(
            header=Header(state.header.stamp, FrameId.MAP, "goto_goal"),
            points=[pt], is_safe=True
        )

    def emergency_stop(self, state):
        from src.integration.mocks import MockLocalPlanner
        return MockLocalPlanner(PlannerConfig(), VehicleConfig()).emergency_stop(state)


def main():
    print("=== Starting Sprint 1 End-to-End Simulation ===")
    
    # 1. Setup Configs
    v_cfg = VehicleConfig(max_speed=8.0, max_steer_angle=0.6, wheelbase=2.5)
    c_cfg = CostmapConfig(width_m=40, height_m=40, resolution=0.5, inflation_radius=1.5)
    p_cfg = PlannerConfig()
    
    # 2. Setup Pipeline
    tf = TransformTree()
    bus = TypedMessageBus()
    
    # We use mocks for Perception and Global Planner for now
    pipeline = IntegrationPipeline(
        perception=MockPerception(),
        costmap_builder=LocalGridCostmapBuilder(c_cfg),
        global_planner=MockGlobalPlanner(),
        local_planner=GoToGoalLocalPlanner(), # Inject our simple Go-To-Goal
        controller=MockController(),
        safety_monitor=EnvelopeSafetyMonitor(v_cfg),
        transform_tree=tf,
        message_bus=bus
    )
    
    # 3. Setup Simulator
    sim = PythonSimulator(v_cfg)
    runner = ScenarioRunner(pipeline, dt=0.1)
    
    # 4. Define Scenario
    start_state = VehicleState(
        header=Header(0.0, FrameId.MAP, "sim"),
        pose=Pose2D(0.0, 0.0, 0.0),
        twist=Twist2D(5.0, 0.0, 0.0),
        steering_angle=0.0
    )
    goal = Pose2D(25.0, 5.0, 0.0) # Goal is slightly to the left
    
    # Inject an obstacle directly into the simulator for plotting
    dummy_obs = Obstacle(
        header=Header(0.0, FrameId.MAP, "sim"), track_id=99,
        class_label=ObstacleClass.ANIMAL, behavior=ObstacleBehavior.STATIC,
        pose=Pose2D(12.0, 2.0, 0.0), length=2.0, width=2.0,
        velocity=Twist2D(), pose_covariance=Covariance2D(), velocity_covariance=Covariance2D(),
        confidence=1.0, is_dynamic=False
    )
    sim.set_obstacles([dummy_obs])

    # 5. Define Sensor & State Updater callbacks for the ScenarioRunner
    def get_sensor(t):
        # In a real sim, this would query CARLA/Gazebo. 
        # Here we just return a blank frame because MockPerception ignores it 
        # and always returns our hardcoded dummy obstacle.
        return SensorFrame(header=Header(t, FrameId.SENSOR_FRONT, "sim"))

    def update_physics(state: VehicleState, cmd: ControlCommand, dt: float):
        return sim.step(state, cmd, dt)

    # 6. Run!
    print("Running scenario for 10 seconds...")
    metrics = runner.run(
        initial_state=start_state,
        goal=goal,
        sensor_provider=get_sensor,
        state_updater=update_physics,
        duration_s=10.0
    )

    # 7. Visualize
    sim.plot_run(goal.x, goal.y, save_path="sprint1_demo.png")
    print("=== Simulation Complete ===")

if __name__ == "__main__":
    main()