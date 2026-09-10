"""PathSense application entry point."""

import logging
from src.integration.message_bus import MessageBus
from src.common.types.vehicle_state import VehicleState
from src.common.types.obstacle import Obstacle
from src.common.types.path import GlobalPath
from src.common.types.trajectory import LocalTrajectory

logger = logging.getLogger(__name__)

class Pipeline:
    """
    The main orchestrator. M1 owns this. 
    It wires the modules together and drives the simulation loop.
    """
    def __init__(self, bus: MessageBus):
        self.bus = bus
        
        # Module placeholders (M2, M3, M4 will inject their classes here)
        self.perception = None
        self.mapper = None
        self.global_planner = None
        self.local_planner = None
        self.controller = None
        
        self._setup_wiring()

    def _setup_wiring(self):
        """Defines the data flow between modules."""
        # When perception finishes, it publishes obstacles. 
        # The mapper listens to this to update the costmap.
        self.bus.subscribe("perception/obstacles", self._process_mapping)
        
        # When mapper finishes, it publishes costmap.
        # Global planner listens to this.
        self.bus.subscribe("mapping/costmap", self._process_global_planning)
        
        # When global planner finishes, it publishes path.
        # Local planner listens to this.
        self.bus.subscribe("planning/global_path", self._process_local_planning)
        
        # When local planner finishes, it publishes trajectory.
        # Controller listens to this.
        self.bus.subscribe("planning/local_trajectory", self._process_control)

    def _process_mapping(self, obstacles: list[Obstacle]):
        logger.info("Pipeline: Triggering Mapping...")
        if self.mapper:
            costmap = self.mapper.update(obstacles)
            self.bus.publish("mapping/costmap", costmap)

    def _process_global_planning(self, costmap):
        logger.info("Pipeline: Triggering Global Planning...")
        if self.global_planner:
            path = self.global_planner.plan(costmap)
            self.bus.publish("planning/global_path", path)

    def _process_local_planning(self, path: GlobalPath):
        logger.info("Pipeline: Triggering Local Planning...")
        if self.local_planner:
            trajectory = self.local_planner.plan(path)
            self.bus.publish("planning/local_trajectory", trajectory)

    def _process_control(self, trajectory: LocalTrajectory):
        logger.info("Pipeline: Triggering Control...")
        if self.controller:
            cmd = self.controller.compute(trajectory)
            self.bus.publish("control/command", cmd)

    def step(self, current_state: VehicleState, sensor_data: dict):
        """
        Called by the Simulator (M5) at every time step.
        """
        logger.info(f"--- Pipeline Step @ t={current_state.timestamp:.2f}s ---")
        
        # 1. Perception
        if self.perception:
            obstacles = self.perception.detect(sensor_data)
            self.bus.publish("perception/obstacles", obstacles)
        else:
            logger.warning("Perception module not initialized!")