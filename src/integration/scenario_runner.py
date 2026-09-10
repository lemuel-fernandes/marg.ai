import time
from typing import Callable, List, Tuple

from src.common.types.base import Pose2D
from src.common.types.control import ControlCommand
from src.common.types.sensor import SensorFrame
from src.common.types.vehicle_state import VehicleState
from src.integration.pipeline import IntegrationPipeline


class ScenarioMetrics:
    def __init__(self):
        self.steps = 0
        self.replans = 0
        self.emergency_stops = 0
        self.latency_ms: List[float] = []

    def log_step(self, cmd: ControlCommand, latency: float):
        self.steps += 1
        self.latency_ms.append(latency)
        if cmd.mode.value == "emergency_stop":
            self.emergency_stops += 1

    def summary(self) -> str:
        avg_lat = sum(self.latency_ms) / len(self.latency_ms) if self.latency_ms else 0
        return (f"Steps: {self.steps} | "
                f"Estops: {self.emergency_stops} | "
                f"Avg Latency: {avg_lat:.2f}ms")


class ScenarioRunner:
    def __init__(self, pipeline: IntegrationPipeline, dt: float = 0.1):
        self.pipeline = pipeline
        self.dt = dt
        # (t, state_before_tick, command_issued) for post-run analysis
        self.log: List[Tuple[float, VehicleState, ControlCommand]] = []

    def run(
        self,
        initial_state: VehicleState,
        goal: Pose2D,
        sensor_provider: Callable[[float], SensorFrame],
        state_updater: Callable[[VehicleState, ControlCommand, float], VehicleState],
        duration_s: float = 10.0,
    ) -> ScenarioMetrics:
        metrics = ScenarioMetrics()
        current_state = initial_state
        current_time = 0.0

        print(f"[ScenarioRunner] Starting scenario for {duration_s}s...")

        while current_time <= duration_s:
            t_start = time.perf_counter()

            sensor_frame = sensor_provider(current_time)

            command = self.pipeline.tick(
                now=current_time,
                vehicle_state=current_state,
                sensor_frame=sensor_frame,
                goal=goal,
            )

            t_end = time.perf_counter()
            metrics.log_step(command, (t_end - t_start) * 1000)
            self.log.append((current_time, current_state, command))

            current_state = state_updater(current_state, command, self.dt)
            current_time += self.dt

        print(f"[ScenarioRunner] Finished. {metrics.summary()}")
        return metrics