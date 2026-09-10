import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.types.config import VehicleConfig
from src.common.types.control import ControlCommand
from src.common.types.obstacle import Obstacle
from src.common.types.path import GlobalPath
from src.common.types.vehicle_state import VehicleState
from src.controller.models.bicycle_model import KinematicBicycleModel


class PythonSimulator:
    def __init__(self, cfg: VehicleConfig):
        self.model = KinematicBicycleModel(cfg)
        self.history: List[Tuple[float, float, float]] = []
        self.obstacles: List[Obstacle] = []

    def step(self, state: VehicleState, cmd: ControlCommand, dt: float) -> VehicleState:
        next_state = self.model.step(state, cmd, dt)
        self.history.append((next_state.pose.x, next_state.pose.y, next_state.pose.heading))
        return next_state

    def set_obstacles(self, obstacles: List[Obstacle]):
        self.obstacles = obstacles

    def plot_run(self, goal_x, goal_y, save_path="sim_output.png", global_path: Optional[GlobalPath] = None):
        if not self.history:
            print("No history to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        path = np.array(self.history)
        ax.plot(path[:, 0], path[:, 1], "b-", linewidth=2, label="Vehicle Path")

        if global_path is not None and global_path.points:
            gp = np.array([[p.pose.x, p.pose.y] for p in global_path.points])
            ax.plot(gp[:, 0], gp[:, 1], "c--", linewidth=1.5, label="Global Path (A*)")

        ax.plot(path[0, 0], path[0, 1], "go", markersize=10, label="Start")
        ax.plot(goal_x, goal_y, "r*", markersize=15, label="Goal")

        for i, obs in enumerate(self.obstacles):
            ax.add_patch(plt.Circle((obs.pose.x, obs.pose.y), max(obs.length, obs.width) / 2,
                                    color="red", alpha=0.5,
                                    label="Obstacle" if i == 0 else None))

        ax.set_aspect("equal")
        ax.grid(True)
        ax.legend()
        ax.set_title("Sprint 2: A* + DWA + Pure Pursuit")
        plt.savefig(save_path)
        print(f"[Simulator] Saved trajectory plot to {os.path.abspath(save_path)}")