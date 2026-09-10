import csv
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.common.types.control import ControlCommand
from src.common.types.obstacle import Obstacle
from src.common.types.vehicle_state import VehicleState
from src.controller.models.bicycle_model import KinematicBicycleModel
from src.common.types.config import VehicleConfig


class PythonSimulator:
    def __init__(self, cfg: VehicleConfig):
        self.model = KinematicBicycleModel(cfg)
        self.history: List[Tuple[float, float, float]] = [] # (x, y, heading)
        self.obstacles: List[Obstacle] = []

    def step(self, state: VehicleState, cmd: ControlCommand, dt: float) -> VehicleState:
        next_state = self.model.step(state, cmd, dt)
        self.history.append((next_state.pose.x, next_state.pose.y, next_state.pose.heading))
        return next_state

    def set_obstacles(self, obstacles: List[Obstacle]):
        self.obstacles = obstacles

    def plot_run(self, goal_x: float, goal_y: float, save_path: str = "sim_output.png"):
        if not self.history:
            print("No history to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot vehicle path
        path = np.array(self.history)
        ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label="Vehicle Path")
        ax.plot(path[0, 0], path[0, 1], 'go', markersize=10, label="Start")
        ax.plot(goal_x, goal_y, 'r*', markersize=15, label="Goal")

        # Plot obstacles
        for obs in self.obstacles:
            circle = plt.Circle((obs.pose.x, obs.pose.y), max(obs.length, obs.width)/2, 
                                color='red', alpha=0.5, label="Obstacle" if obs == self.obstacles[0] else "")
            ax.add_patch(circle)

        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        ax.set_title("Sprint 1 End-to-End Simulation Run")
        
        plt.savefig(save_path)
        print(f"[Simulator] Saved trajectory plot to {os.path.abspath(save_path)}")