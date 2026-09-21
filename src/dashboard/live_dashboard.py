import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from src.common.types.costmap import Costmap
from src.common.types.path import GlobalPath
from src.common.types.trajectory import LocalTrajectory
from src.common.types.vehicle_state import VehicleState
from src.integration.message_bus import Topic, TypedMessageBus

class LiveDashboard:
    def __init__(self, bus: TypedMessageBus):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('PathSense Live Telemetry')
        
        self.state: VehicleState = None
        self.costmap: Costmap = None
        self.global_path: GlobalPath = None
        self.local_traj: LocalTrajectory = None
        self.obstacles = []
        
        # Subscribe to the message bus (built in Sprint 1!)
        bus.subscribe(Topic.PERCEPTION, lambda msg: setattr(self, 'obstacles', msg.obstacles))
        bus.subscribe(Topic.COSTMAP, lambda msg: setattr(self, 'costmap', msg))
        bus.subscribe(Topic.GLOBAL_PATH, lambda msg: setattr(self, 'global_path', msg))
        bus.subscribe(Topic.LOCAL_TRAJECTORY, lambda msg: setattr(self, 'local_traj', msg))

    def tick(self, state: VehicleState):
        self.state = state
        self.draw()

    def draw(self):
        if not self.state: return
        
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.set_title(f"PathSense Live | Speed: {self.state.twist.vx:.1f} m/s | Steering: {np.degrees(self.state.steering_angle):.1f}°", fontsize=14)
        
        # 1. Costmap Heatmap
        if self.costmap is not None and self.costmap.data.size > 0:
            extent = [self.costmap.origin_x, self.costmap.origin_x + self.costmap.width * self.costmap.resolution,
                      self.costmap.origin_y, self.costmap.origin_y + self.costmap.height * self.costmap.resolution]
            self.ax.imshow(self.costmap.data, cmap='Reds', origin='lower', extent=extent, alpha=0.3, vmin=0, vmax=1)

        # 2. Global Path (A*)
        if self.global_path and self.global_path.points:
            gx = [p.pose.x for p in self.global_path.points]
            gy = [p.pose.y for p in self.global_path.points]
            self.ax.plot(gx, gy, 'c--', linewidth=2, label='Global Path (A*)', alpha=0.8)

        # 3. DWA Candidates (The "Wow" Factor - the sampling cloud)
        if self.local_traj and hasattr(self.local_traj, 'debug_candidates'):
            for c in self.local_traj.debug_candidates:
                if hasattr(c, 'points'):
                    cx = [p[0] for p in c.points]
                    cy = [p[1] for p in c.points]
                    self.ax.plot(cx, cy, 'gray', linewidth=0.5, alpha=0.2)

        # 4. Winning Local Trajectory
        if self.local_traj and self.local_traj.points:
            lx = [p.pose.x for p in self.local_traj.points]
            ly = [p.pose.y for p in self.local_traj.points]
            self.ax.plot(lx, ly, 'g-', linewidth=3, label='Local Trajectory (DWA)', alpha=0.9)

        # 5. Obstacles (with velocity vectors for dynamic ones)
        for obs in self.obstacles:
            color = 'purple' if obs.is_dynamic else 'red'
            circle = plt.Circle((obs.pose.x, obs.pose.y), max(obs.length, obs.width)/2, 
                                color=color, alpha=0.7)
            self.ax.add_patch(circle)
            if obs.is_dynamic:
                self.ax.arrow(obs.pose.x, obs.pose.y, obs.velocity.vx * 1.5, obs.velocity.vy * 1.5, 
                              head_width=0.3, head_length=0.2, fc=color, ec=color)

        # 6. Ego Vehicle Polygon
        length, width = 4.2, 1.9
        cos_h = np.cos(self.state.pose.heading)
        sin_h = np.sin(self.state.pose.heading)
        corners = [(-length/2, -width/2), (length/2, -width/2), 
                   (length/2, width/2), (-length/2, width/2)]
        rotated = [(self.state.pose.x + x*cos_h - y*sin_h, self.state.pose.y + x*sin_h + y*cos_h) 
                   for x, y in corners]
        ego_poly = Polygon(rotated, closed=True, color='blue', alpha=0.8, label='Ego Vehicle')
        self.ax.add_patch(ego_poly)
        self.ax.arrow(self.state.pose.x, self.state.pose.y, 2.0 * cos_h, 2.0 * sin_h, 
                      head_width=0.5, head_length=0.3, fc='blue', ec='blue')

        self.ax.legend(loc='upper right')
        
        # Follow the car
        self.ax.set_xlim(self.state.pose.x - 20, self.state.pose.x + 20)
        self.ax.set_ylim(self.state.pose.y - 20, self.state.pose.y + 20)
        
        plt.pause(0.001)