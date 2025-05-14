import numpy as np
from collections import defaultdict
from drone import Drone, FireGrid


class MetricsLogger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = defaultdict(list)
        self.total_fire_cells = 0
        self.extinguished_fire_cells = 0
        self.initial_battery = {}
        self.final_battery = {}
        self.episode_length = 0
        self.collisions = 0
        self.idle_time = 0
        self.detection_time = {}

    def log_extinguished_fire(self, count):
        self.extinguished_fire_cells += count

    def log_extinguishing_time(self, count):
        self.metrics["extinguishing_time"].append(count / 5)

    def log_initial_battery(self, drones: list[Drone]):
        self.initial_battery = {d.id: d.battery_level for d in drones}

    def log_final_battery(self, drones: list[Drone]):
        self.final_battery = {d.id: d.battery_level for d in drones}

    def log_idle_time(self, drones: list[Drone]):
        idle_time = 0
        for drone in drones:
            if hasattr(drone, 'velocity') and np.linalg.norm(drone.velocity) == 0:
                idle_time += 1
        self.idle_time += idle_time / len(drones)

    def log_collision(self):
        self.collisions += 1

    def log_fire_detection_time(self, fire_grid: FireGrid):
        # print(set(fire_grid.fire_detection_times.keys()) - set(fire_grid.fire_start_times.keys()))
        for fire in fire_grid.fire_detection_times:
            if fire not in fire_grid.fire_start_times:
                continue
            self.detection_time[fire] = fire_grid.fire_detection_times[fire] - fire_grid.fire_start_times[fire]

    def calculate_metrics(self):
        energy_used = 0
        for agent_id in self.initial_battery:
            energy_used += self.initial_battery[agent_id] - self.final_battery.get(agent_id, 0)
        energy_used = energy_used / len(self.initial_battery) if self.initial_battery else 0
        
        print(f"total fire cells: {self.total_fire_cells}")
        # print(self.detection_time)
        return {
            "Fire Extinguished %": (self.extinguished_fire_cells / self.total_fire_cells) * 100 if self.total_fire_cells else 0,
            "Avg Extinguishing Time": np.sum(self.metrics["extinguishing_time"]) if self.metrics["extinguishing_time"] else 0,
            "Episode Length": self.episode_length,
            "Total Energy Used": energy_used,
            "Overlaps": self.collisions,
            "Avg Detection Time": np.mean(list(self.detection_time.values())) if self.detection_time else 0,
        }
