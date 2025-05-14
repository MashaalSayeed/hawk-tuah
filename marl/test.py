from drone import Drone, FireGrid
from swarm import PSOSwarm, APFSwarm


class Simulation:
    def __init__(self, fire_grid_size=(100, 100), num_drones=5, algorithm="PSO"):
        self.fire_grid = FireGrid(fire_grid_size)
        self.drones = [Drone(i, role="suppressor", grid_size=fire_grid_size) for i in range(num_drones)]
        
        if algorithm == "PSO":
            self.algo = PSOSwarm(self.drones)
        elif algorithm == "APF":
            self.algo = APFSwarm(self.drones)

        self.reset()

    def run_simulation(self, steps=100):
        self.fire_grid.spread_fire()
        target_positions = self.fire_grid.get_fire_positions()
        for step in range(steps):
            extinguishing_drones = self.algo.run(self.fire_grid, target_positions)
            print(f"Step {step + 1}/{steps}: {len(extinguishing_drones)} drones extinguishing fire. Total fire count: {self.fire_grid.total_fire_count}")
    
    def reset(self):
        self.fire_grid.reset()
        for drone in self.drones:
            drone.reset()
        

if __name__ == "__main__":
    sim = Simulation(fire_grid_size=(80, 80), num_drones=5, algorithm="APF")
    sim.run_simulation(steps=100)