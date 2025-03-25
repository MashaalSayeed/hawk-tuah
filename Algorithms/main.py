import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq
import random
import time

from constants import FIRE_RANGE, GRID_SIZE
from drone import Drone, FireGrid, MAX_INTENSITY
from pso_swarm import PSOSwarm, APFSwarm
from aco_swarm import ACOSwarm, ModifiedACOSwarm

# ENVIRONMENT CONFIGURATION
MAX_ITERATIONS = 1000
VISUALIZATION = True
NUM_ENVIRONMENTS = 100
VIS_CURRENT_SIMULATION = 3

# DRONE CONFIGURATION
NUM_LEADERS = 5
NUM_FOLLOWERS = 5
DRONE_SPEED = 1
SENSING_RADIUS = 20

random.seed(52)
np.random.seed(52)


def generate_sweep_path(search_region, step):
    lx, ly, ux, uy = search_region  # Extract search region boundaries
    path = []
    
    for row in range(lx, ux, step):
        if (row - lx) // step % 2 == 0:  # Alternate row direction for efficiency
            path.extend([(row, col) for col in range(ly, uy, step)])
        else:
            path.extend([(row, col) for col in range(uy - 1, ly - 1, -step)])
    
    return path


def assign_sweep_paths(drones, search_region, sensing_radius):
    paths = []
    lx, ly, ux, uy = search_region
    num_drones = len(drones)

    # Divide search region into horizontal strips
    step = max(2, sensing_radius // 2)  # Smaller steps to improve coverage
    strip_height = (ux - lx) // num_drones  

    for i, drone in enumerate(drones):
        start_row = lx + i * strip_height
        drone_region = (start_row, ly, min(start_row + strip_height, ux), uy)
        paths.append(generate_sweep_path(drone_region, step))  # Generate path for each drone's region

    return paths


class GridSweepSwarm:
    def __init__(self, drones: list[Drone]):
        self.drones = drones
        self.search_region = (0, 0, GRID_SIZE, GRID_SIZE)
        self.sweep_paths = []
        self.current_targets = {}

    def start(self, search_region):
        self.search_region = search_region
        self.sweep_paths = assign_sweep_paths(self.drones, self.search_region, SENSING_RADIUS // 2)
        self.current_targets = {drone: path[0] for drone, path in zip(self.drones, self.sweep_paths)}

    def run(self, fire_grid: FireGrid):
        targets_queue = []
        
        for drone in self.drones:
            target = self.current_targets[drone]
            visible_hotspots = drone.search_area(fire_grid)

            if np.linalg.norm(np.array(target) - np.array(drone.position)) < 2:
                next_index = self.sweep_paths[self.drones.index(drone)].index(target) + 1
                if next_index < len(self.sweep_paths[self.drones.index(drone)]):
                    self.current_targets[drone] = self.sweep_paths[self.drones.index(drone)][next_index]
                else:
                    self.current_targets[drone] = self.sweep_paths[self.drones.index(drone)][0]

            drone.move_to_target(self.drones, self.current_targets[drone], separation=0, attraction=1)

            # Detect fire in the area
            if len(visible_hotspots) > 0:
                for h in visible_hotspots:
                    intensity = fire_grid.grid[h[0], h[1]]
                    targets_queue.append((float(-intensity), tuple(h)))

        targets_found = []
        while targets_queue:
            _, target = heapq.heappop(targets_queue)
            targets_found.append(target)
        return targets_found


class Simulation:
    name = ""

    def __init__(self):
        self.leaders = [Drone(np.random.rand(2) * 20, 'leader') for _ in range(NUM_LEADERS)]
        self.followers = [Drone(np.random.rand(2) * 20, 'follower') for _ in range(NUM_FOLLOWERS)]
        self.iteration = 0

        self.fire_grid = FireGrid()

    def find_fire_center(self):
        fire_coords = np.argwhere(self.fire_grid.grid > 0)
        if len(fire_coords) == 0:
            return np.array([GRID_SIZE // 2, GRID_SIZE // 2])
        return fire_coords.mean(axis=0)

    def run_leader_follower(self):
        target = self.find_fire_center()
        leader_center = np.mean([lead.position for lead in self.leaders], axis=0)

        for leader in self.leaders:
            leader.move_to_target(self.leaders, target, separation=0.5)
            leader.update_position()

        for follower in self.followers:
            leader_positions = np.array([leader.position for leader in self.leaders])
            distances = np.linalg.norm(leader_positions - follower.position, axis=1)
            target_leader = leader_positions[np.argmin(distances)]
            
            follower.move_to_target(self.followers + self.leaders, target_leader)
            follower.update_position()
        
        return target, leader_center

    def run_iteration(self):
        raise NotImplementedError

    def run(self):
        if self.iteration % 2 == 0:
            self.fire_grid.spread_fire()

        self.run_iteration()

        if np.max(self.fire_grid.grid) <= 0:
            print(f"🔥 All fires extinguished! At iteration: {self.iteration}")
            return False
        
        self.iteration += 1
        return True


class Simulation1(Simulation):
    name = "Grid Sweep"

    def __init__(self):
        super().__init__()
        self.current_state = 'DEPLOY'
        self.sweep_swarm = GridSweepSwarm(self.leaders)
        self.pso_swarm = PSOSwarm(self.followers)
    
    def run_iteration(self):
        if self.current_state == 'DEPLOY':
            target, leader_center = self.run_leader_follower()
            if np.linalg.norm(target - leader_center) < 5:
                self.current_state = 'DETECT'

                fire_range = 2 * FIRE_RANGE
                search_region = (target[0] - fire_range, target[1] - fire_range, target[0] + fire_range, target[1] + fire_range)
                search_region = tuple(int(x) for x in search_region)
                self.sweep_swarm.start(search_region)
        
        elif self.current_state == 'DETECT':
            targets_found = self.sweep_swarm.run(self.fire_grid)
            self.pso_swarm.run(self.fire_grid, targets_found)

            for leader in self.leaders:
                leader.update_position()

            # for target in targets_found:
            #     ax.plot(*target, 'gx', markersize=5, label='Target')

            for follower in self.followers:
                follower.update_position()
                follower.suppress_fire(self.fire_grid)


class Simulation2(Simulation):
    name = "ACO"

    def __init__(self):
        super().__init__()
        self.current_state = 'DEPLOY'

        self.fire_grid = FireGrid()
        self.aco_swarm = ACOSwarm(self.leaders)
        self.pso_swarm = PSOSwarm(self.followers)
    
    def run_iteration(self):
        if self.current_state == 'DEPLOY':
            target, leader_center = self.run_leader_follower()

            if np.linalg.norm(target - leader_center) < 5:
                self.current_state = 'DETECT'

                fire_range = 2 * FIRE_RANGE
                search_region = (target[0] - fire_range, target[1] - fire_range, target[0] + fire_range, target[1] + fire_range)
                search_region = tuple(int(x) for x in search_region)
                self.aco_swarm.start(search_region)
        
        elif self.current_state == 'DETECT':
            # targets_found = self.aco_swarm.run(self.fire_grid
            targets_found = self.aco_swarm.run(self.fire_grid)
            self.pso_swarm.run(self.fire_grid, targets_found)

            for leader in self.leaders:
                leader.update_position()

            # for target in targets_found:
            #     ax.plot(*target, 'gx', markersize=5, label='Target')

            for follower in self.followers:
                follower.update_position()
                follower.suppress_fire(self.fire_grid)
    
class Simulation3(Simulation):
    name = "Modified ACO"

    def __init__(self):
        super().__init__()
        self.current_state = 'DEPLOY'
        self.aco_swarm = ModifiedACOSwarm(self.leaders)
        self.pso_swarm = PSOSwarm(self.followers)
    
    def run_iteration(self):
        if self.current_state == 'DEPLOY':
            target, leader_center = self.run_leader_follower()

            if np.linalg.norm(target - leader_center) < 5:
                self.current_state = 'DETECT'

                fire_range = 2 * FIRE_RANGE
                search_region = (target[0] - fire_range, target[1] - fire_range, target[0] + fire_range, target[1] + fire_range)
                search_region = tuple(int(x) for x in search_region)
                self.aco_swarm.start(search_region)
        
        elif self.current_state == 'DETECT':
            targets_found = self.aco_swarm.run(self.fire_grid)
            self.pso_swarm.run(self.fire_grid, targets_found)

            for leader in self.leaders:
                leader.update_position()

            # for target in targets_found:
            #     ax.plot(*target, 'gx', markersize=5, label='Target')

            for follower in self.followers:
                follower.update_position()
                follower.suppress_fire(self.fire_grid)


class Simulation4(Simulation):
    name = "Modified ACO + APF"

    def __init__(self):
        super().__init__()
        self.current_state = 'DEPLOY'
        self.aco_swarm = ModifiedACOSwarm(self.leaders)
        self.apf_swarm = APFSwarm(self.followers)
    
    def run_iteration(self):
        if self.current_state == 'DEPLOY':
            target, leader_center = self.run_leader_follower()

            if np.linalg.norm(target - leader_center) < 5:
                self.current_state = 'DETECT'

                fire_range = 2 * FIRE_RANGE
                search_region = (target[0] - fire_range, target[1] - fire_range, target[0] + fire_range, target[1] + fire_range)
                search_region = tuple(int(x) for x in search_region)
                self.aco_swarm.start(search_region)
        
        elif self.current_state == 'DETECT':
            targets_found = self.aco_swarm.run(self.fire_grid)
            self.apf_swarm.run(self.fire_grid, targets_found)

            for leader in self.leaders:
                leader.update_position()

            # for target in targets_found:
            #     ax.plot(*target, 'gx', markersize=5, label='Target')

            for follower in self.followers:
                follower.update_position()
                follower.suppress_fire(self.fire_grid)


def run_visualization(sim: Simulation):
    fig, ax = plt.subplots(figsize=(8,8))

    img = ax.imshow(sim.fire_grid.grid, cmap='hot', interpolation='nearest', vmin=0, vmax=MAX_INTENSITY)

    plt.colorbar(img)
    plt.title("Fire Suppression Simulation")

    def update(frame):
        ax.clear()
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        
        # Show fire grid as base layer
        fire_img = ax.imshow(sim.fire_grid.grid.T, cmap='hot', interpolation='nearest', vmin=0, vmax=MAX_INTENSITY)
        # pheromone_img = ax.imshow(sim.aco_swarm.pheromones.T, origin='lower', cmap='cool', interpolation='nearest', vmin=0, vmax=5, alpha=0.5)

        # Run simulation
        running = sim.run()

        # Plot leaders and followers
        for leader in sim.leaders:
            ax.plot(*leader.position, 'ro', markersize=4, label='Leader' if frame == 0 else "")

        for follower in sim.followers:
            ax.plot(*follower.position, 'bo', markersize=2, label='Follower' if frame == 0 else "")

        # Stop when all fires are extinguished
        if not running:
            ani.event_source.stop()

        # Return both images to update them
        return fire_img
    
    ani = FuncAnimation(fig, update, frames=300, interval=10)
    plt.legend()
    plt.show()

    for drone in sim.leaders + sim.followers:
        print(f"Drone at {drone.position} ({drone.role}) consumed {drone.energy_consumed} units of energy")



simulations: list[Simulation] = [Simulation1, Simulation2, Simulation3, Simulation4]
def main():
    if VISUALIZATION:
        sim = simulations[VIS_CURRENT_SIMULATION]
        print(f"Running {sim.name} simulation...")
        simulation = sim()
        run_visualization(simulation)
        return
    
    sim_results = []
    for sim in simulations:
        print(f"Running {sim.name} simulation...")
        results = []
        for i in range(NUM_ENVIRONMENTS):
            simulation: Simulation = sim()
            iteration = 0

            while iteration < MAX_ITERATIONS:
                running = simulation.run()
                iteration += 1
                if not running:
                    break

            avg_follower_energy = np.mean([f.energy_consumed for f in simulation.followers])
            avg_leader_energy = np.mean([l.energy_consumed for l in simulation.leaders])
            results.append((running, iteration, avg_follower_energy, avg_leader_energy))

            print(f"Environment {i+1}: {iteration} iterations, Leader Energy: {avg_leader_energy}, Follower Energy: {avg_follower_energy}")
            if running:
                print("Simulation ended before all fires were extinguished")

        print("\n\nFinal Results:")
        total_completed = sum([not r[0] for r in results])
        avg_iterations = np.mean([r[1] for r in results])
        avg_leader_energy = np.mean([r[3] for r in results])
        avg_follower_energy = np.mean([r[2] for r in results])

        sim_results.append((total_completed, avg_iterations, avg_follower_energy, avg_leader_energy))
        print(f"[Completed: {total_completed}/{NUM_ENVIRONMENTS}] Average Iterations: {avg_iterations}, Average Leader Energy: {avg_leader_energy}, Average Follower Energy: {avg_follower_energy}")

    with open("results.csv", "w") as f:
        f.write("Simulation,Completed,Iterations,Leader Energy,Follower Energy\n")
        for sim, result in zip(simulations, sim_results):
            f.write(f"{sim.name},{result[0]},{result[1]},{result[3]},{result[2]}\n")



if __name__ == "__main__":
    main()