from drone import Drone, SuppressorDrone, FireGrid
from env import FireFightingEnv
from simple_env import FireFightingEnvSimple
import numpy as np
import random
import time
import heapq

W = 1
C1 = 0.75
C2 = 0.75

EVAPORATION_RATE = 0.95
DEPOSIT_AMOUNT = 1.0

SENSING_RADIUS = 10

class PSOSwarm:
    def __init__(self, drones: list[Drone]):
        self.drones = drones
        self.best_score_idx = 0
        self.best_position = self.drones[self.best_score_idx].position

        self.drone_bests = {
            drone.id: {
                "position": drone.position.copy(),
                "score": 0
            }
            for drone in self.drones
        }

    def run(self, fire_grid: FireGrid, target_positions):
        target_positions = np.array(list(target_positions))
        if len(target_positions) > 0:
            self.best_position = target_positions[0]

        for drone in self.drones:
            if len(target_positions) == 0:
                drone.velocity = np.zeros(2)
                continue

            r1, r2 = random.random(), random.random()
            drone.velocity = (
                W * drone.velocity + 
                C1 * r1 * (self.drone_bests[drone.id]["position"] - drone.position) + 
                C2 * r2 * (self.best_position - drone.position)
            )

            pos = int(drone.position[0]), int(drone.position[1])
            radius = 2
            fire_intensity = np.sum(fire_grid.grid[pos[0]-radius:pos[0]+radius, pos[1]-radius:pos[1]+radius])
            distance = np.linalg.norm(drone.position - target_positions, axis=1)
            score = np.min(distance) / (fire_intensity + 1e-6)
            if score < self.drone_bests[drone.id]["score"]:
                self.drone_bests[drone.id] = {
                    "position": drone.position.copy(),
                    "score": score
                }

            if distance.min() < 1:
                self.drone_bests[drone.id] = {
                    "position": drone.position.copy(),
                    "score": np.inf
                }

        best_scores = np.array([self.drone_bests[drone.id]["score"] for drone in self.drones])
        self.best_score_idx = np.argmin(best_scores)
        self.best_position = self.drones[self.best_score_idx].position
        return True
    

class APFSwarm:
    def __init__(self, drones: list[SuppressorDrone], ka=10.0, kr=0.0, d0=5.0):
        self.drones = drones
        self.ka = ka  # Attractive force coefficient
        self.kr = kr  # Repulsive force coefficient
        self.d0 = d0  # Threshold distance for repulsive force

    def run(self, fire_grid, target_positions):
        target_positions = np.array(list(target_positions))
        extinguish_drones = []

        for drone in self.drones:
            current_intensity = fire_grid.grid[int(drone.position[0]), int(drone.position[1])]
            if len(target_positions) == 0:
                drone.velocity = np.zeros(2)
                continue

            # Calculate Attractive Force
            nearest_target = min(target_positions, key=lambda t: np.linalg.norm(drone.position - t))
            attractive_force = self.calculate_attractive_force(drone, nearest_target)

            # Calculate Repulsive Force from obstacles (fire zones) and drones
            repulsive_force = self.calculate_repulsive_force(drone, fire_grid)

            # Resultant force is the sum of attractive and repulsive forces
            resultant_force = attractive_force + repulsive_force

            # Update drone velocity and position based on resultant force
            drone.velocity = resultant_force
            # Stop drone if near target or extinguish fire at the target
            distance = np.linalg.norm(drone.position - target_positions, axis=1)
            # print(f"Drone {drone.id} - velocity: {drone.velocity}, distance: {distance}, fire intensity: {current_intensity}")
            if current_intensity > 0 or distance.min() < 1:
                extinguish_drones.append(drone.id)
                continue

        return extinguish_drones

    def calculate_attractive_force(self, drone: Drone, target_position):
        """ Calculate attractive force pulling drone toward the nearest target. """
        direction = target_position - drone.position
        magnitude = self.ka * np.linalg.norm(direction)
        return (direction / np.linalg.norm(direction)) * magnitude if np.linalg.norm(direction) > 0 else np.zeros(2)

    def calculate_repulsive_force(self, drone: Drone, fire_grid: FireGrid):
        """ Calculate repulsive force from fire zones and nearby drones. """
        repulsive_force = np.zeros(2)

        # Repulsive force from obstacles (e.g., fire)
        pos_x, pos_y = int(drone.position[0]), int(drone.position[1])
        radius = 2
        if 0 <= pos_x < fire_grid.grid.shape[0] and 0 <= pos_y < fire_grid.grid.shape[1]:
            fire_intensity = np.sum(fire_grid.grid[pos_x - radius:pos_x + radius, pos_y - radius:pos_y + radius])
            if fire_intensity > 0:  # Strong repulsive force if near fire zone
                repulsive_force += self.kr * (1 / (fire_intensity + 1e-6))

        # Repulsion from other drones
        for other_drone in self.drones:
            if other_drone != drone:
                distance = np.linalg.norm(drone.position - other_drone.position) + 1e-6
                if distance < self.d0:
                    repulsive_force += self.kr * (1 / distance - 1 / self.d0) * (drone.position - other_drone.position)

        return repulsive_force
    

class ModifiedACOSwarm:
    def __init__(self, drones: list[Drone], grid_size=(100, 100)):
        self.pheromones = np.zeros(grid_size)
        self.drones = drones
        self.search_region = (0, 0, grid_size[0], grid_size[1])
    
    def get_neighbors(self, pos):
        neighbours = []
        lx, ly, ux, uy = self.search_region
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            nx, ny = int(pos[0]) + dx * SENSING_RADIUS, int(pos[1]) + dy * SENSING_RADIUS
            if lx <= nx < ux and ly <= ny < uy:
                neighbours.append((nx, ny))
        return neighbours
    
    def deposit_pheromone(self, pos):
        pos = int(pos[0]), int(pos[1])
        self.pheromones[pos] = min(self.pheromones[pos] + DEPOSIT_AMOUNT, 5)

    def start(self, search_region):
        self.search_region = search_region
    
    def run(self, fire_grid: FireGrid):
        targets_queue = []
        for drone in self.drones:
            target_position = (self.search_region[0] + self.search_region[2]) / 2, (self.search_region[1] + self.search_region[3]) / 2
            local_fire_grid = drone.sense_fire(fire_grid)
            neighbours = self.get_neighbors(drone.position)
            self.deposit_pheromone(drone.position)

            hotspot_indices = np.where(local_fire_grid > 0)
            # print(drone.id, hotspot_indices)
            if len(hotspot_indices[0]) > 0:
                x, y = drone.position
                x, y = int(x), int(y)

                offset_x = max(0, x - drone.sensing_radius)
                offset_y = max(0, y - drone.sensing_radius)
                
                # Convert local hotspot coordinates to global coordinates
                visible_hotspots = np.array([
                    [hotspot_indices[0][i] + offset_x, hotspot_indices[1][i] + offset_y]
                    for i in range(len(hotspot_indices[0]))
                ])

                for h in visible_hotspots:
                    h_x, h_y = int(h[0]), int(h[1])
                    if 0 <= h_x < fire_grid.grid.shape[0] and 0 <= h_y < fire_grid.grid.shape[1] and fire_grid.grid[h_x, h_y] > 0:
                        intensity = fire_grid.grid[h_x, h_y]
                        targets_queue.append((float(-intensity), tuple(h)))

    
                # Select the target with the highest fire intensity
                if len(visible_hotspots) > 0:
                    intensities = fire_grid.grid[visible_hotspots[:, 0], visible_hotspots[:, 1]]
                    target_position = visible_hotspots[np.argmax(intensities)]
            elif neighbours:
                pheromone_levels = np.array([self.pheromones[n] for n in neighbours])
                pheromone_levels = np.maximum(pheromone_levels, 1e-6)

                radius = SENSING_RADIUS // 2
                fire_intensity = np.array([np.sum(fire_grid.grid[n[0]-radius:n[0]+radius, n[1]-radius:n[1]+radius]) for n in neighbours])
                fire_intensity = np.maximum(fire_intensity, 1e-6)

                probabilities = (1 / pheromone_levels) * fire_intensity # Modified ACO algorithm
                probabilities /= np.sum(probabilities)

                target_position = neighbours[np.random.choice(len(neighbours), p=probabilities)]
            
            self.deposit_pheromone(target_position)
            direction = target_position - drone.position
            distance = np.linalg.norm(direction)
            if distance > 1e-3:  # avoid division by zero
                velocity = (direction / distance)
                drone.velocity = velocity
            else:
                drone.velocity = np.zeros(2)

        self.pheromones *= EVAPORATION_RATE
        targets_found = []
        while targets_queue:
            _, target = heapq.heappop(targets_queue)
            targets_found.append(target)
        return targets_found


class Simulation:
    def __init__(self, env: FireFightingEnv):
        super().__init__()
        self.env = env
        grid_size = self.env.fire_grid.grid.shape

        self.aco_swarm = ModifiedACOSwarm(self.env.scouts, grid_size=grid_size)
        self.apf_swarm = APFSwarm(self.env.suppressors)

        self.aco_swarm.start((0, 0, grid_size[0], grid_size[1]))
    
    def run_iteration(self):
        targets_found = self.aco_swarm.run(self.env.fire_grid)
        extinguish_drones = self.apf_swarm.run(self.env.fire_grid, targets_found)

        actions = {}
        for agent in self.env.agent_map.values():
            vx, vy = agent.velocity
            if vx == 0 and vy == 0:
                action = 8
            else:
                angle = np.arctan2(vy, vx)
                # Convert angle to action (8 directions)
                action = int(((angle + np.pi) / (2 * np.pi / 8)) % 8)
            if agent.role == "suppressor" and agent.id in extinguish_drones:
                action = 9  # Fire extinguishing action
            actions[agent.id] = action

        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return obs, rewards, terminations, truncations, infos

    def reset(self, seed=None):
        return self.env.reset(seed)


if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)

    # Initialize the environment
    env = FireFightingEnv(grid_size=(80, 80), num_scouts=5, num_suppresors=5, render_mode="human")
    simulation = Simulation(env)
    
    # Reset the environment
    obs, infos = simulation.reset(seed=42)
    
    # Run the simulation for 100 steps
    for i in range(1000):
        obs, rewards, terminations, truncations, infos = simulation.run_iteration()
        
        # Render the environment
        env.render()
        time.sleep(0.5)
        
        # Check if all agents are done
        if all(terminations.values()):
            print("All agents terminated!")
            break

    print(f"Simulation finished after {i+1} steps.")
    print(env.metrics_logger.calculate_metrics())
            
    env.close()