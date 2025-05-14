from drone import Drone, SuppressorDrone, FireGrid
from env import FireFightingEnv
from simple_env import FireFightingEnvSimple
import numpy as np
import random
import time
import heapq

W = 1
C1 = 0.1
C2 = 1.0

EVAPORATION_RATE = 0.9
DEPOSIT_AMOUNT = 1.0

SENSING_RADIUS = 10

class PSOSwarm:
    def __init__(self, drones: list[Drone]):
        self.drones = drones
        self.best_score_idx = 0
        self.best_position = self.drones[self.best_score_idx].position
        self.w = W
        self.c1 = C1
        self.c2 = C2

        self.drone_bests = {
            drone.id: {
                "position": drone.position.copy(),
                "score": 0
            }
            for drone in self.drones
        }

    def run(self, fire_grid: FireGrid, target_positions):
        target_positions = np.array(list(target_positions))
        extinguish_drones = []
        if len(target_positions) > 0:
            self.best_position = target_positions[0]

        for drone in self.drones:
            current_intensity = fire_grid.grid[int(drone.position[0]), int(drone.position[1])]
            if len(target_positions) == 0:
                drone.velocity = np.zeros(2)
                continue

            r1, r2 = random.random(), random.random()
            drone.velocity = (
                self.w * drone.velocity + 
                self.c1 * r1 * (self.drone_bests[drone.id]["position"] - drone.position) + 
                self.c2 * r2 * (self.best_position - drone.position)
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

            if current_intensity > 0 or distance.min() < 1:
                extinguish_drones.append(drone.id)
                self.drone_bests[drone.id] = {
                    "position": drone.position.copy(),
                    "score": np.inf
                }

        best_scores = np.array([self.drone_bests[drone.id]["score"] for drone in self.drones])
        self.best_score_idx = np.argmin(best_scores)
        self.best_position = self.drones[self.best_score_idx].position
        self.w *= 0.99
        self.c1 *= 0.99
        return extinguish_drones
    

class APFSwarm:
    def __init__(self, drones: list[SuppressorDrone], ka=10.0, kr=0.0, d0=10.0):
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

        # # Repulsive force from obstacles (e.g., fire)
        # pos_x, pos_y = int(drone.position[0]), int(drone.position[1])
        # radius = 2
        # if 0 <= pos_x < fire_grid.grid.shape[0] and 0 <= pos_y < fire_grid.grid.shape[1]:
        #     fire_intensity = np.sum(fire_grid.grid[pos_x - radius:pos_x + radius, pos_y - radius:pos_y + radius])
        #     if fire_intensity > 0:  # Strong repulsive force if near fire zone
        #         repulsive_force += self.kr * (1 / (fire_intensity + 1e-6))

        # Repulsion from other drones
        for other_drone in self.drones:
            if other_drone != drone:
                distance = np.linalg.norm(drone.position - other_drone.position) + 1e-6
                if distance < self.d0:
                    repulsive_force += self.kr * (1 / distance - 1 / self.d0) * (drone.position - other_drone.position)

        return repulsive_force
    

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
    def __init__(self, drones: list[Drone], grid_size=(100, 100)):
        self.drones = drones
        self.grid_size = grid_size

        self.search_region = (0, 0, grid_size[0], grid_size[1])
        self.sweep_paths = assign_sweep_paths(self.drones, self.search_region, SENSING_RADIUS // 2)
        self.current_targets = {drone: path[0] for drone, path in zip(self.drones, self.sweep_paths)}
    
    def start(self, search_region):
        self.search_region = search_region
        self.sweep_paths = assign_sweep_paths(self.drones, self.search_region, SENSING_RADIUS // 2)
        self.current_targets = {drone: path[0] for drone, path in zip(self.drones, self.sweep_paths)}

    def run(self, fire_grid: FireGrid):
        targets_queue = []
        for drone in self.drones:
            target = self.current_targets[drone]
            local_fire_grid = drone.sense_fire(fire_grid)
            hotspot_indices = np.where(local_fire_grid > 0)

            if np.linalg.norm(np.array(target) - np.array(drone.position)) < 2:
                next_index = self.sweep_paths[self.drones.index(drone)].index(target) + 1
                if next_index < len(self.sweep_paths[self.drones.index(drone)]):
                    self.current_targets[drone] = self.sweep_paths[self.drones.index(drone)][next_index]
                else:
                    self.current_targets[drone] = self.sweep_paths[self.drones.index(drone)][0]

            target_position = self.current_targets[drone]

            # Detect fire in the area
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
                    intensity = fire_grid.grid[h[0], h[1]]
                    targets_queue.append((float(-intensity), tuple(h)))

            
            # Move towards the target
            direction = target_position - drone.position
            distance = np.linalg.norm(direction)
            if distance > 1e-3:
                velocity = (direction / distance)
                drone.velocity = velocity
            else:
                drone.velocity = np.zeros(2)

        targets_found = []
        while targets_queue:
            _, target = heapq.heappop(targets_queue)
            targets_found.append(target)
        return targets_found

class ModifiedACOSwarm:
    def __init__(self, drones: list[Drone], grid_size=(100, 100), modified=True):
        self.pheromones = np.zeros(grid_size)
        self.drones = drones
        self.modified = modified
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

                if self.modified:
                    probabilities = (1 / pheromone_levels) * fire_intensity # Modified ACO algorithm
                else:
                    probabilities = pheromone_levels * fire_intensity
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
    def __init__(self, env: FireFightingEnv, detection_algorithm="mod_aco", suppression_algorithm="apf"):
        super().__init__()
        self.env = env
        grid_size = self.env.fire_grid.grid.shape

        if detection_algorithm == "aco":
            self.aco_swarm = ModifiedACOSwarm(self.env.scouts, grid_size=grid_size, modified=False)
        elif detection_algorithm == "mod_aco":
            self.aco_swarm = ModifiedACOSwarm(self.env.scouts, grid_size=grid_size)
        elif detection_algorithm == "grid":
            self.aco_swarm = GridSweepSwarm(self.env.scouts, grid_size=grid_size)
        
        if suppression_algorithm == "apf":
            self.apf_swarm = APFSwarm(self.env.suppressors)
        elif suppression_algorithm == "pso":
            self.apf_swarm = PSOSwarm(self.env.suppressors)
        # self.aco_swarm = ModifiedACOSwarm(self.env.scouts, grid_size=grid_size)
        # self.apf_swarm = PSOSwarm(self.env.suppressors)

        self.aco_swarm.start((SENSING_RADIUS // 2, SENSING_RADIUS // 2, grid_size[0] - SENSING_RADIUS // 2, grid_size[1] - SENSING_RADIUS // 2))
    
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


# if __name__ == "__main__":
#     np.random.seed(1)
#     random.seed(1)

#     # Initialize the environment
#     env = FireFightingEnv(grid_size=(80, 80), num_scouts=5, num_suppresors=5, render_mode="human")
#     simulation = Simulation(env, detection_algorithm="mod_aco", suppression_algorithm="apf")
    
#     # Reset the environment
#     obs, infos = simulation.reset(seed=42)
#     import matplotlib.pyplot as plt
    
#     for i in range(1000):
#         obs, rewards, terminations, truncations, infos = simulation.run_iteration()
        
#         # Render the environment
#         # env.render()
#         # time.sleep(0.5)
#         # if i == 40:
#         #     for drone in simulation.aco_swarm.drones:
#         #         plt.scatter(drone.position[0], drone.position[1], c='blue', label=f'Drone {drone.id}')
#         #         print(drone.velocity)
#         #         plt.arrow(drone.position[0], drone.position[1], drone.velocity[0] * 5, drone.velocity[1] * 5, 
#         #                   color='blue', head_width=1, head_length=1, length_includes_head=True)
                
#         #     plt.imshow(simulation.aco_swarm.pheromones, cmap='Greens', interpolation='nearest')
#         #     plt.colorbar(label='Pheromone Intensity')
#         #     plt.title('Pheromone Map')
#         #     plt.xlabel('X')
#         #     plt.ylabel('Y')
#         #     plt.show()
#         # print(f"Step {i+1}:")
        
#         # Check if all agents are done
#         if all(terminations.values()):
#             print("All agents terminated!")
#             break

#     print(f"Simulation finished after {i+1} steps.")
#     print(env.metrics_logger.calculate_metrics())
            

#     # Plot the pheromone map of the ACO swarm

#     env.close()


if __name__ == "__main__":

    np.random.seed(1)
    random.seed(1)
    
    # Initialize the environment
    env = FireFightingEnv(grid_size=(80, 80), num_scouts=5, num_suppresors=5, render_mode="human")
    simulation = Simulation(env, detection_algorithm="aco", suppression_algorithm="pso")
    
    # Reset the environment
    import matplotlib.pyplot as plt
    

    results = {"Fire Extinguished %": [], "Avg Extinguishing Time": [], "Episode Length": [], "Total Energy Used": [], "Overlaps": [], "Avg Detection Time": [], "Mission Success Rate": []}

    for iter in range(100):
        obs, infos = simulation.reset(seed=1)
        for i in range(1000):
            obs, rewards, terminations, truncations, infos = simulation.run_iteration()
            
            # Render the environment
            # env.render()
            # time.sleep(0.5)
            # if i == 40:
            #     for drone in simulation.aco_swarm.drones:
            #         plt.scatter(drone.position[0], drone.position[1], c='blue', label=f'Drone {drone.id}')
            #         print(drone.velocity)
            #         plt.arrow(drone.position[0], drone.position[1], drone.velocity[0] * 5, drone.velocity[1] * 5, 
            #                   color='blue', head_width=1, head_length=1, length_includes_head=True)
                    
            #     plt.imshow(simulation.aco_swarm.pheromones, cmap='Greens', interpolation='nearest')
            #     plt.colorbar(label='Pheromone Intensity')
            #     plt.title('Pheromone Map')
            #     plt.xlabel('X')
            #     plt.ylabel('Y')
            #     plt.show()
            # print(f"Step {i+1}:")

            # if iter == 74:
            #     env.render()
            #     time.sleep(0.1)
            
            # Check if all agents are done
            if all(terminations.values()):
                print("All agents terminated!")
                break

        # print(env.metrics_logger.calculate_metrics())
        metrics = env.metrics_logger.calculate_metrics()
        for key, value in metrics.items():
            results[key].append(value)

        completed = not env.fire_grid.get_fire_positions() or metrics['Fire Extinguished %'] > 95
        results["Mission Success Rate"].append(completed)
        print(f"Simulation {iter} finished after {i+1} steps. {completed}, {metrics['Fire Extinguished %']}%")

    # # Print the average results
    log = []
    for key, values in results.items():
        # print(key, values)
        print(f"{key}: {np.mean(values):.2f} ± {np.std(values):.2f}")
        log.append(np.mean(values))
    
    with open("results.txt", "a") as f:
        log = ",".join([f"{x:.2f}" for x in log])
        f.write(f"{log}\n")

        


            

    # Plot the pheromone map of the ACO swarm

    env.close()