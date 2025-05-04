import heapq
import numpy as np

from drone import Drone, FireGrid, SENSING_RADIUS
GRID_SIZE = 200

# ACO CONFIGURATION
EVAPORATION_RATE = 0.95
DEPOSIT_AMOUNT = 1.0


class ACOSwarm:
    def __init__(self, drones: list[Drone]):
        self.pheromones = np.zeros((GRID_SIZE, GRID_SIZE))
        self.drones = drones
        self.search_region = (0, 0, GRID_SIZE, GRID_SIZE)
    
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
            visible_hotspots = drone.search_area(fire_grid)
            self.deposit_pheromone(drone.position)

            if len(visible_hotspots) > 0:
                for h in visible_hotspots:
                    intensity = fire_grid.grid[h[0], h[1]]
                    targets_queue.append((float(-intensity), tuple(h)))

                target = visible_hotspots[np.argmax(fire_grid.grid[visible_hotspots[:, 0], visible_hotspots[:, 1]])]
                if np.linalg.norm(target - drone.position) < 2:
                    drone.velocity = np.zeros(2)
                else:
                    drone.move_to_target(self.drones, target, attraction=0.05, separation=0.1)
            else:
                neighbours = self.get_neighbors(drone.position)
                if not neighbours:
                    search_center = (self.search_region[0] + self.search_region[2]) / 2, (self.search_region[1] + self.search_region[3]) / 2
                    drone.move_to_target(self.drones, search_center, separation=1)
                    continue

                pheromone_levels = np.array([self.pheromones[n] for n in neighbours])
                pheromone_levels = np.maximum(pheromone_levels, 1e-6)

                radius = SENSING_RADIUS // 2
                fire_intensity = np.array([np.sum(fire_grid.grid[n[0]-radius:n[0]+radius, n[1]-radius:n[1]+radius]) for n in neighbours])
                fire_intensity = np.maximum(fire_intensity, 1e-6)

                # probabilities = (1 / pheromone_levels) * fire_intensity # Modified ACO algorithm
                probabilities = pheromone_levels * fire_intensity
                probabilities /= np.sum(probabilities)

                next_pos = neighbours[np.random.choice(len(neighbours), p=probabilities)]
                drone.move_to_target(self.drones, next_pos, separation=1)
                self.deposit_pheromone(next_pos)

        self.pheromones *= EVAPORATION_RATE
        targets_found = []
        while targets_queue:
            _, target = heapq.heappop(targets_queue)
            targets_found.append(target)
        return targets_found
    

class ModifiedACOSwarm:
    def __init__(self, drones: list[Drone]):
        self.pheromones = np.zeros((GRID_SIZE, GRID_SIZE))
        self.drones = drones
        self.search_region = (0, 0, GRID_SIZE, GRID_SIZE)
    
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
            visible_hotspots = drone.search_area(fire_grid)
            self.deposit_pheromone(drone.position)

            if len(visible_hotspots) > 0:
                for h in visible_hotspots:
                    intensity = fire_grid.grid[h[0], h[1]]
                    targets_queue.append((float(-intensity), tuple(h)))

                target = visible_hotspots[np.argmax(fire_grid.grid[visible_hotspots[:, 0], visible_hotspots[:, 1]])]
                if np.linalg.norm(target - drone.position) < 2:
                    drone.velocity = np.zeros(2)
                else:
                    drone.move_to_target(self.drones, target, attraction=0.05, separation=0.1)
            else:
                neighbours = self.get_neighbors(drone.position)
                if not neighbours:
                    search_center = (self.search_region[0] + self.search_region[2]) / 2, (self.search_region[1] + self.search_region[3]) / 2
                    drone.move_to_target(self.drones, search_center, separation=1)
                    continue

                pheromone_levels = np.array([self.pheromones[n] for n in neighbours])
                pheromone_levels = np.maximum(pheromone_levels, 1e-6)

                radius = SENSING_RADIUS // 2
                fire_intensity = np.array([np.sum(fire_grid.grid[n[0]-radius:n[0]+radius, n[1]-radius:n[1]+radius]) for n in neighbours])
                fire_intensity = np.maximum(fire_intensity, 1e-6)

                probabilities = (1 / pheromone_levels) * fire_intensity # Modified ACO algorithm
                probabilities /= np.sum(probabilities)

                next_pos = neighbours[np.random.choice(len(neighbours), p=probabilities)]
                drone.move_to_target(self.drones, next_pos, separation=1)
                self.deposit_pheromone(next_pos)

        self.pheromones *= EVAPORATION_RATE
        targets_found = []
        while targets_queue:
            _, target = heapq.heappop(targets_queue)
            targets_found.append(target)
        return targets_found