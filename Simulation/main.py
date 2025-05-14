import time
import math
import threading

from dronev import Drone
import numpy as np
import random
import matplotlib.pyplot as plt
import heapq
from dronekit import LocationGlobalRelative
import contextily as ctx
from pyproj import Transformer


# FIRE GRID CONFIGURATION
BASE_SPREAD_RATE = 0.05
BASE_DECAY_RATE = 0.04
WIND_STRENGTH = 0.2
SUPPRESSION_RATE = 2
FIRE_THRESHOLD = 0.1
MAX_INTENSITY = 1.0


class FireGrid:
    def __init__(self, grid_size, fire_count=10, fire_range=35, fire_offset=(0, 0), spread_rate=BASE_SPREAD_RATE, decay_rate=BASE_DECAY_RATE):
        self.width, self.height = grid_size
        self.grid_size = grid_size

        self.starting_fire_count = fire_count
        self.fire_range = fire_range
        self.fire_offset = fire_offset

        self.base_spread_rate = spread_rate
        self.decay_rate = decay_rate

        self.reset()

    def init_fires(self):
        for _ in range(self.starting_fire_count):
            x, y = self.fire_offset[0] + np.random.randint(-self.fire_range, self.fire_range), self.fire_offset[1] + np.random.randint(-self.fire_range, self.fire_range)
            self.grid[x, y] = np.random.uniform(0.5, MAX_INTENSITY)
            self.spread_rate[x, y] = np.random.uniform(0.1, self.base_spread_rate)

        self.total_fire_count = self.starting_fire_count

    def spread_fire(self):
        if random.random() < 0.05:
            self.wind_direction = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])

        new_fire = self.grid.copy()
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                if self.grid[i, j] > 0:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), 
                                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nx, ny = i + dx, j + dy
                        if self.grid[nx, ny] == 0:
                            spread_chance = self.spread_rate[i, j] * self.grid[i, j]
                            if (dx, dy) == self.wind_direction:
                                spread_chance += WIND_STRENGTH
                            if self.grid[i, j] > 0.5 and np.random.rand() < spread_chance:
                                new_fire[nx, ny] = self.grid[i, j] * np.random.uniform(0.25, 0.75)
                                self.spread_rate[nx, ny] = np.random.uniform(0.1, self.base_spread_rate)
                    
                    r = np.random.rand()
                    if r < 0.1:
                        new_fire[i, j] += np.random.uniform(0.03, 0.08)
                        new_fire[i, j] = min(new_fire[i, j], MAX_INTENSITY)
                    elif r < 0.3:
                        new_fire[i, j] = max(new_fire[i, j] - self.decay_rate, 0)

        self.total_fire_count += np.sum(new_fire > 0) - np.sum(self.grid > 0)
        self.grid = new_fire

    def is_empty(self):
        return len(np.where(self.grid > 0)[0]) == 0

    def suppress_fire(self, x, y):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    self.grid[nx, ny] = max(self.grid[nx, ny] - SUPPRESSION_RATE, 0)

        # self.grid[x, y] = max(self.grid[x, y] - SUPPRESSION_RATE, 0)
    
    def reset(self):
        self.grid = np.zeros(self.grid_size)
        self.spread_rate = np.zeros(self.grid_size)
        self.wind_direction = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        self.init_fires()


EVAPORATION_RATE = 0.95
SENSING_RADIUS = 10

class ACOSwarm:
    def __init__(self, drones: list[Drone], grid_size: tuple[int, int], fire_location: LocationGlobalRelative, pheromone_decay=0.1, pheromone_strength=1.0):
        self.drones = drones
        self.grid_size = grid_size
        self.pheromone_decay = pheromone_decay
        self.pheromone_strength = pheromone_strength
        self.pheromones = np.zeros(grid_size)

        # Search entire grid
        self.search_region = (0, 0, grid_size[0], grid_size[1])
        self.grid_center = fire_location

    def deposit_pheromone(self, position):
        x, y = position
        radius = SENSING_RADIUS // 2
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    distance = math.sqrt(dx**2 + dy**2)
                    if distance <= radius:
                        self.pheromones[nx, ny] += self.pheromone_strength / (1 + distance)

    def get_neighbors(self, pos):
        neighbours = []
        lx, ly, ux, uy = self.search_region
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            nx, ny = int(pos[0]) + dx * SENSING_RADIUS, int(pos[1]) + dy * SENSING_RADIUS
            if lx <= nx < ux and ly <= ny < uy:
                neighbours.append((nx, ny))
        return neighbours
    
    def sense_fire(self, fire_grid: FireGrid, position):
        x, y = position

        # Calculate bounds of the local grid within sensing radius while staying inside grid limits
        lx, ly = max(0, x - SENSING_RADIUS), max(0, y - SENSING_RADIUS)
        hx, hy = min(fire_grid.width, x + SENSING_RADIUS + 1), min(fire_grid.height, y + SENSING_RADIUS + 1)
        
        # Extract the local fire grid
        local_fire_grid = fire_grid.grid[lx:hx, ly:hy]
        
        # Calculate padding needed on each side if sensing radius exceeds grid boundaries
        pad_x_before = max(0, SENSING_RADIUS - x)
        pad_x_after = max(0, (x + SENSING_RADIUS + 1) - fire_grid.width)
        pad_y_before = max(0, SENSING_RADIUS - y)
        pad_y_after = max(0, (y + SENSING_RADIUS + 1) - fire_grid.height)

        # Pad the local grid with -1 (representing out-of-bounds) to handle edges
        local_fire_grid = np.pad(
            local_fire_grid,
            pad_width=((pad_x_before, pad_x_after), (pad_y_before, pad_y_after)),
            mode='constant',
            constant_values=-1
        )

        return local_fire_grid

    def run(self, fire_grid: FireGrid):
        targets_queue = []
        for drone in self.drones:
            x, y, alt = drone.get_relative_position(self.grid_center)
            x, y = int(x) + self.grid_size[0] // 2, int(y) + self.grid_size[1] // 2
            drone_position = x, y
            target_position = (self.search_region[0] + self.search_region[2]) / 2, (self.search_region[1] + self.search_region[3]) / 2
            
            local_fire_grid = self.sense_fire(fire_grid, drone_position)
            neighbours = self.get_neighbors(drone_position)
            self.deposit_pheromone(drone_position)

            hotspot_indices = np.where(local_fire_grid > 0)
            if len(hotspot_indices[0]) > 0:
                offset_x = max(0, x - SENSING_RADIUS)
                offset_y = max(0, y - SENSING_RADIUS)
                
                # Convert local hotspot coordinates to global coordinates
                visible_hotspots = np.array([
                    [hotspot_indices[0][i] + offset_x, hotspot_indices[1][i] + offset_y]
                    for i in range(len(hotspot_indices[0]))
                ])

                for h in visible_hotspots:
                    h_x, h_y = int(h[0]), int(h[1])
                    if 0 <= h_x < fire_grid.grid.shape[0] and 0 <= h_y < fire_grid.grid.shape[1]:
                        print(h_x, h_y)
                        intensity = fire_grid.grid[h_x, h_y]
                        targets_queue.append((float(-intensity), tuple(h)))

                if len(visible_hotspots) > 0:
                    intensities = fire_grid.grid[visible_hotspots[:, 0], visible_hotspots[:, 1]]
                    target_position = visible_hotspots[np.argmax(intensities)]
                    print(f"{drone.connection_string} go to fire loc {target_position}")
            elif neighbours:
                pheromone_levels = np.array([self.pheromones[n] for n in neighbours])
                pheromone_levels = np.maximum(pheromone_levels, 1e-6)

                radius = SENSING_RADIUS // 2
                fire_intensity = np.array([np.sum(fire_grid.grid[n[0]-radius:n[0]+radius, n[1]-radius:n[1]+radius]) for n in neighbours])
                fire_intensity = np.maximum(fire_intensity, 1e-6)

                probabilities = (1 / pheromone_levels) * fire_intensity # Modified ACO algorithm
                probabilities /= np.sum(probabilities)

                target_position = neighbours[np.random.choice(len(neighbours), p=probabilities)]
                print(f"{drone.connection_string} Curr Pos: {x, y} Next Pos: {target_position}")

            target_position_global = {
                "lat": self.grid_center.lat + (target_position[0] - fire_grid.width // 2) / 111320,
                "lon": self.grid_center.lon + (target_position[1] - fire_grid.height // 2) / (40075000 * math.cos(math.radians(self.grid_center.lat)) / 360),
                "alt": self.grid_center.alt
            }
            drone.move_to_position(target_position_global)
        # self.deposit_pheromone(target_position)
        self.pheromones *= EVAPORATION_RATE
        targets_found = set()
        while targets_queue:
            _, target = heapq.heappop(targets_queue)
            targets_found.add(target)
        return targets_found


class APFSwarm:
    def __init__(self, drones: list[Drone], fire_location: LocationGlobalRelative, ka=10.0, kr=0.0, d0=5.0):
        self.drones = drones
        self.ka = ka  # Attractive force coefficient
        self.kr = kr  # Repulsive force coefficient
        self.d0 = d0  # Threshold distance for repulsive force

        self.grid_center = fire_location

    def calculate_attractive_force(self, drone, target):
        pass

    def calculate_repulsive_force(self, drone, fire_grid):
        return np.zeros(2)

    def run(self, fire_grid: FireGrid, target_positions):
        target_positions = np.array(list(target_positions))
        print(f"Target positions: {target_positions}")
        grid_size = fire_grid.grid.shape

        if len(target_positions) == 0:
            return

        for drone in self.drones:
            x, y, alt = drone.get_relative_position(self.grid_center)
            x, y = int(x) + grid_size[0] // 2, int(y) + grid_size[1] // 2
            drone_pos = np.array([x, y])

            current_intensity = fire_grid.grid[x, y]
            if len(target_positions) == 0:
                continue

            nearest_target = min(target_positions, key=lambda t: np.linalg.norm(drone_pos - t))
            # attractive_force = self.calculate_attractive_force(drone, nearest_target)
            # repulsive_force = self.calculate_repulsive_force(drone, fire_grid)

            # resultant_force = attractive_force + repulsive_force

            next_position = nearest_target
            distance = np.linalg.norm(drone_pos - target_positions, axis=1)
            detected_fires = [fire_grid.grid[x + dx, y + dy] > 0 for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]]
            if current_intensity > 0 or any(detected_fires):
                print("Suppressing Fire")
                fire_grid.suppress_fire(x, y)
            else:
                target_position_global = {
                    "lat": self.grid_center.lat + (next_position[0] - fire_grid.width // 2) / 111320,
                    "lon": self.grid_center.lon + (next_position[1] - fire_grid.height // 2) / (40075000 * math.cos(math.radians(self.grid_center.lat)) / 360),
                    "alt": self.grid_center.alt
                }

                drone.move_to_position(target_position_global)


class SwarmSimulation:
    def __init__(self, drones: list[Drone], leader_index=0):
        self.drones = drones
        self.leader: Drone = drones[leader_index] if drones else None
        self.fire_grid = None
        self.acoswarm = None
        self.apf_swarm = None
        self.control_freq = 0.5
        self.max_steps = 200
        self.running = False
        self.step = 0

    def initialize_swarm(self, altitude):
        def initialize_drone(drone):
            drone.connect()
            drone.arm_and_takeoff(altitude)
            # drone.get_status()
            print(drone.position)

        threads = []
        for drone in self.drones:
            t = threading.Thread(target=initialize_drone, args=(drone,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        print("Swarm initialized.")

    def move_leader_to(self, target_position: LocationGlobalRelative):
        if self.leader:
            print(f"Leader {self.leader.connection_string} moving to {target_position}.")
            self.leader.move_to_position(target_position)

    def update_follower_positions(self):
        if not self.leader:
            print("No leader assigned.")
            return

        leader_position = self.leader.position
        for i, drone in enumerate(self.drones):
            if drone != self.leader:
                side = -1 if i % 2 == 0 else 1
                row = (i + 1) // 2

                offset_lat = row * 10 / 111320  # Convert 10 meters to latitude degrees
                offset_lon = side * row * 10 / (40075000 * math.cos(math.radians(leader_position["lat"])) / 360)  # Convert 10 meters to longitude degrees
                follower_target = {
                    "lat": leader_position["lat"] + offset_lat,
                    "lon": leader_position["lon"] + offset_lon,
                    "alt": leader_position["alt"]
                }

                print(f"Follower {drone.connection_string} moving to formation position {follower_target}.")
                drone.move_to_position(follower_target)

    def return_to_launch(self):
        for drone in self.drones:
            drone.return_to_launch()
        print("Swarm returning to launch.")

    def disconnect_all(self):
        for drone in self.drones:
            drone.disconnect()
        print("All drones disconnected.")

    def run_leader_follower(self, target_position: LocationGlobalRelative):
        self.move_leader_to(target_position)
        while True:
            self.leader.update_position()
            self.update_follower_positions()
            leader_dist = math.sqrt(
                ((self.leader.position["lat"] - target_position.lat) * 111320) ** 2 +
                ((self.leader.position["lon"] - target_position.lon) * 40075000 * math.cos(math.radians(self.leader.position["lat"])) / 360) ** 2
            )
            print(f"Leader reached: {leader_dist}")
            if leader_dist < 2:
                print("Leader has reached the target position.")
                break
            time.sleep(1)
        print("Swarm reached the destination.")

    def create_plot(self, fire_location: LocationGlobalRelative, fire_radius):
        grid_size = self.fire_grid.grid.shape
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        fire_x, fire_y = transformer.transform(fire_location.lon, fire_location.lat)
        display_size = fire_radius * 2

        # Create a figure and axis using subplots
        self.ax.clear()  # Clear the previous plot
        self.ax.set_title(f"Fire Grid and Drone Positions - Step {self.step + 1}")

        for drone in self.drones:
            # Convert latitude and longitude to meters relative to the fire grid's center
            drone_x, drone_y = transformer.transform(drone.position["lon"], drone.position["lat"])
            # drone_x = int((drone.position["lat"] - fire_location.lat) * 111320 + fire_radius)
            # drone_y = int((drone.position["lon"] - fire_location.lon) * 40075000 * math.cos(math.radians(fire_location.lat)) / 360 + fire_radius)
            self.ax.scatter(drone_x, drone_y, label=f"Drone {drone.connection_string}", s=100)
            circle = plt.Circle((drone_x, drone_y), 10, color='blue', fill=False, linestyle='--', linewidth=1.5)
            self.ax.add_patch(circle)

        try:
            ctx.add_basemap(self.ax, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            print(f"Error loading map tiles: {e}")
            print("If you don't have internet connection or contextily installed, you'll see a blank map.")

        self.ax.imshow(
            np.ma.masked_where(self.fire_grid.grid.T == 0, self.fire_grid.grid.T), 
            cmap="Reds", 
            origin="lower", 
            extent=(fire_x - fire_radius, fire_x + fire_radius, fire_y - fire_radius, fire_y + fire_radius), 
            alpha=0.6
        )

        self.ax.legend()
        self.ax.set_xlim(fire_x - display_size, fire_x + display_size)
        self.ax.set_ylim(fire_y - display_size, fire_y + display_size)
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")
        plt.pause(0.1)  # Pause to visualize each step

    # def create_plot(self, fire_location: LocationGlobalRelative, fire_radius):
    #     grid = self.fire_grid.grid  # shape (50, 50)
    #     grid_shape = grid.shape
    #     cell_size = fire_radius * 2 / grid_shape[0]  # estimate cell size in meters

    #     # Transformer for GPS to Web Mercator
    #     transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    #     fire_x, fire_y = transformer.transform(fire_location.lon, fire_location.lat)

    #     # Compute the extent of the grid in Web Mercator (centered on fire_x, fire_y)
    #     half_width = grid_shape[0] * cell_size / 2
    #     half_height = grid_shape[1] * cell_size / 2
    #     extent = (
    #         fire_x - half_width,
    #         fire_x + half_width,
    #         fire_y - half_height,
    #         fire_y + half_height
    #     )

    #     # Clear previous plot
    #     self.ax.clear()
    #     self.ax.set_title(f"Fire Grid and Drone Positions - Step {self.step + 1}")

    #     # Plot the fire heatmap grid


    #     # Plot each drone
    #     for drone in self.drones:
    #         drone_x, drone_y = transformer.transform(drone.position["lon"], drone.position["lat"])
    #         self.ax.scatter(drone_x, drone_y, label=f"Drone {drone.connection_string}", s=100, c='blue')
    #         circle = plt.Circle((drone_x, drone_y), 10, color='blue', fill=False, linestyle='--', linewidth=1.5)
    #         self.ax.add_patch(circle)

    #     # Add OpenStreetMap basemap
    #     try:
    #         ctx.add_basemap(self.ax, source=ctx.providers.OpenStreetMap.Mapnik, crs='EPSG:3857')
    #     except Exception as e:
    #         print(f"Error loading map tiles: {e}")

    #     self.ax.imshow(grid.T, cmap="hot", origin="lower", extent=extent, alpha=0.6)

    #     self.ax.legend()
    #     self.ax.set_xlim(fire_x - 2*fire_radius, fire_x + 2*fire_radius)
    #     self.ax.set_ylim(fire_y - 2*fire_radius, fire_y + 2*fire_radius)
    #     plt.pause(0.1)

    def run_swarm(self, fire_location: LocationGlobalRelative, fire_radius):
        self.running = True
        grid_size = (fire_radius * 2, fire_radius * 2)

        self.fire_grid = FireGrid(grid_size, fire_count=3, fire_range=fire_radius, fire_offset=(fire_radius, fire_radius))
        scout_drones = [d for d in self.drones if d.role == "scout"]
        suppressor_drones = [d for d in self.drones if d.role == "suppressor"]

        self.acoswarm = ACOSwarm(scout_drones, grid_size, fire_location)
        self.apf_swarm = APFSwarm(suppressor_drones, fire_location)

        self.initialize_swarm(fire_location.alt)
        self.run_leader_follower(fire_location)

        self.fig, self.ax = plt.subplots(figsize=(6, 6))  # Create a single figure and axis for all steps

        self.step = 0
        last_time = time.time()
        while True:
            if self.step > self.max_steps:
                break

            self.create_plot(fire_location, fire_radius)

            if self.fire_grid.is_empty():
                break

            if time.time() - last_time > self.control_freq:
                self.fire_grid.spread_fire()
                last_time = time.time()

                # Spread fire and run the swarm
                targets_found = self.acoswarm.run(self.fire_grid)
                self.apf_swarm.run(self.fire_grid, targets_found)

                print(f"Step {self.step + 1}: Targets Found: ", targets_found)
                for drone in self.drones:
                    drone.update_position()

                self.step += 1
        
        self.running = False
        # plt.close()
        


if __name__ == "__main__":
    np.random.seed(10)
    random.seed(10)

    drones = [
        Drone("udp:127.0.0.1:14551", role="scout"),
        Drone("udp:127.0.0.1:14561", role="suppressor"),
        Drone("tcp:127.0.0.1:14571", role="scout"),
    ]

    fire_location = LocationGlobalRelative(26.861406, 75.812826, 10)
    fire_radius = 25

    swarm = SwarmSimulation(drones)
    swarm.run_swarm(fire_location, fire_radius)
    swarm.return_to_launch()
    swarm.disconnect_all()
