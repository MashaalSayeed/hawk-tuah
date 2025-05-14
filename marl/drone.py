import random
import numpy as np
from gymnasium import spaces

# DRONE CONFIGURATION
SENSING_RADIUS = 10
DRONE_SPEED = 1
COMMUNICATION_ENERGY = 0.0001  # Small constant energy drain for communication (per step)
IDLE_ENERGY = 0.001  # Energy consumed while idle (per step)
MOVE_ENERGY = 0.003  # Energy consumed per unit distance traveled (when moving)
SUPPRESSION_ENERGY = 0.05

# FIRE GRID CONFIGURATION
BASE_SPREAD_RATE = 0.05
BASE_DECAY_RATE = 0.04
WIND_STRENGTH = 0.2
SUPPRESSION_RATE = 0.5
FIRE_THRESHOLD = 0.1
MAX_INTENSITY = 1.0


class FireGrid:
    def __init__(self, grid_size, fire_count=10, fire_range=30, fire_offset=(0, 0), spread_rate=BASE_SPREAD_RATE, decay_rate=BASE_DECAY_RATE):
        self.width, self.height = grid_size
        self.grid_size = grid_size

        self.starting_fire_count = fire_count
        self.fire_range = fire_range
        self.fire_offset = grid_size[0] // 2 + fire_offset[0], grid_size[1] // 2 + fire_offset[1]

        self.base_spread_rate = spread_rate
        self.decay_rate = decay_rate

        self.step = 0

        self.fire_start_times = {}  # Dictionary to track first start times for each fire
        self.fire_detection_times = {}  # Dictionary to track first detection times for each fire

        self.reset()

    def init_fires(self):
        for _ in range(self.starting_fire_count):
            x, y = self.fire_offset[0] + np.random.randint(-self.fire_range, self.fire_range), self.fire_offset[1] + np.random.randint(-self.fire_range, self.fire_range)
            self.grid[x, y] = np.random.uniform(0.5, MAX_INTENSITY)
            self.spread_rate[x, y] = np.random.uniform(0.1, self.base_spread_rate)
            self.fire_start_times[(x, y)] = self.step  # Track the start time of the fire

        self.total_fire_count = self.starting_fire_count

    def spread_fire(self):
        self.step += 1
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
                                self.fire_start_times[(nx, ny)] = self.step
                    
                    r = np.random.rand()
                    if r < 0.1:
                        new_fire[i, j] += np.random.uniform(0.03, 0.08)
                        new_fire[i, j] = min(new_fire[i, j], MAX_INTENSITY)
                    elif r < 0.3:
                        new_fire[i, j] = max(new_fire[i, j] - self.decay_rate, 0)

        self.total_fire_count += np.sum(new_fire > 0) - np.sum(self.grid > 0)
        self.grid = new_fire

    def suppress_fire(self, x, y):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    self.grid[nx, ny] = max(self.grid[nx, ny] - SUPPRESSION_RATE, 0)

    def get_fire_positions(self):
        fire_positions = np.argwhere(self.grid > FIRE_THRESHOLD)
        fire_positions = [(x, y) for x, y in fire_positions if 0 <= x < self.width and 0 <= y < self.height]
        return fire_positions

    def detect_fires(self, fire_positions):
        for fire_position in fire_positions:
            x, y = int(fire_position[0]), int(fire_position[1])
            if self.grid[x, y] > FIRE_THRESHOLD and (x, y) not in self.fire_detection_times:
                self.fire_detection_times[(x, y)] = self.step

    def reset(self):
        self.grid = np.zeros(self.grid_size)
        self.spread_rate = np.zeros(self.grid_size)
        self.wind_direction = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        self.fire_detection_times = {}  # Reset detection times
        self.fire_start_times = {}  # Reset start times
        self.step = 0
        self.total_fire_count = 0
        self.init_fires()


class Drone:
    def __init__(self, i, role, grid_size, sensing_radius=SENSING_RADIUS):
        self.id = f"{role}_{i}"
        self.role = role
        self.grid_size = grid_size
        self.sensing_radius = sensing_radius
        self.battery_level = 1.0
        self.fire_extinguishing = False

        self.reset()

    def sense_fire(self, fire_grid: FireGrid):
        x, y = self.position
        x, y = int(x), int(y)

        # Calculate bounds of the local grid within sensing radius while staying inside grid limits
        lx, ly = max(0, x - self.sensing_radius), max(0, y - self.sensing_radius)
        hx, hy = min(fire_grid.width, x + self.sensing_radius + 1), min(fire_grid.height, y + self.sensing_radius + 1)
        
        # Extract the local fire grid
        local_fire_grid = fire_grid.grid[lx:hx, ly:hy]
        
        # Calculate padding needed on each side if sensing radius exceeds grid boundaries
        pad_x_before = max(0, self.sensing_radius - x)
        pad_x_after = max(0, (x + self.sensing_radius + 1) - fire_grid.width)
        pad_y_before = max(0, self.sensing_radius - y)
        pad_y_after = max(0, (y + self.sensing_radius + 1) - fire_grid.height)

        # Pad the local grid with -1 (representing out-of-bounds) to handle edges
        local_fire_grid = np.pad(
            local_fire_grid,
            pad_width=((pad_x_before, pad_x_after), (pad_y_before, pad_y_after)),
            mode='constant',
            constant_values=-1
        )

        return local_fire_grid

    def update_position(self):
        energy_consumed = COMMUNICATION_ENERGY
        if np.linalg.norm(self.velocity) == 0:
            # self.velocity = np.zeros(2)
            energy_consumed = IDLE_ENERGY
        else:
            # self.velocity = self.velocity / np.linalg.norm(self.velocity) * DRONE_SPEED
            distance_traveled = np.linalg.norm(self.velocity)
            energy_consumed = distance_traveled * MOVE_ENERGY


        if self.fire_extinguishing:
            energy_consumed += SUPPRESSION_ENERGY

        self.position += self.velocity
        self.position = np.clip(self.position, 0, self.grid_size[0] - 1)

        self.battery_level -= energy_consumed
        self.battery_level = max(0, self.battery_level)


    def suppress_fire(self, fire_grid: FireGrid):
        x, y = self.position
        x, y = int(x), int(y)
        fire_grid.suppress_fire(x, y)

    def step(self, action):   
        moves = {
            0: np.array([-1, 0]),      # Left
            1: np.array([-1, -1]),     # Down-Left
            2: np.array([0, -1]),      # Down
            3: np.array([1, -1]),      # Down-Right
            4: np.array([1, 0]),       # Right
            5: np.array([1, 1]),       # Up-Right
            6: np.array([0, 1]),       # Up
            7: np.array([-1, 1]),      # Up-Left
            8: np.array([0, 0]),
        }

        if action == 9:
            self.fire_extinguishing = True
        else:
            self.fire_extinguishing = False

        # Update the velocity based on the action
        self.velocity = moves.get(action, np.array([0, 0]))  # Default to idle if action is invalid
        self.update_position()  
    
    def reset(self):
        self.position = np.random.randint(0, self.grid_size[0], size=2)
        self.velocity = np.zeros(2)
        self.battery_level = 1.0

    def observation_space(self):
        fire_grid_shape = (self.sensing_radius * 2 + 1, self.sensing_radius * 2 + 1)
        low = np.array([0, 0, 0.0] + [-1] * (fire_grid_shape[0] * fire_grid_shape[1]), dtype=np.float32)
        high = np.array(
            [1.0, 1.0, 1.0] + [1.0] * (fire_grid_shape[0] * fire_grid_shape[1]),
            dtype=np.float32
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)


class ScoutDrone(Drone):
    def __init__(self, i, grid_size):
        super(ScoutDrone, self).__init__(i, "scout", grid_size)
    
 

class SuppressorDrone(Drone):
    def __init__(self, i, grid_size):
        super(SuppressorDrone, self).__init__(i, "suppressor", grid_size)

    def step(self, action):
        super(SuppressorDrone, self).step(action)
        if action == 9:
            self.fire_extinguishing = True
        else:
            self.fire_extinguishing = False

    def observation_space(self):
        # Position (x, y), battery level, fire intensity at current location
        low = np.array([0, 0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)
