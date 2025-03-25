import numpy as np
import random

from constants import GRID_SIZE, STARTING_FIRE_COUNT, FIRE_OFFSET, FIRE_RANGE


# DRONE CONFIGURATION
DRONE_SPEED = 1
SENSING_RADIUS = 20

# FIRE GRID CONFIGURATION
BASE_SPREAD_RATE = 0.05
BASE_DECAY_RATE = 0.04
WIND_STRENGTH = 0.2
SUPPRESSION_RATE = 0.25
FIRE_THRESHOLD = 0.1
MAX_INTENSITY = 1.0

# Energy Constants
IDLE_ENERGY = 0.1
MOVE_ENERGY = 0.5
SUPPRESS_ENERGY = 0.75
COMMUNICATION_ENERGY = 0.1
SENSOR_ENERGY = 0.1

# Artificial Potential Field Constants
ATTRACTION_STRENGTH = 0.05
SEPARATION_STRENGTH = 0.1
SEPARATION_DISTANCE = 20


class FireGrid:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.spread_rate = np.zeros((GRID_SIZE, GRID_SIZE))
        self.wind_direction = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        self.init_fires()

    def init_fires(self):
        for _ in range(STARTING_FIRE_COUNT):
            x, y = FIRE_OFFSET[0] + np.random.randint(-FIRE_RANGE, FIRE_RANGE), FIRE_OFFSET[1] + np.random.randint(-FIRE_RANGE, FIRE_RANGE)
            self.grid[x, y] = np.random.uniform(0.5, MAX_INTENSITY)
            self.spread_rate[x, y] = np.random.uniform(0.1, BASE_SPREAD_RATE)

    def spread_fire(self):
        if random.random() < 0.05:
            self.wind_direction = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])

        new_fire = self.grid.copy()

        for i in range(1, GRID_SIZE - 1):
            for j in range(1, GRID_SIZE - 1):
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
                                self.spread_rate[nx, ny] = np.random.uniform(0.1, BASE_SPREAD_RATE)
                    
                    r = np.random.rand()
                    if r < 0.1:
                        new_fire[i, j] += np.random.uniform(0.03, 0.08)
                        new_fire[i, j] = min(new_fire[i, j], MAX_INTENSITY)
                    elif r < 0.3:
                        new_fire[i, j] = max(new_fire[i, j] - BASE_DECAY_RATE, 0)

        self.grid = new_fire

    def suppress_fire(self, x, y):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    self.grid[nx, ny] = max(self.grid[nx, ny] - SUPPRESSION_RATE, 0)

        # self.grid[x, y] = max(self.grid[x, y] - SUPPRESSION_RATE, 0)


class Drone:
    def __init__(self, position, role):
        self.position = position
        self.velocity = np.zeros(2)
        self.role = role

        self.best_position = position
        self.best_score = np.inf
        self.best_scores = [float('inf'), float('inf')]
        self.energy_consumed = 0

    def move_to_target(self, neighbours, target, attraction=ATTRACTION_STRENGTH, separation=SEPARATION_STRENGTH):
        # Use Leader-Follower Algorithm
        separation_force = np.zeros(2)
        attraction_force = np.zeros(2)

        for neighbour in neighbours:
            diff = self.position - neighbour.position
            dist = np.linalg.norm(diff)
            if dist < SEPARATION_DISTANCE and dist > 0:
                separation_force += separation * (diff / dist) / dist

        diff = np.array([target[0] - self.position[0], target[1] - self.position[1]])
        dist = np.linalg.norm(diff)
        if dist > 0:
            attraction_force += attraction * diff / dist

        self.velocity = separation_force + attraction_force
        # self.velocity = self.velocity / np.linalg.norm(self.velocity) * DRONE_SPEED
    
    def update_position(self):
        self.energy_consumed += COMMUNICATION_ENERGY

        distance_traveled = 0
        if np.linalg.norm(self.velocity) == 0:
            self.velocity = np.zeros(2)
            self.energy_consumed += IDLE_ENERGY
        else:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * DRONE_SPEED
            distance_traveled = np.linalg.norm(self.velocity)
            self.energy_consumed += distance_traveled * MOVE_ENERGY
        self.position += self.velocity

    def suppress_fire(self, fire_grid):
        x, y = int(self.position[0]), int(self.position[1])
        fire_grid.suppress_fire(x, y)
        self.energy_consumed += SUPPRESS_ENERGY

    def search_area(self, fire_grid: FireGrid):
        x, y = self.position  # Avoid unnecessary conversion to int
        x, y = int(x), int(y)  # Convert only once
        min_x, max_x = max(0, x - SENSING_RADIUS), min(fire_grid.grid.shape[0], x + SENSING_RADIUS + 1)
        min_y, max_y = max(0, y - SENSING_RADIUS), min(fire_grid.grid.shape[1], y + SENSING_RADIUS + 1)

        subgrid = fire_grid.grid[min_x:max_x, min_y:max_y]
        local_hotspots = np.argwhere(subgrid > FIRE_THRESHOLD)
        visible_hotspots = local_hotspots + [min_x, min_y]

        self.energy_consumed += SENSOR_ENERGY
        
        return visible_hotspots