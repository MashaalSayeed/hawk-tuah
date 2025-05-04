import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# Constants
GRID_SIZE = 100
BASE_SPREAD_RATE = 0.1
BASE_DECAY_RATE = 0.04
WIND_STRENGTH = 0.2
DRONE_COUNT = 10
SENSING_RADIUS = 10
SUPPRESSION_RATE = 0.1
FIRE_THRESHOLD = 0.1
MAX_INTENSITY = 1.0

# Initialize fire grid with random intensities
fire_grid = np.zeros((GRID_SIZE, GRID_SIZE))
spread_rate_grid = np.zeros((GRID_SIZE, GRID_SIZE))

# Start fire at random locations with random intensities and spread rates
for _ in range(5): 
    x, y = np.random.randint(0, GRID_SIZE, size=2)
    fire_grid[x, y] = np.random.uniform(0.5, MAX_INTENSITY)
    spread_rate_grid[x, y] = np.random.uniform(0.1, BASE_SPREAD_RATE)

# Initialize drones at random locations
drone_positions = [tuple(np.random.randint(0, GRID_SIZE, size=2)) for _ in range(DRONE_COUNT)]

# Random wind direction (can change over time)
wind_direction = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])

def spread_fire():
    global fire_grid, spread_rate_grid, wind_direction
    
    # Occasionally change wind direction
    if random.random() < 0.05:
        wind_direction = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])

    new_fire = fire_grid.copy()

    for i in range(1, GRID_SIZE - 1):
        for j in range(1, GRID_SIZE - 1):
            if fire_grid[i, j] > 0:
                # Spread fire based on neighboring spread rates
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = i + dx, j + dy
                    if fire_grid[nx, ny] == 0:
                        spread_chance = spread_rate_grid[i, j] * fire_grid[i, j]

                        # Wind effect increases chance in wind direction
                        if (dx, dy) == wind_direction:
                            spread_chance += WIND_STRENGTH

                        # Spread to neighbor with a chance based on spread rate
                        if fire_grid[i, j] > 0.5 and np.random.rand() < spread_chance:
                            new_fire[nx, ny] = fire_grid[i, j] * np.random.uniform(0.25, 0.75)
                            spread_rate_grid[nx, ny] = np.random.uniform(0.1, BASE_SPREAD_RATE)

                # Fire intensity can increase randomly (simulating wind gusts)
                if np.random.rand() < 0.1:
                    new_fire[i, j] += np.random.uniform(0.05, 0.1)
                    new_fire[i, j] = min(new_fire[i, j], MAX_INTENSITY)  # Cap at max intensity

                # Decay over time
                r = np.random.rand()
                if r < 0.25:
                    new_fire[i, j] = max(new_fire[i, j] - BASE_DECAY_RATE, 0)

    fire_grid = new_fire

def suppress_fire():
    global fire_grid
    for x, y in drone_positions:
        for dx in range(-SENSING_RADIUS, SENSING_RADIUS + 1):
            for dy in range(-SENSING_RADIUS, SENSING_RADIUS + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    if fire_grid[nx, ny] > 0:
                        fire_grid[nx, ny] -= SUPPRESSION_RATE
                        if fire_grid[nx, ny] < 0:
                            fire_grid[nx, ny] = 0

def update_drones():
    global drone_positions
    new_positions = []

    for x, y in drone_positions:
        # Define neighbors (within the grid)
        neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)] 
                     if 0 <= x + dx < GRID_SIZE and 0 <= y + dy < GRID_SIZE]

        # ACO-like behavior: Move toward high-intensity fire if sensed
        if any(fire_grid[nx, ny] > FIRE_THRESHOLD for nx, ny in neighbors):
            next_pos = max(neighbors, key=lambda pos: fire_grid[pos[0], pos[1]])
        else:
            # Explore using pheromone-based behavior
            pheromone_levels = np.array([fire_grid[nx, ny] for nx, ny in neighbors])
            weights = 1.0 / (pheromone_levels + 1e-3)
            probabilities = weights / np.sum(weights)
            next_pos = random.choices(neighbors, weights=probabilities, k=1)[0]
        
        new_positions.append(next_pos)

    drone_positions = new_positions

# Setup plot
fig, ax = plt.subplots()
img = ax.imshow(fire_grid, cmap='hot', interpolation='nearest', vmin=0, vmax=MAX_INTENSITY)
# scat_drones = ax.scatter([], [], c='blue', s=30, label="Drones")
plt.colorbar(img)
plt.title("Fire Suppression Simulation")

# Display wind direction as an arrow
# wind_arrow = ax.arrow(
#     GRID_SIZE // 2, GRID_SIZE // 2, wind_direction[0] * 5, wind_direction[1] * 5,
#     head_width=2, head_length=3, fc='white', ec='white'
# )

iteration = 0
def update(frame):
    global iteration
    iteration += 1

    # Spread fire
    spread_fire()

    # Update drones
    # update_drones()

    # Suppress fire
    # suppress_fire()

    # Update visualization
    img.set_data(fire_grid)
    # scat_drones.set_offsets(drone_positions)

    # Update wind arrow
    # wind_arrow.remove()
    # wind_arrow = ax.arrow(
    #     GRID_SIZE // 2, GRID_SIZE // 2, wind_direction[0] * 5, wind_direction[1] * 5,
    #     head_width=2, head_length=3, fc='white', ec='white'
    # )

    # Stop when all fires are extinguished
    if np.max(fire_grid) <= 0:
        print("🔥 All fires extinguished! ✅")
        ani.event_source.stop()

    print(iteration)

    return img

# Run animation
ani = FuncAnimation(fig, update, frames=300, interval=100)
plt.legend()
plt.show()
