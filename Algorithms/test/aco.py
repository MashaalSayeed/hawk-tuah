import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# -------------------------------
# Parameters and Initialization
# -------------------------------
grid_size = (100, 100)
num_drones = 10
num_hotspots = 3
drone_speed = 1  # drones move 1 unit per timestep
sensing_radius = 10  # drones can sense hotspots within this distance
evaporation_rate = 0.99  # pheromone evaporation factor
deposit_amount = 1.0  # pheromone deposited per visit

# Create a pheromone map (initially all zeros)
pheromone = np.zeros(grid_size)

# Initialize drones at random positions
drones = []
for _ in range(num_drones):
    pos = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))
    drones.append(pos)

# Initialize fire hotspots at random positions
hotspots = []
for _ in range(num_hotspots):
    pos = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))
    hotspots.append(pos)

# -------------------------------
# Helper Functions
# -------------------------------
def get_neighbors(pos):
    """Return valid neighboring positions (8-connected grid)."""
    x, y = pos
    moves = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1)]
    neighbors = []
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]:
            neighbors.append((nx, ny))
    return neighbors

def euclidean_distance(a, b):
    """Return the Euclidean distance between two points a and b."""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def choose_next_position(pos):
    """
    Choose the next position for a drone. 
    If a hotspot is within sensing_radius, move towards the closest hotspot.
    Otherwise, use the modified ACO rule (favoring low-pheromone cells).
    """
    # Check if any hotspot is within sensing radius
    visible_hotspots = [h for h in hotspots if euclidean_distance(pos, h) <= sensing_radius]
    
    neighbors = get_neighbors(pos)
    
    if visible_hotspots:
        # Find the closest hotspot
        target = min(visible_hotspots, key=lambda h: euclidean_distance(pos, h))
        # From the available neighbors, choose one that minimizes the distance to the target
        distances = [euclidean_distance(n, target) for n in neighbors]
        min_distance = min(distances)
        # Get all neighbors that are as close as the minimum distance (tie-breaking)
        best_neighbors = [n for n, d in zip(neighbors, distances) if d == min_distance]
        next_pos = random.choice(best_neighbors)
    else:
        # Use modified ACO: favor cells with lower pheromone levels
        pheromone_levels = np.array([pheromone[n] for n in neighbors])
        weights = 1.0 / (pheromone_levels + 1e-3)  # add small constant to avoid division by zero
        probabilities = weights / np.sum(weights)
        next_pos = random.choices(neighbors, weights=probabilities, k=1)[0]
    return next_pos

def simulation_step():
    """Perform one simulation step for all drones."""
    global drones, hotspots, pheromone
    new_positions = []
    for pos in drones:
        next_pos = choose_next_position(pos)
        new_positions.append(next_pos)
        # Deposit pheromone at the new position.
        pheromone[next_pos] += deposit_amount
        # If a hotspot is detected at the new position, remove it.
        if next_pos in hotspots:
            hotspots.remove(next_pos)
    drones = new_positions
    # Evaporate pheromones across the grid.
    pheromone *= evaporation_rate

# -------------------------------
# Visualization Setup
# -------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(pheromone.T, origin='lower', cmap='hot', interpolation='nearest', vmin=0, vmax=5)
scat_drones = ax.scatter([pos[0] for pos in drones], [pos[1] for pos in drones], 
                         c='blue', s=50, label='Drones')
scat_hotspots = ax.scatter([pos[0] for pos in hotspots], [pos[1] for pos in hotspots],
                           c='green', marker='*', s=200, label='Hotspots')
ax.legend(loc='upper right')
ax.set_title("Drone Swarm Search with Modified ACO and Sensing Radius")

# -------------------------------
# Animation Update Function
# -------------------------------
def update(frame):
    simulation_step()
    im.set_data(pheromone.T)
    scat_drones.set_offsets(np.array(drones))
    # Update hotspots display; if none remain, clear the marker.
    if hotspots:
        scat_hotspots.set_offsets(np.array(hotspots))
    else:
        scat_hotspots.set_offsets(np.empty((0, 2)))
        print(f"Hotspots found in {frame} iterations!")
        exit(0)

    ax.set_title(f"Iteration {frame} - Remaining Hotspots: {len(hotspots)}")
    return im, scat_drones, scat_hotspots

# Create the animation (stops once all hotspots are found or after max iterations)
ani = animation.FuncAnimation(fig, update, frames=500, interval=100, blit=False, repeat=False)
plt.show()
